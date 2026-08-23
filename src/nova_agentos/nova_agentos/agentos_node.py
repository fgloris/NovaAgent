#!/usr/bin/env python3
"""agentos_node: LLM 驱动的技能调度。接收任务 → 选择技能 → 下发 → 校验/重规划。"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from nova_interfaces.msg import SkillCommand, SkillResult, TaskState
from nova_interfaces.srv import ListSkills, LoadSkills
from nova_agentos.context_builder import (
    IDENTITY,
    SYSTEM_TEMPLATE,
    build_skills_summary,
    build_user_message,
)
from nova_agentos.llm_client import LLMClient, LLMError

STATE_IDLE = "idle"
STATE_PLANNING = "planning"
STATE_WAITING = "waiting"
STATE_FINISHED = "finished"


def parse_decision(text: str) -> dict[str, Any]:
    """从 LLM 输出里提取第一个 JSON 对象。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class AgentOSNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_agentos")

        self.declare_parameter("llm_config_path", "")
        self.declare_parameter("max_replan", 2)
        self.declare_parameter("decision_timeout_sec", 30.0)
        self.declare_parameter("auto_start_on_instruction", False)

        llm_config_path = str(self.get_parameter("llm_config_path").value)
        self.max_replan = int(self.get_parameter("max_replan").value)
        self.decision_timeout_sec = float(self.get_parameter("decision_timeout_sec").value)
        self.auto_start = bool(self.get_parameter("auto_start_on_instruction").value)

        config = self._load_llm_config(llm_config_path)
        self.llm = LLMClient(config, logger=self.get_logger())

        self.list_client = self.create_client(ListSkills, "/nova/skills/list")
        self.load_client = self.create_client(LoadSkills, "/nova/skills/load")

        self.cmd_pub = self.create_publisher(SkillCommand, "/nova/skills/cmd", 10)
        self.task_pub = self.create_publisher(TaskState, "/nova/agentos/task", 10)

        self.create_subscription(String, "/nova/robocasa/state", self.state_callback, 10)
        self.create_subscription(String, "/nova/agentos/goal", self.goal_callback, 10)
        self.create_subscription(SkillResult, "/nova/skills/result", self.result_callback, 10)

        self.obs: dict[str, Any] | None = None
        self.state = STATE_IDLE
        self.instruction = ""
        self.task_id = ""
        self.history: list[dict] = []
        self.skill_details: dict[str, str] = {}
        self.replan_count = 0
        self.waiting_deadline = 0.0
        self.pending_command: SkillCommand | None = None
        self.final_summary = ""

        self.create_timer(0.2, self.tick)
        self.get_logger().info("nova_agentos ready")

    def _load_llm_config(self, path: str) -> dict:
        import yaml

        if path:
            config_path = Path(path).expanduser()
        else:
            try:
                from ament_index_python.packages import get_package_share_directory

                config_path = Path(get_package_share_directory("nova_agentos")) / "config" / "llm.yaml"
            except Exception:
                config_path = Path(__file__).resolve().parents[1] / "config" / "llm.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"llm config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ---- 输入 ----
    def state_callback(self, msg: String) -> None:
        try:
            self.obs = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"bad state payload: {exc}")

        if (
            self.auto_start
            and self.state == STATE_IDLE
            and self.obs
            and self.obs.get("instruction")
        ):
            self.start_task(self.obs["instruction"])

    def goal_callback(self, msg: String) -> None:
        self.start_task(msg.data)

    def start_task(self, instruction: str) -> None:
        if not instruction.strip():
            self.get_logger().warn("empty goal, ignored")
            return
        self.instruction = instruction
        self.task_id = uuid.uuid4().hex[:8]
        self.history = []
        self.skill_details = {}
        self.replan_count = 0
        self.final_summary = ""
        self.state = STATE_PLANNING
        self.get_logger().info(f"task {self.task_id} started: {instruction}")
        self._publish_task()

    # ---- 主循环 ----
    def tick(self) -> None:
        if self.state == STATE_IDLE:
            return

        # 环境已结束(done),直接收尾
        if self.state in (STATE_PLANNING, STATE_WAITING) and self.obs and self.obs.get("done"):
            self.finish_task("environment episode terminated")

        if self.state == STATE_PLANNING:
            self._plan_step()
        elif (
            self.state == STATE_WAITING
            and self.pending_command is not None
            and time.monotonic() > self.waiting_deadline
        ):
            self._on_result(SkillResult(), status="timeout", success=False, info="decision timeout")

    def _plan_step(self) -> None:
        try:
            skills = self._call_list_skills()
            system = SYSTEM_TEMPLATE.format(
                identity=IDENTITY,
                skills_summary=build_skills_summary(skills),
            )
            user = build_user_message(self.instruction, self.obs, self.history, self.skill_details)
            text = self.llm.chat(system, [{"role": "user", "content": user}])
            decision = parse_decision(text)
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"LLM decision failed: {exc}")
            self._fail_and_replan(f"LLM decision failed: {exc}")
            return

        if decision.get("done"):
            self.finish_task(str(decision.get("summary", "task finished")))
            return

        skill_id = str(decision.get("skill_id", "")).strip()
        if not skill_id:
            self._fail_and_replan("LLM returned no skill_id")
            return
        if not any(s["name"] == skill_id and s["available"] for s in skills):
            self._fail_and_replan(f"LLM selected unavailable skill: {skill_id}")
            return

        params = decision.get("params") or {}
        goal = str(decision.get("goal", ""))
        self._load_skill_details(skill_id)

        command = SkillCommand()
        command.task_id = self.task_id
        command.skill_id = skill_id
        command.goal = goal
        command.params_json = json.dumps(params, ensure_ascii=False)
        self.pending_command = command
        self.state = STATE_WAITING
        self.waiting_deadline = time.monotonic() + self.decision_timeout_sec
        self.cmd_pub.publish(command)
        self.get_logger().info(f"dispatch skill={skill_id} goal={goal} params={params}")
        self._publish_task()

    def _on_result(self, result: SkillResult, *, status: str | None = None, success: bool = False, info: str = "") -> None:
        if status is None:
            status = result.status
            success = result.success
            info = result.info
        if self.state != STATE_WAITING:
            return
        command = self.pending_command
        self.pending_command = None
        if command is None:
            return
        self.history.append(
            {"skill_id": command.skill_id, "goal": command.goal, "status": status, "info": info}
        )
        self.get_logger().info(f"result for {command.skill_id}: {status} {info}")
        if success:
            self.state = STATE_PLANNING
            self._publish_task()
        else:
            self._fail_and_replan(info)

    def result_callback(self, result: SkillResult) -> None:
        if result.task_id != self.task_id:
            return
        self._on_result(result)

    def _fail_and_replan(self, reason: str) -> None:
        self.replan_count += 1
        if self.replan_count > self.max_replan:
            self.finish_task(f"failed after {self.max_replan} replans: {reason}")
            return
        self.get_logger().warn(f"replan {self.replan_count}/{self.max_replan}: {reason}")
        self.state = STATE_PLANNING
        self._publish_task()

    def finish_task(self, summary: str) -> None:
        self.final_summary = summary
        self.state = STATE_FINISHED
        self.pending_command = None
        self.get_logger().info(f"task {self.task_id} finished: {summary}")
        self._publish_task()
        # 2 秒后复位,等待下一个任务
        self._reset_timer = self.create_timer(2.0, self._reset_to_idle, oneshot=True)

    def _reset_to_idle(self) -> None:
        self.state = STATE_IDLE
        self._reset_timer.destroy()
        self._publish_task()

    def _publish_task(self) -> None:
        task = TaskState()
        task.task_id = self.task_id
        task.instruction = self.instruction
        task.status = self.state
        task.plan_json = json.dumps(self.history, ensure_ascii=False)
        task.current_step = self.history[-1]["skill_id"] if self.history else ""
        task.detail = self.final_summary
        self.task_pub.publish(task)

    # ---- ROS 服务调用 ----
    def _call_list_skills(self) -> list[dict]:
        if not self.list_client.wait_for_service(timeout_sec=2.0):
            return []
        future = self.list_client.call(ListSkills.Request())
        return [
            {
                "name": info.name,
                "description": info.description,
                "available": info.available,
                "requires": info.requires,
            }
            for info in future.skills
        ]

    def _load_skill_details(self, skill_id: str) -> None:
        """读取选中技能的 SKILL.md 全文,供下次决策上下文使用。"""
        if not self.load_client.wait_for_service(timeout_sec=2.0):
            return
        request = LoadSkills.Request()
        request.names = [skill_id]
        response = self.load_client.call(request)
        if response.content:
            self.skill_details[skill_id] = response.content
            self.get_logger().info(f"loaded SKILL.md for {skill_id} ({len(response.content)} chars)")


def main(args=None) -> int:
    rclpy.init(args=args)
    node = AgentOSNode()
    executor = MultiThreadedExecutor(num_threads=5)
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
