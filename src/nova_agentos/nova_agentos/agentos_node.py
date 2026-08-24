#!/usr/bin/env python3
# AgentOS 主节点:接收指令 -> skill 注入规划 -> DAG 校验 -> 调度执行。
import json
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node

from nova_common.llm_client import LLMClient
from nova_interfaces.msg import TaskState
from nova_interfaces.srv import RunTask

from nova_agentos.dag_executor import DagExecutor
from nova_agentos.dag_validator import validate
from nova_agentos.mcp_adapter import McpAdapter
from nova_agentos.planner import Planner
from nova_agentos.skill_store import SkillStore


class AgentosNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_agentos")
        self.declare_parameter("skills_dir", "")
        self.declare_parameter("list_tools_service", "/nova/executor_manager/list_tools")
        self.declare_parameter("execute_action", "/nova/executor_manager/execute")
        self.declare_parameter("run_task_service", "/nova/agentos/run")

        skills_dir = str(self.get_parameter("skills_dir").value)
        if not skills_dir:
            from ament_index_python.packages import get_package_share_directory
            skills_dir = str(Path(get_package_share_directory("nova_agentos")) / "skills")

        self.skills = SkillStore(skills_dir)
        self.llm = LLMClient()
        self.planner = Planner(self.llm, self.skills)
        self.adapter = McpAdapter(
            self,
            list_tools_srv=str(self.get_parameter("list_tools_service").value),
            execute_action=str(self.get_parameter("execute_action").value),
        )
        self.executor = DagExecutor(self.adapter)
        self.state_pub = self.create_publisher(TaskState, "/nova/agentos/task_state", 10)

        self.create_service(
            RunTask, str(self.get_parameter("run_task_service").value), self._run_task_cb
        )
        self.get_logger().info(f"AgentOS 就绪,skill 目录: {skills_dir}")

    def _run_task_cb(self, request, response):
        task_id = uuid.uuid4().hex[:8]
        instruction = request.instruction
        self._publish_state(task_id, instruction, "planning")
        try:
            descriptors = self.adapter.fetch_tools()
            tools_text = "\n".join(f"- {d.name}: {d.description}" for d in descriptors)
            graph = self.planner.plan(instruction, tools_text)
            validate(graph, [d.name for d in descriptors])
            plan_json = json.dumps(graph, ensure_ascii=False)
            self._publish_state(task_id, instruction, "executing", plan_json)

            def on_step(nid):
                self._publish_state(task_id, instruction, "executing", plan_json, current_step=nid)

            self.executor.execute(graph, task_id, on_step=on_step)
            self._publish_state(task_id, instruction, "done", plan_json)
            response.task_id = task_id
            response.success = True
            response.message = "任务完成"
        except Exception as exc:
            self.get_logger().error(f"任务 {task_id} 失败: {exc}")
            self._publish_state(task_id, instruction, "failed", detail=str(exc))
            response.task_id = task_id
            response.success = False
            response.message = str(exc)
        return response

    def _publish_state(self, task_id, instruction, status, plan_json="", current_step="", detail=""):
        msg = TaskState()
        msg.task_id = task_id
        msg.instruction = instruction
        msg.status = status
        msg.plan_json = plan_json
        msg.current_step = current_step
        msg.detail = detail
        self.state_pub.publish(msg)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = AgentosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
