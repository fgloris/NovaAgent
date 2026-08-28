#!/usr/bin/env python3
# AgentOS 主节点:接收指令入队 -> 后台 agent 循环持续处理(上下文跨任务累积)。
# 状态经每任务话题 /nova/agentos/task_state/t_{task_id} 发布(最后一条消息 + 是否完成)。
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node

from nova_common.llm_client import LLMClient
from nova_interfaces.msg import TaskState
from nova_interfaces.srv import RunTask

from nova_agentos.agent_loop import AgentLoop
from nova_agentos.mcp_adapter import McpAdapter
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
        self.adapter = McpAdapter(
            self,
            list_tools_srv=str(self.get_parameter("list_tools_service").value),
            execute_action=str(self.get_parameter("execute_action").value),
        )

        self._state_pubs: dict[str, object] = {}
        self.loop = AgentLoop(self.llm, self.skills, self.adapter, on_state=self._on_state)
        self.loop.start()

        self.create_service(
            RunTask, str(self.get_parameter("run_task_service").value), self._run_task_cb
        )
        self.get_logger().info(f"AgentOS 就绪,skill 目录: {skills_dir}")

    # RunTask 非阻塞:入队即返回 task_id,结果经状态话题观察
    def _run_task_cb(self, request, response):
        task_id = uuid.uuid4().hex[:8]
        self.loop.submit(task_id, request.instruction)
        response.task_id = task_id
        response.success = True
        response.message = f"已入队,状态见 /nova/agentos/task_state/t_{task_id}"
        self.get_logger().info(f"任务 {task_id} 已入队: {request.instruction}")
        return response

    # agent loop 线程回调 -> 发布到每任务话题
    def _on_state(self, task_id: str, status: str, message: str, done: bool) -> None:
        pub = self._state_pubs.get(task_id)
        if pub is None:
            topic = f"/nova/agentos/task_state/t_{task_id}"
            pub = self.create_publisher(TaskState, topic, 10)
            self._state_pubs[task_id] = pub
        msg = TaskState()
        msg.task_id = task_id
        msg.status = status
        msg.done = done
        msg.message = message
        pub.publish(msg)
        if done:
            self.get_logger().info(f"任务 {task_id} [{status}]: {message}")

    def destroy_node(self) -> bool:
        self.loop.stop()
        return super().destroy_node()


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
