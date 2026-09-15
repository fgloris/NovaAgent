#!/usr/bin/env python3
"""AgentOS 主节点:接收指令入队 -> 后台 agent 循环持续处理(上下文跨任务累积)。

agent 全部消息(规划文本/工具调用与结果/完成)经全局话题 /nova/agentos/agent_msg 发布。
"""
from pathlib import Path

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node

from nova_common.llm_client import LLMClient
from nova_interfaces.msg import TaskState
from nova_interfaces.srv import EndSession, ResumeSession, RunTask, StartSession

from nova_agentos.api_logger import ApiLogger
from nova_agentos.agent_loop import AgentLoop
from nova_agentos.mcp_adapter import McpAdapter
from nova_agentos.memory import SessionManager
from nova_agentos.robot_state_observer import RobotStateObserver
from nova_agentos.skill_store import SkillStore
from nova_agentos.vision_observer import VisionObserver


class AgentosNode(Node):
    """AgentOS 的 ROS 门面:装配记忆/技能/LLM/视觉/工具适配器,并暴露会话与任务服务。"""

    def __init__(self) -> None:
        super().__init__("nova_agentos")
        self.declare_parameter("skills_dir", "")
        self.declare_parameter("list_tools_service", "/nova/executor_manager/list_tools")
        self.declare_parameter("execute_action", "/nova/executor_manager/execute")
        self.declare_parameter("run_task_service", "/nova/agentos/run")
        self.declare_parameter("agent_msg_topic", "/nova/agentos/agent_msg")
        self.declare_parameter("env_ns", "/nova/env")
        self.declare_parameter(
            "vlm_camera_names", [], ParameterDescriptor(dynamic_typing=True)
        )
        self.declare_parameter("vlm_max_images", 6)
        self.declare_parameter("vlm_max_image_size", 768)
        self.declare_parameter("vlm_jpeg_quality", 80)
        self.declare_parameter("context_budget_tokens", 12000)
        self.declare_parameter("context_compaction_enabled", True)
        self.declare_parameter("max_recent_tasks", 8)
        self.declare_parameter("session_dir", "")
        self.declare_parameter("api_log_enabled", True)
        self.declare_parameter("api_log_images", True)
        self.declare_parameter("api_log_dir", "")
        self.declare_parameter("api_log_retention_days", 30)

        skills_dir = str(self.get_parameter("skills_dir").value)
        if not skills_dir:
            from ament_index_python.packages import get_package_share_directory
            skills_dir = str(Path(get_package_share_directory("nova_agentos")) / "skills")

        self.skills = SkillStore(skills_dir)
        api_dir = str(self.get_parameter("api_log_dir").value)
        self.api_logger = ApiLogger(
            enabled=bool(self.get_parameter("api_log_enabled").value),
            images=bool(self.get_parameter("api_log_images").value),
            directory=api_dir or None,
            retention_days=int(self.get_parameter("api_log_retention_days").value),
        )
        self.llm = LLMClient(vision=True, api_logger=self.api_logger)
        session_dir = str(self.get_parameter("session_dir").value)
        self.sessions = SessionManager(session_dir or None)
        self.vision = VisionObserver(
            self,
            env_ns=str(self.get_parameter("env_ns").value),
            camera_names=list(self.get_parameter("vlm_camera_names").value),
            max_images=int(self.get_parameter("vlm_max_images").value),
            max_image_size=int(self.get_parameter("vlm_max_image_size").value),
            jpeg_quality=int(self.get_parameter("vlm_jpeg_quality").value),
        )
        self.robot_state = RobotStateObserver(
            self, env_ns=str(self.get_parameter("env_ns").value)
        )
        self.adapter = McpAdapter(
            self,
            list_tools_srv=str(self.get_parameter("list_tools_service").value),
            execute_action=str(self.get_parameter("execute_action").value),
        )

        self._msg_pub = self.create_publisher(
            TaskState, str(self.get_parameter("agent_msg_topic").value), 10
        )
        self.loop = AgentLoop(
            self.llm,
            self.skills,
            self.adapter,
            session_manager=self.sessions,
            on_state=self._on_state,
            observation_provider=self.vision.snapshot_message,
            robot_context_provider=self.vision.get_robot_context,
            robot_state_provider=self.robot_state.snapshot_message,
            context_budget_tokens=int(self.get_parameter("context_budget_tokens").value),
            context_compaction_enabled=bool(self.get_parameter("context_compaction_enabled").value),
            max_recent_tasks=int(self.get_parameter("max_recent_tasks").value),
        )
        self.loop.start()

        self.create_service(
            RunTask, str(self.get_parameter("run_task_service").value), self._run_task_cb
        )
        self.create_service(StartSession, "/nova/agentos/session/start", self._start_session_cb)
        self.create_service(ResumeSession, "/nova/agentos/session/resume", self._resume_session_cb)
        self.create_service(EndSession, "/nova/agentos/session/end", self._end_session_cb)
        self.get_logger().info(f"AgentOS 就绪,skill 目录: {skills_dir}")

    def _run_task_cb(self, request, response):
        """RunTask 非阻塞:入队即返回 task_id,agent 消息经 /nova/agentos/agent_msg 观察。"""
        try:
            task = self.sessions.create_task(request.session_id, request.instruction)
        except (FileNotFoundError, ValueError) as exc:
            response.task_id = ""
            response.success = False
            response.message = str(exc)
            return response
        task_id = task.task_id
        self.loop.submit(task_id, request.session_id, request.instruction)
        response.task_id = task_id
        response.success = True
        response.message = "已入队,消息见 /nova/agentos/agent_msg"
        self.get_logger().info(f"任务 {task_id} 已入队: {request.instruction}")
        return response

    def _start_session_cb(self, request, response):
        """处理 StartSession:创建并激活一个新 session。"""
        try:
            record = self.sessions.start(request.name)
            response.session_id = record.session_id
            response.success = True
            response.message = f"session 已创建并激活: {record.name}"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _resume_session_cb(self, request, response):
        """处理 ResumeSession:恢复已有 session 为 active。"""
        try:
            record = self.sessions.resume(request.session_id)
            response.success = True
            response.name = record.name
            response.message = f"session 已恢复: {record.name}"
        except (FileNotFoundError, ValueError) as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _end_session_cb(self, request, response):
        """处理 EndSession:结束 session 并保留其文件。"""
        try:
            record = self.sessions.end(request.session_id)
            response.success = True
            response.archive_path = str(self.sessions.root / record.session_id)
            response.message = f"session 已结束，文件已保留在 {response.archive_path}"
        except (FileNotFoundError, ValueError) as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _on_state(
        self,
        task_id: str,
        session_id: str,
        status: str,
        message: str,
        done: bool,
        kind: str,
    ) -> None:
        """agent loop 线程回调:把状态事件发布到全局话题(消息按 task_id 区分)。"""
        msg = TaskState()
        msg.task_id = task_id
        msg.status = status
        msg.done = done
        msg.kind = kind
        msg.message = message
        self._msg_pub.publish(msg)
        if done:
            self.get_logger().info(f"任务 {task_id} [{status}]: {message}")

    def destroy_node(self) -> bool:
        """销毁节点前先停止后台 agent 循环线程。"""
        self.loop.stop()
        return super().destroy_node()


def main(args=None) -> int:
    """初始化 ROS 并运行 AgentOS 节点。"""
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
