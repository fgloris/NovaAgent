#!/usr/bin/env python3
"""AgentOS 主节点:接收指令入队 -> 后台 agent 循环持续处理(上下文跨任务累积)。

agent 全部消息(规划文本/工具调用与结果/完成)经全局话题 /nova/agentos/agent_msg 发布。
"""
import os
import threading
from pathlib import Path

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node

from nova_common.llm_client import LLMClient
from nova_interfaces.msg import TaskState
from nova_interfaces.srv import (
    EndSession,
    ListSessions,
    RenameSession,
    ResumeSession,
    RunTask,
    StartSession,
)

from nova_agentos.api_logger import ApiLogger
from nova_agentos.agent_loop import AgentLoop
from nova_agentos.mcp_adapter import McpAdapter
from nova_agentos.memory import ImageMemory, ImageSampler, SessionManager
from nova_agentos.robot_state_observer import RobotStateObserver
from nova_agentos.doc_store import DocStore
from nova_agentos.skill_store import SkillStore
from nova_agentos.vision_observer import VisionObserver

# 自动命名:这些占位名(或空名)会在首个任务时由 VLM 概括为"场景-做什么"
_PLACEHOLDER_NAMES = {"", "default", "未命名", "新会话", "new session"}
_NAME_PROMPT = (
    "你是具身机器人会话的命名助手。根据用户的第一条指令,生成一个简短的会话名,"
    "格式为『场景-做什么』,例如『厨房-收拾桌面』『客厅-拿水杯』。"
    "只输出会话名本身,不要引号、不要标点解释,不超过 12 个字。"
)


class AgentosNode(Node):
    """AgentOS 的 ROS 门面:装配记忆/技能/LLM/视觉/工具适配器,并暴露会话与任务服务。"""

    def __init__(self) -> None:
        super().__init__("nova_agentos")
        self.declare_parameter("skills_dir", "")
        self.declare_parameter("docs_dir", "")
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

        # ---------- 图像记忆 ----------
        self.declare_parameter("memory_dir", "")
        self.declare_parameter("image_sample_period_sec", 1.0)
        self.declare_parameter("image_diff_mse_threshold", 0.0005)
        self.declare_parameter("image_state_diff_threshold", 0.5)
        self.declare_parameter("image_history_depth", 4)
        self.declare_parameter("image_processed_depth", 3)
        self.declare_parameter("image_link_max", 60)
        self.declare_parameter("image_processed_max", 16)

        skills_dir = str(self.get_parameter("skills_dir").value)
        docs_dir = str(self.get_parameter("docs_dir").value)
        if not skills_dir or not docs_dir:
            from ament_index_python.packages import get_package_share_directory
            share = Path(get_package_share_directory("nova_agentos"))
            skills_dir = skills_dir or str(share / "skills")
            docs_dir = docs_dir or str(share / "docs")

        self.skills = SkillStore(skills_dir)
        self.docs = DocStore(docs_dir)
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

        # session 级图像记忆:按需创建并缓存,采样器写入当前运行任务所属 session
        default_memory = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        self._memory_dir = Path(
            str(self.get_parameter("memory_dir").value) or default_memory / "nova_agentos" / "memory"
        ).expanduser()
        self._memory_cache: dict[str, ImageMemory] = {}
        self._image_history_depth = max(1, int(self.get_parameter("image_history_depth").value))
        self._image_processed_depth = max(0, int(self.get_parameter("image_processed_depth").value))
        self._image_link_max = max(1, int(self.get_parameter("image_link_max").value))
        self._image_processed_max = max(1, int(self.get_parameter("image_processed_max").value))
        self._image_diff_mse = float(self.get_parameter("image_diff_mse_threshold").value)
        self._image_state_diff = float(self.get_parameter("image_state_diff_threshold").value)
        self._image_max_size = int(self.get_parameter("vlm_max_image_size").value)
        self._image_jpeg_quality = int(self.get_parameter("vlm_jpeg_quality").value)

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
            image_memory_factory=self._image_memory,
            frame_provider=self.vision.latest_frames,
            image_history_depth=self._image_history_depth,
            image_processed_depth=self._image_processed_depth,
            docs=self.docs,
        )
        self.loop.start()
        self.sampler = ImageSampler(
            self,
            self.vision.latest_frames,
            self.loop.current_memory,
            state_provider=self.robot_state.gripper_state,
            period_sec=float(self.get_parameter("image_sample_period_sec").value),
        )

        self.create_service(
            RunTask, str(self.get_parameter("run_task_service").value), self._run_task_cb
        )
        self.create_service(StartSession, "/nova/agentos/session/start", self._start_session_cb)
        self.create_service(ResumeSession, "/nova/agentos/session/resume", self._resume_session_cb)
        self.create_service(EndSession, "/nova/agentos/session/end", self._end_session_cb)
        self.create_service(ListSessions, "/nova/agentos/session/list", self._list_sessions_cb)
        self.create_service(RenameSession, "/nova/agentos/session/rename", self._rename_session_cb)
        self.get_logger().info(
            f"AgentOS 就绪,skill 目录: {skills_dir}, doc 目录: {docs_dir}, 图像记忆目录: {self._memory_dir}"
        )

    def _image_memory(self, session_id: str) -> ImageMemory:
        """返回(并按需创建)指定 session 的图像记忆。"""
        memory = self._memory_cache.get(session_id)
        if memory is None:
            memory = ImageMemory(
                self._memory_dir / session_id,
                link_max=self._image_link_max,
                processed_max=self._image_processed_max,
                jpeg_quality=self._image_jpeg_quality,
                max_size=self._image_max_size,
                diff_mse_threshold=self._image_diff_mse,
                state_diff_threshold=self._image_state_diff,
            )
            self._memory_cache[session_id] = memory
        return memory

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
        self._maybe_name_session(task_id, request.session_id, request.instruction)
        response.task_id = task_id
        response.success = True
        response.message = "已入队,消息见 /nova/agentos/agent_msg"
        self.get_logger().info(f"任务 {task_id} 已入队: {request.instruction}")
        return response

    def _maybe_name_session(self, task_id: str, session_id: str, instruction: str) -> None:
        """首个任务且会话名仍是占位符时,后台线程用 VLM 概括为"场景-做什么"并改名。"""
        try:
            record = self.sessions.get(session_id, allow_ended=True)
        except (FileNotFoundError, ValueError):
            return
        if len(record.task_ids) != 1 or record.name.strip() not in _PLACEHOLDER_NAMES:
            return
        threading.Thread(
            target=self._name_session,
            args=(task_id, session_id, instruction),
            daemon=True,
        ).start()

    def _name_session(self, task_id: str, session_id: str, instruction: str) -> None:
        """调用 VLM 生成会话名并改名;失败静默跳过,不影响任务执行。"""
        try:
            result = self.llm.chat(
                [
                    {"role": "system", "content": _NAME_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.2,
                max_tokens=512,
                task_id=task_id,
                session_id=session_id,
            )
            name = (result.content or "").strip().strip("\"'“”‘’ \n")
            name = name.splitlines()[0].strip() if name else ""
            if not name:
                return
            name = name[:24]
            self.sessions.rename(session_id, name)
            self._publish(task_id, session_id, "working", name, False, "session_renamed")
            self.get_logger().info(f"会话 {session_id} 已命名: {name}")
        except Exception as exc:
            self.get_logger().warn(f"会话自动命名失败: {exc}")

    def _start_session_cb(self, request, response):
        """处理 StartSession:创建并激活一个新 session。"""
        try:
            record = self.sessions.start(request.name)
            self.loop.activate_session(record.session_id)
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
            self.loop.activate_session(record.session_id)
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
            self.loop.deactivate_session(record.session_id)
            response.success = True
            response.archive_path = str(self.sessions.root / record.session_id)
            response.message = f"session 已结束，文件已保留在 {response.archive_path}"
        except (FileNotFoundError, ValueError) as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _list_sessions_cb(self, request, response):
        """处理 ListSessions:返回全部 session 的 id/名称/状态/更新时间。"""
        try:
            records = self.sessions.list()
            response.session_ids = [r.session_id for r in records]
            response.names = [r.name for r in records]
            response.statuses = [r.status for r in records]
            response.updated_at = [r.updated_at for r in records]
            response.success = True
            response.message = f"共 {len(records)} 个 session"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _rename_session_cb(self, request, response):
        """处理 RenameSession:修改 session 名称。"""
        try:
            record = self.sessions.rename(request.session_id, request.name)
            response.success = True
            response.message = f"已重命名: {record.name}"
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
        """agent loop 线程回调:把状态事件发布到全局话题(消息按 task_id/session_id 区分)。"""
        self._publish(task_id, session_id, status, message, done, kind)
        if done:
            self.get_logger().info(f"任务 {task_id} [{status}]: {message}")

    def _publish(
        self,
        task_id: str,
        session_id: str,
        status: str,
        message: str,
        done: bool,
        kind: str,
    ) -> None:
        """构造并发布一条 TaskState。"""
        msg = TaskState()
        msg.task_id = task_id
        msg.session_id = session_id
        msg.status = status
        msg.done = done
        msg.kind = kind
        msg.message = message
        self._msg_pub.publish(msg)

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
