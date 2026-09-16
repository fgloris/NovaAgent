"""CLI 的 ROS 客户端封装:会话/任务/环境服务 + agent 消息订阅。"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from nova_common.llm_client import LLMClient
from nova_interfaces.msg import TaskState
from nova_interfaces.srv import (
    EndSession,
    EnvInfo,
    ListSessions,
    RenameSession,
    ResumeSession,
    RunTask,
    StartSession,
)

DEFAULT_TIMEOUT = 10.0


class RosClient:
    """持有 ROS 节点与后台 spin 线程,对外暴露同步调用方法。"""

    def __init__(self) -> None:
        self.node = Node(f"nova_agentos_cli_{os.urandom(3).hex()}")
        cg = MutuallyExclusiveCallbackGroup()
        self._run = self.node.create_client(RunTask, "/nova/agentos/run", callback_group=cg)
        self._start = self.node.create_client(StartSession, "/nova/agentos/session/start", callback_group=cg)
        self._resume = self.node.create_client(ResumeSession, "/nova/agentos/session/resume", callback_group=cg)
        self._end = self.node.create_client(EndSession, "/nova/agentos/session/end", callback_group=cg)
        self._list = self.node.create_client(ListSessions, "/nova/agentos/session/list", callback_group=cg)
        self._rename = self.node.create_client(RenameSession, "/nova/agentos/session/rename", callback_group=cg)
        self._reset = self.node.create_client(Trigger, "/nova/env/reset", callback_group=cg)
        self._info = self.node.create_client(EnvInfo, "/nova/env/info", callback_group=cg)
        self._events: queue.Queue[TaskState] = queue.Queue()
        self.node.create_subscription(TaskState, "/nova/agentos/agent_msg", self._on_msg, 10)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _on_msg(self, msg: TaskState) -> None:
        """订阅回调(ROS 线程):把消息放进队列,由 UI 线程消费。"""
        self._events.put(msg)

    def poll(self) -> list[TaskState]:
        """取出当前已到达的全部 agent 消息。"""
        out: list[TaskState] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def _call(self, client, request, timeout_sec: float = DEFAULT_TIMEOUT):
        """同步调用 ROS 服务:等待服务可用后轮询 future,超时抛异常。"""
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"服务 {client.srv_name} 不可用(AgentOS 是否已启动?)")
        future = client.call_async(request)
        deadline = time.time() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.time() > deadline:
                raise RuntimeError(f"服务 {client.srv_name} 调用超时")
            time.sleep(0.02)
        return future.result()

    # ---------- 会话 ----------
    def start_session(self, name: str = "未命名") -> tuple[str, str]:
        """创建并激活新 session,返回 (session_id, name)。"""
        resp = self._call(self._start, StartSession.Request(name=name))
        if not resp.success:
            raise RuntimeError(resp.message)
        return resp.session_id, name

    def resume_session(self, session_id: str) -> tuple[str, str]:
        """恢复已有 session,返回 (session_id, name)。"""
        resp = self._call(self._resume, ResumeSession.Request(session_id=session_id))
        if not resp.success:
            raise RuntimeError(resp.message)
        return session_id, resp.name

    def end_session(self, session_id: str) -> str:
        """结束 session 并保留文件。"""
        if not session_id:
            return ""
        resp = self._call(self._end, EndSession.Request(session_id=session_id))
        return resp.archive_path if resp.success else resp.message

    def list_sessions(self) -> list[dict[str, str]]:
        """返回全部 session 的 {id, name, status, updated_at}。"""
        resp = self._call(self._list, ListSessions.Request())
        if not resp.success:
            raise RuntimeError(resp.message)
        return [
            {
                "id": sid,
                "name": name,
                "status": status,
                "updated_at": updated,
            }
            for sid, name, status, updated in zip(
                resp.session_ids, resp.names, resp.statuses, resp.updated_at
            )
        ]

    def rename_session(self, session_id: str, name: str) -> str:
        """重命名 session。"""
        resp = self._call(self._rename, RenameSession.Request(session_id=session_id, name=name))
        if not resp.success:
            raise RuntimeError(resp.message)
        return resp.message

    # ---------- 任务 / 环境 ----------
    def send_message(self, session_id: str, instruction: str) -> str:
        """提交指令,返回 task_id。"""
        resp = self._call(self._run, RunTask.Request(session_id=session_id, instruction=instruction))
        if not resp.success:
            raise RuntimeError(resp.message)
        return resp.task_id

    def reset_env(self) -> str:
        """重置仿真环境。"""
        resp = self._call(self._reset, Trigger.Request())
        return resp.message if resp.success else f"重置失败: {resp.message}"

    def env_info(self) -> str:
        """查询并格式化仿真环境规格。"""
        resp = self._call(self._info, EnvInfo.Request())
        if not resp.success:
            return f"获取环境信息失败: {resp.message}"
        info: dict[str, Any] = json.loads(resp.spec_json)
        lines = [f"sim={info.get('sim')} robots={info.get('robots')} controller={info.get('controller')}"]
        lines.append(f"action_spec={info.get('action_spec')}")
        lines.append(f"state_keys={sorted((info.get('obs_spec') or {}).get('state', {}).keys())}")
        lines.append(f"cameras={sorted((info.get('obs_spec') or {}).get('cameras', {}).keys())}")
        lines.append(f"instruction={info.get('instruction')!r}")
        return "\n".join(lines)

    def ping_llm(self) -> str:
        """逐个 LLM provider 探测连接延迟。"""
        lines = []
        for p in LLMClient().ping():
            if p["ok"]:
                lines.append(f"  {p['name']}: {p['latency_ms']}ms OK")
            else:
                lines.append(f"  {p['name']}: FAILED ({p['error']})")
        return "\n".join(lines) or "  (无 provider)"

    def shutdown(self) -> None:
        """停止 spin 线程并销毁节点。"""
        try:
            self._executor.shutdown()
        finally:
            self.node.destroy_node()
