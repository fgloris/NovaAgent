#!/usr/bin/env python3
# nova_console server:FastAPI(静态页 + REST + WS)+ 内嵌 ROS 节点(订阅 agent_msg、调 RunTask)。
# 运行环境需 ROS(rclpy)+ fastapi/uvicorn。
import argparse
import asyncio
import contextlib
import queue
import threading
import time
from collections import deque
from pathlib import Path

import rclpy
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from nova_interfaces.msg import TaskState
from nova_interfaces.srv import RunTask

from nova_console.orchestrator import Orchestrator

CHAT_HISTORY_LIMIT = 200


def find_web_dir() -> Path:
    # 源码/--symlink-install:web/ 在包目录里;普通 install:web/ 在 share/nova_console/web
    here = Path(__file__).resolve().parent
    for cand in (here / "web",):
        if cand.exists():
            return cand
    try:
        from ament_index_python.packages import get_package_share_directory

        cand = Path(get_package_share_directory("nova_console")) / "web"
        if cand.exists():
            return cand
    except Exception:
        pass
    raise RuntimeError("找不到 nova_console 的 web 前端目录(重新 colcon build 或加 --symlink-install)")


WEB_DIR = find_web_dir()


class ConsoleRosNode(Node):
    # 只做聊天桥:订阅 agent_msg + 调 RunTask;会话编排是纯 Python,不依赖 ROS。
    def __init__(self, agent_msg_topic: str, run_task_service: str) -> None:
        super().__init__("nova_console")
        self.on_msg = None  # callable(TaskState),由 create_app 注入
        cg = MutuallyExclusiveCallbackGroup()
        self._run_client = self.create_client(RunTask, run_task_service, callback_group=cg)
        self.create_subscription(TaskState, agent_msg_topic, self._sub_cb, 10)

    def _sub_cb(self, msg: TaskState) -> None:
        if self.on_msg:
            self.on_msg(msg)

    def send_message(self, instruction: str, timeout_sec: float = 15.0) -> str:
        if not self._run_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("RunTask 服务不可用")
        fut = self._run_client.call_async(RunTask.Request(instruction=instruction))
        deadline = time.time() + timeout_sec
        while rclpy.ok() and not fut.done():
            if time.time() > deadline:
                raise RuntimeError("RunTask 调用超时")
            time.sleep(0.05)
        return fut.result().task_id


class ChatBody(BaseModel):
    message: str


class InputBody(BaseModel):
    text: str


def _chat_evt(msg: TaskState) -> dict:
    return {
        "type": "chat",
        "task_id": msg.task_id,
        "status": msg.status,
        "done": msg.done,
        "kind": msg.kind,
        "message": msg.message,
    }


def create_app(orch: Orchestrator, ros: ConsoleRosNode | None, event_queue: queue.Queue) -> FastAPI:
    app = FastAPI(title="NovaAgent Console")
    clients: set[WebSocket] = set()
    chat_history: deque = deque(maxlen=CHAT_HISTORY_LIMIT)

    def emit(evt: dict) -> None:
        event_queue.put(evt)

    orch.event_sink = emit

    # 聊天消息:ROS 订阅回调 -> 历史 + 广播队列
    def on_ros_msg(msg: TaskState) -> None:
        evt = _chat_evt(msg)
        chat_history.append(evt)
        emit(evt)

    if ros is not None:
        ros.on_msg = on_ros_msg

    # ---------- WS 广播 ----------
    async def broadcast_loop() -> None:
        while True:
            evt = await asyncio.to_thread(event_queue.get)
            dead = []
            for ws in list(clients):
                try:
                    await ws.send_json(evt)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(broadcast_loop())
        yield
        task.cancel()

    app.router.lifespan_context = lifespan

    # ---------- REST ----------
    @app.get("/api/profiles")
    def get_profiles():
        return {"profiles": list(orch.profiles.keys()), "active": orch.active_profile}

    @app.post("/api/start/{profile}")
    def start_profile(profile: str):
        try:
            orch.start_profile(profile)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/sessions")
    def get_sessions():
        return {"sessions": orch.list_sessions()}

    @app.get("/api/sessions/{sid}/out")
    def session_out(sid: str, after: int = 0):
        # 增量输出:返回 after 之后的新输出,前端 WS 断连时轮询兜底
        sess = orch.sessions.get(sid)
        if sess is None:
            return {"ok": False, "error": "未知会话"}
        seq, data = sess.output_since(after)
        return {"ok": True, "seq": seq, "data": data}

    @app.post("/api/sessions/{sid}/stop")
    def stop_session(sid: str):
        try:
            orch.stop(sid)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/sessions/{sid}/restart")
    def restart_session(sid: str):
        try:
            orch.restart(sid)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/sessions/{sid}/input")
    def session_input(sid: str, body: InputBody):
        try:
            orch.send_input(sid, body.text)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/chat")
    def get_chat(after: int = 0):
        return {"messages": list(chat_history)[after:]}

    @app.post("/api/chat")
    def send_chat(body: ChatBody):
        if ros is None:
            return {"ok": False, "error": "ROS 未初始化"}
        try:
            task_id = ros.send_message(body.message)
            return {"ok": True, "task_id": task_id}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    # 静态前端兜底(先注册上面的 /api、/ws 路由,最后挂 /)
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description="NovaAgent web console server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--config", default=None, help="sessions.yaml 路径")
    parser.add_argument("--agent-msg-topic", default="/nova/agentos/agent_msg")
    parser.add_argument("--run-task-service", default="/nova/agentos/run")
    a = parser.parse_args()

    orch = Orchestrator()
    orch.load_profiles(a.config)

    rclpy.init()
    event_queue: queue.Queue = queue.Queue()
    ros = ConsoleRosNode(a.agent_msg_topic, a.run_task_service)
    spin = threading.Thread(target=rclpy.spin, args=(ros,), daemon=True)
    spin.start()

    app = create_app(orch, ros, event_queue)
    print(f"NovaConsole 起于 http://{a.host}:{a.port}", flush=True)
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port)
    ros.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
