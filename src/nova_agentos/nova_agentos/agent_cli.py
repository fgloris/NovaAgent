#!/usr/bin/env python3
# NovaAgent 终端聊天 CLI:输入指令发给 agent,实时打印 agent 每轮规划/工具调用/结果。
# 命令:
#   /reset  重置仿真环境(/nova/env/reset)
#   /ping   测每个 LLM provider 连接延迟
#   /env    查询仿真环境规格(相机/state/action 键)
#   /help   显示命令帮助
#   /quit   /exit 退出
import sys
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from nova_common.llm_client import LLMClient
from nova_interfaces.msg import TaskState
from nova_interfaces.srv import EnvInfo, RunTask

# ANSI 颜色:kind -> (颜色, 标签)
_KIND_STYLE = {
    "status": ("\033[90m", "system"),
    "text": ("\033[32m", "agent"),
    "tool_call": ("\033[33m", "tool"),
    "tool_result": ("\033[36m", "result"),
}
_RESET = "\033[0m"


class AgentCliNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_agentos_cli")
        self.declare_parameter("run_task_service", "/nova/agentos/run")
        self.declare_parameter("agent_msg_topic", "/nova/agentos/agent_msg")
        self.declare_parameter("env_reset_service", "/nova/env/reset")
        self.declare_parameter("env_info_service", "/nova/env/info")
        self.declare_parameter("console_url", "http://127.0.0.1:8090")

        cg = MutuallyExclusiveCallbackGroup()
        self._run_client = self.create_client(
            RunTask, str(self.get_parameter("run_task_service").value), callback_group=cg
        )
        self._reset_client = self.create_client(
            Trigger, str(self.get_parameter("env_reset_service").value), callback_group=cg
        )
        self._info_client = self.create_client(
            EnvInfo, str(self.get_parameter("env_info_service").value), callback_group=cg
        )
        self.create_subscription(
            TaskState,
            str(self.get_parameter("agent_msg_topic").value),
            self._on_msg,
            10,
        )

    # ---------- 消息接收 ----------
    def _on_msg(self, msg: TaskState) -> None:
        color, label = _KIND_STYLE.get(msg.kind, ("\033[0m", msg.kind))
        if msg.done:
            color = "\033[1;32m" if msg.status == "done" else "\033[1;31m"
        text = f"[{msg.task_id}][{label}] {msg.message}"
        print(f"\n{color}{text}{_RESET}", flush=True)
        print("你> ", end="", flush=True)

    # ---------- 服务调用(轮询 future,节点由后台线程 spin) ----------
    def _call(self, client, request, timeout_sec: float = 10.0):
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"服务 {client.srv_name} 不可用")
        future = client.call_async(request)
        deadline = time.time() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.time() > deadline:
                raise RuntimeError(f"服务 {client.srv_name} 调用超时")
            time.sleep(0.05)
        return future.result()

    def send_message(self, instruction: str) -> str:
        resp = self._call(self._run_client, RunTask.Request(instruction=instruction))
        return resp.task_id

    def reset_env(self) -> str:
        resp = self._call(self._reset_client, Trigger.Request())
        return resp.message if resp.success else f"重置失败: {resp.message}"

    def env_info(self) -> str:
        resp = self._call(self._info_client, EnvInfo.Request())
        if not resp.success:
            return f"获取环境信息失败: {resp.message}"
        import json

        info = json.loads(resp.spec_json)
        lines = [f"sim={info.get('sim')} robots={info.get('robots')} controller={info.get('controller')}"]
        lines.append(f"action_spec={info.get('action_spec')}")
        lines.append(f"state_keys={sorted((info.get('obs_spec') or {}).get('state', {}).keys())}")
        lines.append(f"cameras={sorted((info.get('obs_spec') or {}).get('cameras', {}).keys())}")
        lines.append(f"instruction={info.get('instruction')!r}")
        return "\n".join(lines)

    def ping_llm(self) -> str:
        lines = []
        for p in LLMClient().ping():
            if p["ok"]:
                lines.append(f"  {p['name']}: {p['latency_ms']}ms OK")
            else:
                lines.append(f"  {p['name']}: FAILED ({p['error']})")
        return "\n".join(lines) or "  (无 provider)"

    def help_text(self) -> str:
        return (
            "命令:\n"
            "  /reset  重置仿真环境\n"
            "  /ping   测每个 LLM provider 连接延迟\n"
            "  /env    查询仿真环境规格(相机/state/action 键)\n"
            "  /setup [profile]  经 nova_console 拉起整套栈(默认 robocasa_loop)\n"
            "  /sessions          查看 nova_console 会话状态\n"
            "  /help   显示此帮助\n"
            "  /quit   退出\n"
            "其余输入作为指令发给 agent(上下文跨任务累积)"
        )

    # ---------- nova_console 交互(HTTP,不依赖 ROS 服务) ----------
    def _console(self, path: str, method: str = "GET") -> dict:
        import requests

        url = str(self.get_parameter("console_url").value).rstrip("/") + path
        resp = requests.request(method, url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def setup_console(self, profile: str | None) -> str:
        name = profile or "robocasa_loop"
        data = self._console(f"/api/start/{name}", "POST")
        return f"启动 {name}: {'OK' if data.get('ok') else '失败 ' + str(data.get('error', ''))}"

    def sessions_console(self) -> str:
        data = self._console("/api/sessions")
        sessions = data.get("sessions") or []
        if not sessions:
            return "(无会话)"
        return "\n".join(f"  {s['id']:<10} {s['status']:<10} {s['name']}" for s in sessions)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = AgentCliNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    print("NovaAgent CLI 已连接。输入指令,或 /help 查看命令。", flush=True)

    try:
        while rclpy.ok():
            try:
                line = input("你> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            try:
                if line.startswith("/"):
                    cmd, *rest = line.split(maxsplit=1)
                    if cmd == "/setup":
                        text = node.setup_console(rest[0] if rest else None)
                    elif cmd == "/sessions":
                        text = node.sessions_console()
                    else:
                        text = {
                            "/help": node.help_text(),
                            "/reset": node.reset_env(),
                            "/ping": node.ping_llm(),
                            "/env": node.env_info(),
                        }.get(cmd)
                    if text is None:
                        print(f"未知命令: {cmd}(/help 查看)", flush=True)
                    else:
                        print(f"\033[90m{text}\033[0m", flush=True)
                else:
                    task_id = node.send_message(line)
                    print(f"\033[90m[入队] task_id={task_id}\033[0m", flush=True)
            except Exception as exc:
                print(f"\033[1;31m错误: {exc}\033[0m", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
