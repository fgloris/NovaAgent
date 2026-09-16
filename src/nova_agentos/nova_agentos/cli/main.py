#!/usr/bin/env python3
"""NovaAgent CLI 入口:默认启动 Textual TUI,--plain 使用行式 REPL。"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import rclpy

from .app import HELP_TEXT, NovaCliApp
from .client import RosClient

_RESET = "\033[0m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


def _run_plain(client: RosClient, resume_session: str | None) -> int:
    """行式 REPL 回退模式:后台轮询打印消息,前台读取输入。"""
    if resume_session:
        sid, name = client.resume_session(resume_session)
        print(f"已恢复会话 {name} ({sid})")
    else:
        sid, name = client.start_session("未命名")
        print(f"已自动创建会话 ({sid}),首个任务后自动命名")

    stop = threading.Event()

    def printer() -> None:
        while not stop.is_set():
            for msg in client.poll():
                if msg.session_id and msg.session_id != sid:
                    continue
                if msg.kind == "session_renamed":
                    print(_color(f"[会话命名] {msg.message}", "\033[35m"))
                    continue
                print(_color(f"[{msg.task_id}][{msg.kind}] {msg.message}", "\033[36m"))
            time.sleep(0.05)

    thread = threading.Thread(target=printer, daemon=True)
    thread.start()
    print("NovaAgent CLI 已连接。输入指令,或 /help 查看命令。")
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
            if line == "/help":
                print(HELP_TEXT)
                continue
            if line.startswith("/"):
                print(_color(f"未知命令: {line}(/help 查看)", "\033[31m"))
                continue
            try:
                task_id = client.send_message(sid, line)
                print(_color(f"[入队] task_id={task_id}", "\033[90m"))
            except Exception as exc:
                print(_color(f"错误: {exc}", "\033[1;31m"))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        try:
            client.end_session(sid)
        except Exception:
            pass
    return 0


def main(args=None) -> int:
    """解析参数,初始化 ROS 与客户端,启动 TUI 或 plain 模式。"""
    parser = argparse.ArgumentParser(description="NovaAgent 终端交互界面")
    parser.add_argument("--plain", action="store_true", help="使用行式 REPL(不启动 TUI)")
    parser.add_argument("--resume", metavar="SESSION_ID", default=None, help="启动时恢复指定 session")
    opts, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    client = RosClient()
    try:
        if opts.plain:
            return _run_plain(client, opts.resume)
        NovaCliApp(client, resume_session=opts.resume).run()
        return 0
    finally:
        client.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
