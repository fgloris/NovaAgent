"""会话区消息控件:把 TaskState 事件渲染成带配色的控件。"""
from __future__ import annotations

from rich.text import Text
from textual.widget import Widget
from textual.widgets import Collapsible, Markdown, Static

from .. import theme
from ..theme import KIND_STYLE


def _first_line(text: str, limit: int = 120) -> str:
    """取首行并截断,用于折叠标题。"""
    line = (text or "").strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


class UserMessage(Static):
    """用户输入。"""

    def __init__(self, text: str) -> None:
        super().__init__(Text(f"你> {text}", style=f"bold {theme.SECONDARY}"), classes="event event-user")


class AgentText(Markdown):
    """agent 正文(markdown 渲染)。"""

    def __init__(self, text: str) -> None:
        super().__init__(text, classes="event event-agent")


class SystemLine(Static):
    """系统/状态行。"""

    def __init__(self, text: str, color: str = theme.MUTED, icon: str = "·") -> None:
        super().__init__(Text(f"{icon} {text}", style=color), classes="event event-system")


class FeedbackLine(Static):
    """工具反馈行。"""

    def __init__(self, text: str) -> None:
        super().__init__(Text(f"↳ {text}", style=theme.ACCENT), classes="event event-feedback")


class ToolCallLine(Static):
    """工具调用(单行,黄色)。"""

    def __init__(self, text: str) -> None:
        super().__init__(Text(f"⚙ {text}", style=theme.WARNING), classes="event")


class ToolResult(Collapsible):
    """工具结果(可折叠,成功青色/失败红色)。"""

    def __init__(self, message: str, failed: bool = False) -> None:
        title = Text()
        title.append("✔ " if not failed else "✘ ", style=theme.ERROR if failed else theme.SUCCESS)
        title.append(_first_line(message), style=theme.INFO if not failed else theme.ERROR)
        body = Static(Text(message, style=theme.TEXT))
        super().__init__(body, title=title, collapsed=True)
        if failed:
            self.add_class("-failed")


class DoneBanner(Static):
    """任务结束横幅。"""

    def __init__(self, message: str, ok: bool) -> None:
        color = theme.SUCCESS if ok else theme.ERROR
        label = "任务完成" if ok else "任务失败"
        super().__init__(
            Text(f"{label}: {message}", style=f"bold {color}"),
            classes="event event-done" if ok else "event event-failed",
        )


def build_event_widget(msg) -> Widget | None:
    """按 TaskState.kind 构造对应控件。"""
    kind = msg.kind
    if msg.done:
        return DoneBanner(msg.message, msg.status == "done")
    if kind == "text":
        return AgentText(msg.message)
    if kind == "tool_call":
        return ToolCallLine(msg.message)
    if kind == "tool_feedback":
        return FeedbackLine(msg.message)
    if kind == "tool_result":
        failed = "工具执行失败" in msg.message
        return ToolResult(msg.message, failed=failed)
    if kind == "session_renamed":
        return SystemLine(f"会话已命名: {msg.message}", color=theme.SECONDARY, icon="✎")
    _, label, icon = KIND_STYLE.get(kind, (theme.MUTED, kind, "·"))
    return SystemLine(msg.message, icon=icon)
