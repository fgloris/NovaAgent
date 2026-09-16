"""CLI 主题色板与事件配色,参照 opencode 默认暗色主题。"""
from __future__ import annotations

# opencode dark 语义色
BACKGROUND = "#0a0a0a"
PANEL = "#141414"
ELEMENT = "#1e1e1e"
PRIMARY = "#fab283"
SECONDARY = "#5c9cf5"
ACCENT = "#9d7cd8"
SUCCESS = "#7fd88f"
WARNING = "#f5a742"
ERROR = "#e06c75"
INFO = "#56b6c2"
TEXT = "#eeeeee"
MUTED = "#808080"
BORDER = "#484848"
BORDER_ACTIVE = "#606060"
BORDER_SUBTLE = "#3c3c3c"

# TaskState.kind -> (颜色, 标签, 图标)
KIND_STYLE: dict[str, tuple[str, str, str]] = {
    "status": (MUTED, "system", "·"),
    "text": (PRIMARY, "agent", "◆"),
    "tool_call": (WARNING, "tool", "⚙"),
    "tool_feedback": (ACCENT, "feedback", "↳"),
    "tool_result": (INFO, "result", "✔"),
    "session_renamed": (SECONDARY, "session", "✎"),
}

APP_CSS = f"""
Screen {{
    background: {BACKGROUND};
    color: {TEXT};
}}

#conversation {{
    background: {BACKGROUND};
    padding: 0 1;
    scrollbar-color: {BORDER} {BACKGROUND};
    scrollbar-color-hover: {BORDER_ACTIVE} {BACKGROUND};
    scrollbar-color-active: {PRIMARY} {BACKGROUND};
    scrollbar-background: {BACKGROUND};
    scrollbar-background-hover: {BACKGROUND};
}}

.event {{
    margin: 0 0 1 0;
    padding: 0 0 0 1;
}}

.event-user {{
    border-left: thick {SECONDARY};
    color: {TEXT};
    background: {PANEL};
    padding: 0 1;
    margin: 0 0 1 0;
}}

.event-agent {{
    border-left: thick {PRIMARY};
    padding: 0 0 0 1;
    margin: 0 0 1 0;
    background: {BACKGROUND};
}}

.event-system {{
    color: {MUTED};
}}

.event-feedback {{
    color: {ACCENT};
}}

.event-done {{
    border-left: thick {SUCCESS};
    background: {PANEL};
    padding: 0 1;
    margin: 1 0;
}}

.event-failed {{
    border-left: thick {ERROR};
    background: {PANEL};
    padding: 0 1;
    margin: 1 0;
}}

ToolResult {{
    background: {PANEL};
    border-left: thick {INFO};
    margin: 0 0 1 0;
    padding: 0 1;
}}

ToolResult.-failed {{
    border-left: thick {ERROR};
}}

Collapsible {{
    background: {PANEL};
    border-left: thick {BORDER};
    margin: 0 0 1 0;
}}

Collapsible > .collapsible--title {{
    color: {INFO};
}}

Markdown {{
    background: {BACKGROUND};
}}

#prompt {{
    background: {ELEMENT};
    border: tall {BORDER_ACTIVE};
    color: {TEXT};
    margin: 0 1;
    height: auto;
    max-height: 10;
    min-height: 3;
}}

#prompt:focus {{
    border: tall {PRIMARY};
}}

#hint {{
    color: {MUTED};
    padding: 0 2;
    height: 1;
}}

#statusbar {{
    background: {PANEL};
    color: {MUTED};
    height: 1;
    padding: 0 1;
    dock: bottom;
}}
"""
