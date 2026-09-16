"""底部状态栏:会话名、session_id 与忙碌指示。"""
from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from .. import theme


class StatusBar(Static):
    """展示当前会话与运行状态。"""

    def __init__(self) -> None:
        super().__init__(id="statusbar")
        self._name = ""
        self._session_id = ""
        self._busy = False
        self._message = "就绪"

    def set_session(self, name: str, session_id: str) -> None:
        """更新当前会话。"""
        self._name, self._session_id = name, session_id
        self._refresh_text()

    def set_busy(self, busy: bool) -> None:
        """更新忙碌状态。"""
        self._busy = busy
        self._refresh_text()

    def set_message(self, message: str) -> None:
        """更新右侧提示文本。"""
        self._message = message
        self._refresh_text()

    def _refresh_text(self) -> None:
        text = Text()
        dot = "●" if self._busy else "○"
        text.append(f"{dot} ", style=theme.PRIMARY if self._busy else theme.MUTED)
        text.append(self._name or "未命名", style=f"bold {theme.TEXT}")
        if self._session_id:
            text.append(f"  {self._session_id}", style=theme.MUTED)
        text.append(f"   {self._message}", style=theme.MUTED)
        text.append("    /help 查看命令", style=theme.BORDER_ACTIVE)
        self.update(text)
