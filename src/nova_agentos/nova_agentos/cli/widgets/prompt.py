"""多行输入框:回车发送、Shift+回车换行、方向键历史、鼠标点击定位光标。"""
from __future__ import annotations

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class Prompt(TextArea):
    """支持多行编辑与历史的输入框。"""

    class Submitted(Message):
        """用户提交一条输入。"""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    BINDINGS = [
        Binding("enter", "submit", "发送", show=True, priority=True),
        Binding("shift+enter", "newline", "换行", show=False, priority=True),
        Binding("ctrl+j", "newline", "换行", show=False, priority=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(soft_wrap=True, show_line_numbers=False, tab_behavior="focus", **kwargs)
        self._history: list[str] = []
        self._hist_index: int | None = None
        self._draft = ""

    def add_history(self, text: str) -> None:
        """提交后记录历史(去重连续项),并复位历史指针。"""
        text = text.strip()
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._hist_index = None
        self._draft = ""

    def action_submit(self) -> None:
        """回车:内容非空则发出 Submitted。"""
        value = self.text.strip()
        if value:
            self.post_message(self.Submitted(value))

    def action_newline(self) -> None:
        """插入换行。"""
        self.insert("\n")

    def clear(self) -> None:
        """清空输入并复位历史。"""
        self.load_text("")
        self._hist_index = None
        self._draft = ""

    def _on_key(self, event) -> None:
        """光标已在首/末行时,上/下键改为浏览历史。"""
        if event.key == "up" and self.cursor_at_first_line and self._history:
            event.stop()
            self._history_move(-1)
            return
        if event.key == "down" and self.cursor_at_last_line and self._history:
            event.stop()
            self._history_move(1)
            return
        super()._on_key(event)

    def _history_move(self, delta: int) -> None:
        """在历史中前后移动,超出末尾回到草稿。"""
        if self._hist_index is None:
            self._draft = self.text
            self._hist_index = len(self._history)
        self._hist_index = max(0, min(len(self._history), self._hist_index + delta))
        text = self._draft if self._hist_index >= len(self._history) else self._history[self._hist_index]
        self.load_text(text)
        last_row = text.count("\n")
        self.move_cursor((last_row, len(text.splitlines()[-1]) if text else 0))
