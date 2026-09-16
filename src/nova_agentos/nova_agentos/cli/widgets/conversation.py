"""会话区容器:追加消息并自动滚到底部。"""
from __future__ import annotations

from textual.widget import Widget
from textual.widgets import Markdown
from textual.containers import VerticalScroll


class ConversationView(VerticalScroll):
    """可滚动的消息列表。"""

    def add(self, widget: Widget) -> None:
        """挂载一个消息控件并滚到底部。"""
        self.mount(widget)
        if isinstance(widget, Markdown):
            self.call_after_refresh(self.scroll_end, animate=False)
        else:
            self.scroll_end(animate=False)
