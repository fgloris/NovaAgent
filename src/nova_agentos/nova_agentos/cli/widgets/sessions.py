"""会话列表弹层:展示 name + id + status,选中可恢复。"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static


class SessionListScreen(ModalScreen[str | None]):
    """模态列出全部 session;回车/点击返回所选 session_id。"""

    BINDINGS = [Binding("escape", "dismiss_none", "关闭")]

    def __init__(self, sessions: list[dict[str, str]]) -> None:
        super().__init__()
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="session-list"):
            yield Static("会话列表(回车/点击恢复,esc 关闭)", id="session-list-title")
            yield DataTable(id="session-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#session-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("名称", "session_id", "状态", "更新时间")
        for s in self._sessions:
            table.add_row(s["name"] or "(未命名)", s["id"], s["status"], s["updated_at"], key=s["id"])
        table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.dismiss(str(event.row_key.value))

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
