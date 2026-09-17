"""NovaAgent 交互式 TUI:会话自动创建、对话渲染、命令处理。"""
from __future__ import annotations

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static

from . import theme
from .client import RosClient
from .widgets.conversation import ConversationView
from .widgets.messages import SystemLine, UserMessage, build_event_widget
from .widgets.prompt import Prompt
from .widgets.sessions import SessionListScreen
from .widgets.statusbar import StatusBar

COMMANDS: dict[str, str] = {
    "/help": "显示命令帮助",
    "/session list": "列出所有 session",
    "/session new [name]": "新建 session",
    "/session resume <id>": "恢复指定 session",
    "/session rename <name>": "重命名当前 session",
    "/session info": "查看当前 session",
    "/session end": "结束当前 session 并新建",
    "/ping": "测 LLM provider 延迟",
    "/env": "查询仿真环境规格",
    "/reset": "重置仿真环境",
    "/clear": "清空会话显示",
    "/quit": "退出",
}

HELP_TEXT = "命令:\n" + "\n".join(f"  {cmd:<26} {desc}" for cmd, desc in COMMANDS.items())


class NovaCliApp(App):
    """NovaAgent 终端 UI。"""

    CSS = theme.APP_CSS
    BINDINGS = [Binding("ctrl+q", "quit_app", "退出", priority=True)]
    TITLE = "NovaAgent"

    def __init__(self, client: RosClient, resume_session: str | None = None) -> None:
        super().__init__()
        self.client = client
        self._resume_session = resume_session
        self.session_id = ""
        self.session_name = ""
        self._active_tasks = 0

    # ---------- 布局 ----------
    def compose(self) -> ComposeResult:
        yield ConversationView(id="conversation")
        yield Static("Enter 发送 · Shift+Enter 换行 · Ctrl-C 复制 · /help 命令", id="hint")
        yield Prompt(id="prompt")
        yield StatusBar()

    @property
    def conversation(self) -> ConversationView:
        return self.query_one("#conversation", ConversationView)

    @property
    def prompt(self) -> Prompt:
        return self.query_one("#prompt", Prompt)

    @property
    def statusbar(self) -> StatusBar:
        return self.query_one(StatusBar)

    def on_mount(self) -> None:
        """挂载后初始化会话并开始轮询 agent 消息。"""
        self.conversation.add(
            SystemLine("NovaAgent CLI 已连接。输入指令开始任务,或 /help 查看命令。", color=theme.SECONDARY, icon="◆")
        )
        self.prompt.focus()
        self._init_session()
        self.set_interval(0.05, self._drain)

    # ---------- 会话 ----------
    @work(thread=True, exclusive=True)
    def _init_session(self) -> None:
        """启动时自动创建新会话;若指定则恢复。"""
        try:
            if self._resume_session:
                sid, name = self.client.resume_session(self._resume_session)
                note = f"已恢复会话 {name} ({sid})"
            else:
                sid, name = self.client.start_session("未命名")
                note = f"已自动创建会话 ({sid}),首个任务后自动命名"
            self.call_from_thread(self._set_session, sid, name, note)
        except Exception as exc:
            self.call_from_thread(self._append_system, f"创建会话失败: {exc}", theme.ERROR)

    def _set_session(self, session_id: str, name: str, note: str = "") -> None:
        self.session_id, self.session_name = session_id, name
        self.statusbar.set_session(name, session_id)
        if note:
            self.conversation.add(SystemLine(note, color=theme.SECONDARY, icon="✎"))

    # ---------- 事件轮询 ----------
    def _drain(self) -> None:
        """把队列里的 agent 消息渲染到会话区。"""
        for msg in self.client.poll():
            if msg.session_id and msg.session_id != self.session_id:
                continue
            if msg.kind == "session_renamed":
                self.session_name = msg.message
                self.statusbar.set_session(self.session_name, self.session_id)
                self.conversation.add(SystemLine(f"会话已命名: {msg.message}", color=theme.SECONDARY, icon="✎"))
                continue
            widget = build_event_widget(msg)
            if widget is not None:
                self.conversation.add(widget)
            if msg.done:
                self._active_tasks = max(0, self._active_tasks - 1)
                self.statusbar.set_busy(self._active_tasks > 0)

    def _append_system(self, text: str, color: str = theme.MUTED) -> None:
        self.conversation.add(Static(Text(text, style=color), classes="event event-system"))

    # ---------- 输入 ----------
    @on(Prompt.Submitted)
    def _on_submitted(self, event: Prompt.Submitted) -> None:
        value = event.value
        self.prompt.add_history(value)
        self.prompt.clear()
        self.statusbar.set_message("就绪")
        if value.startswith("/"):
            self._run_command(value)
        else:
            self.conversation.add(UserMessage(value))
            self._send_task(value)

    @on(Prompt.Changed)
    def _on_changed(self) -> None:
        """输入以 / 开头时在提示行给出命令候选。"""
        text = self.prompt.text
        if text.startswith("/") and " " not in text:
            matches = [c for c in COMMANDS if c.startswith(text)]
            if matches:
                self.query_one("#hint", Static).update(Text("  ".join(matches[:6]), style=theme.MUTED))
                return
        self.query_one("#hint", Static).update(
            Text("Enter 发送 · Shift+Enter 换行 · Ctrl-C 复制 · /help 命令", style=theme.MUTED)
        )

    # ---------- 任务 ----------
    @work(thread=True, exclusive=True)
    def _send_task(self, instruction: str) -> None:
        """提交任务(后台线程),成功后标记忙碌。"""
        if not self.session_id:
            self.call_from_thread(self._append_system, "当前没有 active session", theme.ERROR)
            return
        try:
            task_id = self.client.send_message(self.session_id, instruction)
            self._active_tasks += 1
            self.call_from_thread(self.statusbar.set_busy, True)
            self.call_from_thread(self._append_system, f"已入队 task_id={task_id}", theme.MUTED)
        except Exception as exc:
            self.call_from_thread(self._append_system, f"发送失败: {exc}", theme.ERROR)

    # ---------- 命令 ----------
    def _run_command(self, line: str) -> None:
        cmd, *rest = line.split(maxsplit=1)
        arg = rest[0].strip() if rest else ""
        if cmd == "/help":
            self._append_system(HELP_TEXT, theme.MUTED)
        elif cmd == "/clear":
            self.conversation.remove_children()
        elif cmd in ("/quit", "/exit"):
            self.action_quit_app()
        elif cmd == "/ping":
            self._run_ping()
        elif cmd == "/env":
            self._run_env()
        elif cmd == "/reset":
            self._run_reset()
        elif cmd == "/session":
            self._run_session(arg)
        else:
            self._append_system(f"未知命令: {cmd}(/help 查看)", theme.ERROR)

    def _run_session(self, arg: str) -> None:
        action, _, value = arg.partition(" ")
        value = value.strip()
        if action == "list":
            self._list_sessions()
        elif action == "new":
            self._new_session(value or "未命名")
        elif action == "resume" and value:
            self._resume(value)
        elif action == "rename" and value:
            self._rename(value)
        elif action == "info":
            self._append_system(
                f"session={self.session_id or '(无)'} name={self.session_name or '(未命名)'}", theme.MUTED
            )
        elif action == "end":
            self._end_and_new()
        else:
            self._append_system("用法: /session list | new [name] | resume <id> | rename <name> | info | end", theme.MUTED)

    @work(thread=True, exclusive=True)
    def _list_sessions(self) -> None:
        try:
            sessions = self.client.list_sessions()
            self.call_from_thread(self._show_sessions, sessions)
        except Exception as exc:
            self.call_from_thread(self._append_system, f"获取会话列表失败: {exc}", theme.ERROR)

    def _show_sessions(self, sessions: list[dict[str, str]]) -> None:
        self.push_screen(SessionListScreen(sessions), callback=self._on_session_picked)

    def _on_session_picked(self, session_id: str | None) -> None:
        if session_id:
            self._resume(session_id)

    @work(thread=True, exclusive=True)
    def _resume(self, session_id: str) -> None:
        try:
            sid, name = self.client.resume_session(session_id)
            self.call_from_thread(self._set_session, sid, name, f"已恢复会话 {name} ({sid})")
        except Exception as exc:
            self.call_from_thread(self._append_system, f"恢复失败: {exc}", theme.ERROR)

    @work(thread=True, exclusive=True)
    def _new_session(self, name: str) -> None:
        try:
            sid, nm = self.client.start_session(name)
            self.call_from_thread(self._set_session, sid, nm, f"已新建会话 {nm} ({sid})")
        except Exception as exc:
            self.call_from_thread(self._append_system, f"新建失败: {exc}", theme.ERROR)

    @work(thread=True, exclusive=True)
    def _rename(self, name: str) -> None:
        try:
            self.client.rename_session(self.session_id, name)
            self.call_from_thread(self._set_session, self.session_id, name, f"已重命名为 {name}")
        except Exception as exc:
            self.call_from_thread(self._append_system, f"重命名失败: {exc}", theme.ERROR)

    @work(thread=True, exclusive=True)
    def _end_and_new(self) -> None:
        try:
            self.client.end_session(self.session_id)
            sid, nm = self.client.start_session("未命名")
            self.call_from_thread(self._set_session, sid, nm, f"已结束旧会话并新建 ({sid})")
        except Exception as exc:
            self.call_from_thread(self._append_system, f"操作失败: {exc}", theme.ERROR)

    @work(thread=True, exclusive=True)
    def _run_ping(self) -> None:
        try:
            self.call_from_thread(self._append_system, self.client.ping_llm(), theme.MUTED)
        except Exception as exc:
            self.call_from_thread(self._append_system, f"ping 失败: {exc}", theme.ERROR)

    @work(thread=True, exclusive=True)
    def _run_env(self) -> None:
        try:
            self.call_from_thread(self._append_system, self.client.env_info(), theme.MUTED)
        except Exception as exc:
            self.call_from_thread(self._append_system, f"获取环境信息失败: {exc}", theme.ERROR)

    @work(thread=True, exclusive=True)
    def _run_reset(self) -> None:
        try:
            self.call_from_thread(self._append_system, self.client.reset_env(), theme.MUTED)
        except Exception as exc:
            self.call_from_thread(self._append_system, f"重置失败: {exc}", theme.ERROR)

    # ---------- 退出 ----------
    def action_quit_app(self) -> None:
        """结束当前会话后退出。"""
        self._finish_and_exit()

    @work(thread=True, exclusive=True)
    def _finish_and_exit(self) -> None:
        try:
            self.client.end_session(self.session_id)
        except Exception:
            pass
        self.call_from_thread(self.exit)
