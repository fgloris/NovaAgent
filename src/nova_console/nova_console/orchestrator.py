# 会话编排:加载 profile,按 depends_on DAG 启动;某会话就绪后尝试启动依赖者。
from __future__ import annotations

import threading

from nova_console.config import load_profiles
from nova_console.session import Session


class Orchestrator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.sessions: dict[str, Session] = {}
        self.profiles: dict = {}
        self.active_profile: str | None = None
        self.event_sink = None  # callable(dict):可选的事件出口(输出/状态),由 server 注入

    def load_profiles(self, config_path: str | None = None) -> dict:
        self.profiles = load_profiles(config_path)
        return self.profiles

    # 启动整套 profile:重建会话表(旧的先全停)
    def start_profile(self, name: str) -> None:
        pf = self.profiles.get(name)
        if pf is None:
            raise ValueError(f"未知 profile: {name}")
        self.stop_all()
        with self._lock:
            self.sessions = {}
            for s in pf["sessions"]:
                sess = Session(s)
                sess.on_status(self._on_session_status)
                sess.on_output(self._on_session_output)
                self.sessions[sess.id] = sess
            self.active_profile = name
        for sess in list(self.sessions.values()):
            self._maybe_start(sess)

    def _emit(self, evt: dict) -> None:
        if self.event_sink:
            try:
                self.event_sink(evt)
            except Exception:
                pass

    def _on_session_output(self, sess: Session, text: str) -> None:
        self._emit(
            {
                "type": "session_output",
                "id": sess.id,
                "name": sess.name,
                "data": text,
                "seq": sess._out_seq,
            }
        )

    # 会话是否已就绪:ready / 无 wait_for 的 running / 正常退出(exit 0)
    @staticmethod
    def _is_ready(sess: Session) -> bool:
        if sess.status == "ready":
            return True
        if sess.status == "running" and sess.wait_re is None:
            return True
        if sess.status == "exited" and sess.exit_code == 0:
            return True
        return False

    def _maybe_start(self, sess: Session) -> None:
        if sess.status != "created":
            return
        if all(self._is_ready(self.sessions[d]) for d in sess.depends_on):
            sess.start()

    def _on_session_status(self, sess: Session) -> None:
        self._emit(
            {
                "type": "session_status",
                "id": sess.id,
                "name": sess.name,
                "status": sess.status,
                "exit_code": sess.exit_code,
                "depends_on": sess.depends_on,
            }
        )
        # 状态变化 -> 尝试启动所有还处于 created 的依赖者
        with self._lock:
            targets = [s for s in self.sessions.values() if s.status == "created"]
        for s in targets:
            self._maybe_start(s)

    def stop_all(self) -> None:
        for sess in list(self.sessions.values()):
            sess.stop()

    def stop(self, sid: str) -> None:
        self.sessions[sid].stop()

    def restart(self, sid: str) -> None:
        self.sessions[sid].restart()

    def send_input(self, sid: str, text: str) -> None:
        self.sessions[sid].send_input(text)

    def list_sessions(self) -> list[dict]:
        return [s.snapshot() for s in self.sessions.values()]
