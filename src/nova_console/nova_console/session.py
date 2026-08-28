# PTY 会话运行时:用 ptyprocess 起一个 shell 子进程,后台线程读输出。
# 支持 wait_for 就绪判断、stdin 输入、按进程组 stop/restart。
from __future__ import annotations

import codecs
import logging
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

from ptyprocess import PtyProcess

RING_LIMIT = 65536  # 内存环形缓冲字节上限(前端 tail/回看)
DEFAULT_LOG_DIR = "/tmp/nova_console_logs"


def _build_script(cfg: dict) -> str:
    parts = []
    venv = cfg.get("venv")
    if venv:
        parts.append(f"source {Path(venv).expanduser()}/bin/activate")
    for pre in cfg.get("pre") or []:
        parts.append(pre)
    parts.append(cfg["command"])
    return " && ".join(parts)


class Session:
    def __init__(self, cfg: dict, log_dir: str | None = None) -> None:
        self.cfg = cfg
        self.id: str = cfg["id"]
        self.name: str = cfg.get("name", cfg["id"])
        self.script = _build_script(cfg)
        self.wait_re = re.compile(cfg["wait_for"]) if cfg.get("wait_for") else None
        self.wait_timeout = float(cfg.get("wait_timeout_sec", 0)) or None
        self.depends_on: list[str] = list(cfg.get("depends_on") or [])
        self.status = "created"
        self.exit_code: int | None = None
        self.proc: PtyProcess | None = None
        self._lock = threading.Lock()
        self._readers: list = []
        self._watchers: list = []

        log_base = Path(log_dir or os.environ.get("NOVA_CONSOLE_LOG_DIR", DEFAULT_LOG_DIR))
        self.log_file = log_base / f"{self.id}.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self._ring: deque = deque()
        self._ring_len = 0
        self._ring_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None

        # 增量输出缓冲:每条输出块带自增 seq,前端可"从 after 之后拉取"(WS 断连时轮询兜底)
        self._out_seq = 0
        self._outs: deque = deque()
        self._outs_len = 0

    # ---------- 事件 ----------
    def on_output(self, fn) -> None:
        self._readers.append(fn)

    def on_status(self, fn) -> None:
        self._watchers.append(fn)

    def _notify_output(self, text: str) -> None:
        for fn in list(self._readers):
            try:
                fn(self, text)
            except Exception:
                logging.exception("on_output callback")

    def _notify_status(self) -> None:
        for fn in list(self._watchers):
            try:
                fn(self)
            except Exception:
                logging.exception("on_status callback")

    # ---------- 内部 ----------
    def _set_status(self, status: str) -> None:
        with self._lock:
            self.status = status
        self._notify_status()

    def _append(self, text: str) -> None:
        with self._ring_lock:
            data = text.encode("utf-8", "replace")
            self._ring.append(data)
            self._ring_len += len(data)
            while self._ring_len > RING_LIMIT and self._ring:
                self._ring_len -= len(self._ring.popleft())
            self._out_seq += 1
            self._outs.append((self._out_seq, text))
            self._outs_len += len(data)
            while self._outs_len > RING_LIMIT and len(self._outs) > 1:
                _, dropped = self._outs.popleft()
                self._outs_len -= len(dropped.encode("utf-8", "replace"))
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text)
        self._notify_output(text)

    # 返回 (当前 seq, after 之后的新输出拼接);客户端用返回的 seq 继续轮询
    def output_since(self, after: int) -> tuple[int, str]:
        with self._ring_lock:
            parts = [t for s, t in self._outs if s > after]
            return self._out_seq, "".join(parts)

    def _read_loop(self) -> None:
        proc = self.proc
        if proc is None:
            return
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while proc.isalive():
                try:
                    chunk = os.read(proc.fileno(), 4096)
                except OSError:
                    break
                if not chunk:
                    break
                self._append(dec.decode(chunk))
            # 排空剩余输出
            while True:
                try:
                    rest = os.read(proc.fileno(), 4096)
                except OSError:
                    break
                if not rest:
                    break
                self._append(dec.decode(rest))
        finally:
            self._append(dec.decode(b"", final=True))
            try:
                self.exit_code = proc.wait()
            except Exception:
                self.exit_code = None
            self._set_status("exited")
            self.proc = None

    def _wait_for_ready(self) -> None:
        t0 = time.time()
        while True:
            # 先查日志:标记可能已在进程退出前写入
            try:
                if self.wait_re and self.wait_re.search(
                    self.log_file.read_text(encoding="utf-8", errors="replace")
                ):
                    self._set_status("ready")
                    return
            except OSError:
                pass
            if self.status == "exited":
                self._set_status("failed")
                return
            if self.wait_timeout and time.time() - t0 > self.wait_timeout:
                self._set_status("failed")
                return
            time.sleep(0.2)

    # ---------- 控制 ----------
    def start(self) -> None:
        if self.proc is not None and self.proc.isalive():
            return
        workdir = self.cfg.get("workdir")
        cwd = str(Path(workdir).expanduser()) if workdir else None
        env = os.environ.copy()
        env.update(self.cfg.get("env") or {})
        self._set_status("starting")
        self.proc = PtyProcess.spawn(
            ["bash", "-lc", self.script], cwd=cwd, env=env, echo=True, dimensions=(30, 120)
        )
        self._set_status("running")
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        if self.wait_re is not None:
            threading.Thread(target=self._wait_for_ready, daemon=True).start()

    def send_input(self, text: str) -> None:
        if self.proc is not None and self.proc.isalive():
            self.proc.write(text)

    def _signal_group(self, sig: int) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
        except (ProcessLookupError, OSError):
            pass

    def stop(self, timeout_sec: float = 5.0) -> None:
        proc = self.proc
        if proc is None or not proc.isalive():
            self._set_status("stopped")
            return
        self._signal_group(2)  # SIGINT
        t0 = time.time()
        while proc.isalive() and time.time() - t0 < timeout_sec:
            time.sleep(0.1)
        if proc.isalive():
            self._signal_group(15)  # SIGTERM
            t0 = time.time()
            while proc.isalive() and time.time() - t0 < timeout_sec:
                time.sleep(0.1)
        if proc.isalive():
            self._signal_group(9)  # SIGKILL
        self._set_status("stopped")

    def restart(self) -> None:
        self.stop()
        time.sleep(0.3)
        with self._ring_lock:
            self._ring.clear()
            self._ring_len = 0
            self._out_seq = 0
            self._outs.clear()
            self._outs_len = 0
        self.exit_code = None
        try:
            self.log_file.write_text("", encoding="utf-8")
        except OSError:
            pass
        self.start()

    # ---------- 查询 ----------
    def snapshot(self) -> dict:
        with self._ring_lock:
            text = b"".join(self._ring).decode("utf-8", "replace")
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "exit_code": self.exit_code,
            "depends_on": self.depends_on,
            "command": self.cfg.get("command"),
            "script": self.script,
            "workdir": self.cfg.get("workdir"),
            "venv": self.cfg.get("venv"),
            "pre": list(self.cfg.get("pre") or []),
            "wait_for": self.cfg.get("wait_for"),
            "out_seq": self._out_seq,
            "tail": text[-4000:],
        }
