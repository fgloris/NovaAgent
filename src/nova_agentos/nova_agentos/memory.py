"""Short-lived session and task memory for AgentOS."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
import calendar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


@dataclass
class TaskMemory:
    task_id: str
    session_id: str
    instruction: str
    events: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = "running"
    summary: str = ""
    started_at: str = field(default_factory=_now)
    ended_at: str = ""
    duration_ms: int = 0
    _started_epoch: float = field(default_factory=time.time, repr=False)

    def add_event(self, event_type: str, **values: Any) -> dict:
        event = {"type": event_type, **values, "time": _now()}
        self.events.append(event)
        return event

    def finish(self, outcome: str, summary: str = "") -> None:
        self.outcome = outcome
        self.summary = summary
        self.ended_at = _now()
        self.duration_ms = round((time.time() - self._started_epoch) * 1000)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "instruction": self.instruction,
            "events": self.events,
            "outcome": self.outcome,
            "summary": self.summary,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskMemory":
        task = cls(
            task_id=str(data["task_id"]),
            session_id=str(data["session_id"]),
            instruction=str(data.get("instruction", "")),
            events=list(data.get("events") or []),
            outcome=str(data.get("outcome", "running")),
            summary=str(data.get("summary", "")),
            started_at=str(data.get("started_at") or _now()),
            ended_at=str(data.get("ended_at", "")),
            duration_ms=int(data.get("duration_ms", 0)),
        )
        try:
            task._started_epoch = calendar.timegm(
                time.strptime(task.started_at, "%Y-%m-%dT%H:%M:%SZ")
            )
        except (ValueError, OverflowError):
            task._started_epoch = time.time()
        return task


@dataclass
class SessionRecord:
    session_id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    task_ids: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        default = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        self.root = Path(root_dir or Path(default) / "nova_agentos" / "sessions").expanduser()
        self._lock = threading.RLock()

    def _dir(self, session_id: str) -> Path:
        return self.root / session_id

    def _load_record(self, session_id: str) -> SessionRecord:
        path = self._dir(session_id) / "session.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SessionRecord(
                session_id=str(data["session_id"]),
                name=str(data.get("name", "")),
                status=str(data.get("status", "ended")),
                created_at=str(data["created_at"]),
                updated_at=str(data["updated_at"]),
                task_ids=list(data.get("task_ids") or []),
                context=self._load_context(session_id),
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"session 不存在: {session_id}") from None
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"session 文件损坏: {path}: {exc}") from exc

    def _save_record(self, record: SessionRecord) -> None:
        _write_json_atomic(
            self._dir(record.session_id) / "session.json",
            {
                "session_id": record.session_id,
                "name": record.name,
                "status": record.status,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "task_ids": record.task_ids,
            },
        )
        _write_json_atomic(self._dir(record.session_id) / "context.json", record.context)

    def _load_context(self, session_id: str) -> dict:
        path = self._dir(session_id) / "context.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"context 文件损坏: {path}: {exc}") from exc

    def start(self, name: str = "default") -> SessionRecord:
        with self._lock:
            stamp = _now()
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
            record = SessionRecord(
                session_id,
                name.strip() or "default",
                "active",
                stamp,
                stamp,
                context={"system_prompt": ""},
            )
            self._save_record(record)
            return record

    def resume(self, session_id: str) -> SessionRecord:
        with self._lock:
            record = self._load_record(session_id)
            record.status = "active"
            record.updated_at = _now()
            self._save_record(record)
            return record

    def end(self, session_id: str) -> SessionRecord:
        with self._lock:
            record = self._load_record(session_id)
            record.status = "ended"
            record.updated_at = _now()
            self._save_record(record)
            return record

    def get(self, session_id: str, allow_ended: bool = False) -> SessionRecord:
        with self._lock:
            record = self._load_record(session_id)
            if record.status != "active" and not allow_ended:
                raise ValueError(f"session 非 active: {session_id}")
            return record

    def create_task(self, session_id: str, instruction: str) -> TaskMemory:
        with self._lock:
            record = self.get(session_id)
            task = TaskMemory(f"task_{uuid.uuid4().hex[:12]}", session_id, instruction)
            record.task_ids.append(task.task_id)
            record.updated_at = _now()
            self._save_record(record)
            self.save_task(task)
            return task

    def save_task(self, task: TaskMemory) -> None:
        _write_json_atomic(self._dir(task.session_id) / "tasks" / f"{task.task_id}.json", task.to_dict())

    def load_task(self, session_id: str, task_id: str) -> TaskMemory:
        path = self._dir(session_id) / "tasks" / f"{task_id}.json"
        try:
            return TaskMemory.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise FileNotFoundError(f"task 不存在: {task_id}") from None
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"task 文件损坏: {path}: {exc}") from exc

    def tasks(self, session_id: str) -> list[TaskMemory]:
        record = self._load_record(session_id)
        return [self.load_task(session_id, task_id) for task_id in record.task_ids]

    def save_context(self, session_id: str, context: dict) -> None:
        record = self._load_record(session_id)
        record.context = context
        record.updated_at = _now()
        self._save_record(record)


class Compactor:
    def __init__(self, budget_tokens: int = 12000, enabled: bool = True, max_recent_tasks: int = 8):
        self.budget_tokens = max(100, int(budget_tokens))
        self.enabled = bool(enabled)
        self.max_recent_tasks = max(0, int(max_recent_tasks))

    @staticmethod
    def _estimate(value: Any) -> int:
        return max(1, len(json.dumps(value, ensure_ascii=False)) // 4)

    @staticmethod
    def summary(task: TaskMemory) -> str:
        methods = [
            str(e.get("tool_name"))
            for e in task.events
            if e.get("type") == "tool_result" and e.get("tool_name")
        ]
        method_text = ", ".join(dict.fromkeys(methods)) or "未调用工具"
        return f"用户指令: {task.instruction}\n执行方法: {method_text}\n结果: {task.outcome} - {task.summary}"

    def compact(self, tasks: list[TaskMemory], current: TaskMemory) -> tuple[list[dict], bool]:
        split = len(tasks) if self.max_recent_tasks == 0 else -self.max_recent_tasks
        detailed = [] if self.max_recent_tasks == 0 else [t.to_dict() for t in tasks[split:]]
        historical = [
            {"task_id": t.task_id, "summary": self.summary(t)} for t in tasks[:split]
        ]
        payload = {"historical_summaries": historical, "recent_tasks": detailed}
        current_cost = self._estimate(
            {"instruction": current.instruction, "events": current.events}
        )
        if not self.enabled or self._estimate(payload) + current_cost <= self.budget_tokens:
            return payload, False
        while historical and self._estimate(payload) + current_cost > self.budget_tokens:
            historical.pop(0)
            payload = {"historical_summaries": historical, "recent_tasks": detailed}
        if self._estimate(payload) + current_cost > self.budget_tokens:
            payload = {
                "historical_summaries": [],
                "recent_tasks": detailed[-1:] if detailed else [],
            }
        return payload, True


class ContextBuilder:
    def __init__(self, compactor: Compactor | None = None):
        self.compactor = compactor or Compactor()

    def build(
        self,
        session: SessionRecord,
        current: TaskMemory,
        observation: dict | None,
        tools: list,
        skill_index: str = "",
    ) -> list[dict]:
        tasks = session.context.get("_tasks", [])
        context, compacted = self.compactor.compact(
            [TaskMemory.from_dict(t) for t in tasks] if tasks else [], current
        )
        context["system_prompt"] = session.context.get("system_prompt", "")
        session.context = context
        messages = [
            {"role": "system", "content": session.context.get("system_prompt", "") or ""},
            {
                "role": "user",
                "content": (
                    f"# session 历史任务\n{json.dumps(context, ensure_ascii=False)}\n"
                    f"# skill 索引\n{skill_index or '(无)'}\n"
                    f"# 当前用户指令\n{current.instruction}\n"
                    f"# 当前任务事件\n{json.dumps(current.events, ensure_ascii=False)}"
                ),
            },
        ]
        if observation:
            messages.append(observation)
        messages.append({"role": "user", "content": f"# 可用工具 schema\n{json.dumps(tools, ensure_ascii=False)}"})
        if compacted:
            current.add_event("context_compacted", message="历史任务已按预算压缩")
        return messages
