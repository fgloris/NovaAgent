import json
import time
from pathlib import Path

from nova_agentos.api_logger import ApiLogger
from nova_agentos.memory import Compactor, SessionManager, TaskMemory


def test_session_lifecycle_and_task_events(tmp_path):
    manager = SessionManager(tmp_path / "sessions")
    session = manager.start("demo")
    task = manager.create_task(session.session_id, "pick cup")
    task.add_event("assistant_text", content="planning")
    task.add_event("tool_feedback", tool_name="grasp", status="running", message="moving")
    task.add_event("tool_result", tool_name="grasp", success=False, summary="failed")
    manager.save_task(task)
    manager.end(session.session_id)
    assert manager.resume(session.session_id).status == "active"
    loaded = manager.load_task(session.session_id, task.task_id)
    assert [event["type"] for event in loaded.events] == [
        "assistant_text",
        "tool_feedback",
        "tool_result",
    ]


def test_missing_and_corrupt_session_are_explicit(tmp_path):
    manager = SessionManager(tmp_path)
    try:
        manager.resume("sess_missing")
        assert False
    except FileNotFoundError as exc:
        assert "session 不存在" in str(exc)
    session_dir = tmp_path / "sess_bad"
    session_dir.mkdir()
    (session_dir / "session.json").write_text("{", encoding="utf-8")
    try:
        manager.resume("sess_bad")
        assert False
    except ValueError as exc:
        assert "session 文件损坏" in str(exc)


def test_compactor_keeps_current_and_summarizes_history(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.start()
    old = manager.create_task(session.session_id, "old task")
    old.add_event("tool_result", tool_name="move", success=True, summary="ok")
    old.finish("success", "done")
    manager.save_task(old)
    current = manager.create_task(session.session_id, "current task")
    payload, compacted = Compactor(budget_tokens=10, max_recent_tasks=0).compact([old], current)
    assert compacted
    assert current.task_id not in json.dumps(payload)
    assert "old task" in json.dumps(payload)


def test_api_logger_pairs_events_redacts_and_saves_image(tmp_path):
    logger = ApiLogger(directory=tmp_path, images=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "secret"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,SGVsbG8="}},
            ],
        }
    ]
    logger.request("req_test", {"task_id": "task_1"}, messages, [], {"messages": messages})
    logger.response("req_test", {}, {"choices": [{"message": {"tool_calls": []}}]}, http_status=200)
    log = next((tmp_path / time.strftime("%Y-%m-%d")).glob("*.jsonl"))
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [line["type"] for line in lines] == ["request", "response"]
    assert "Authorization" not in log.read_text(encoding="utf-8")
    assert list((tmp_path / "images" / "req_test").glob("*.jpg"))


def test_session_task_events_and_context_survive_manager_restart(tmp_path):
    first = SessionManager(tmp_path)
    session = first.start("restart")
    task = first.create_task(session.session_id, "move cup")
    task.add_event("tool_result", tool_name="move_eef", success=True, summary="done")
    first.save_task(task)
    first.save_context(session.session_id, {"robot_context_schema": "robot_context_v1"})

    restarted = SessionManager(tmp_path)
    loaded_session = restarted.resume(session.session_id)
    loaded_task = restarted.load_task(session.session_id, task.task_id)
    assert loaded_session.context == {"robot_context_schema": "robot_context_v1"}
    assert loaded_task.instruction == "move cup"
    assert loaded_task.events[0]["tool_name"] == "move_eef"
