import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from nova_agentos.agent_loop import AgentLoop, TaskRunner
from nova_agentos.memory import ContextBuilder, ImageMemory, SessionManager


class _Skills:
    def index_text(self):
        return ""


class _Adapter:
    def fetch_tools(self):
        return []


class _LLM:
    def chat(self, messages, **kwargs):
        del messages, kwargs
        return SimpleNamespace(content="complete", reasoning_content="", tool_calls=[])


class _CapturingLLM:
    def __init__(self):
        self.messages = None

    def chat(self, messages, **kwargs):
        del kwargs
        self.messages = messages
        return SimpleNamespace(content="complete", reasoning_content="", tool_calls=[])


@pytest.mark.parametrize("with_provider", [False, True])
def test_agent_loop_starts_with_optional_robot_context(tmp_path, with_provider):
    manager = SessionManager(tmp_path)
    session = manager.start()
    task = manager.create_task(session.session_id, "pick the cup")
    provider = (
        lambda: {
            "context_schema": "robot_context_v1",
            "context_markdown": "PandaOmron constraints",
            "context_json": {"robot_type": "PandaOmron"},
            "description_sha256": "abc",
        }
    ) if with_provider else None
    loop = AgentLoop(
        _LLM(), _Skills(), _Adapter(), session_manager=manager,
        robot_context_provider=provider,
    )
    loop._running = True
    loop.queue.put((task.task_id, session.session_id, task.instruction))
    loop.queue.put(None)
    loop._run()
    assert manager.load_task(session.session_id, task.task_id).outcome == "success"


def test_system_prompt_mentions_image_memory_depths(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.start()
    task = manager.create_task(session.session_id, "draw")
    memory = ImageMemory(tmp_path / "memory", processed_max=9)
    llm = _CapturingLLM()
    loop = AgentLoop(
        llm, _Skills(), _Adapter(), session_manager=manager,
        image_memory_factory=lambda _sid: memory,
        image_history_depth=5, image_processed_depth=2,
    )
    loop._running = True
    loop.queue.put((task.task_id, session.session_id, task.instruction))
    loop.queue.put(None)
    loop._run()
    system = llm.messages[0]["content"]
    assert "图像记忆" in system
    assert "9" in system and "2" in system and "5" in system


def test_context_builder_injects_robot_and_instruction_but_persists_only_identity(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.start()
    task = manager.create_task(session.session_id, "open the cabinet")
    context = {
        "context_schema": "robot_context_v1",
        "context_markdown": "PandaOmron constraints",
        "context_json": {"large": "robot description"},
        "description_sha256": "abc",
    }
    messages = ContextBuilder().build(session, task, None, [], robot_context=context)
    assert "PandaOmron constraints" in messages[0]["content"]
    assert "open the cabinet" in messages[1]["content"]
    assert session.context["robot_description_sha256"] == "abc"
    assert "context_json" not in session.context


class _ImageAdapter:
    def fetch_tools(self):
        return []

    def execute(self, name, params, trace_id, timeout_sec=120.0, feedback_callback=None):
        return {
            "ok": True,
            "status": "drew successfully",
            "images": {"viz_1": "data:image/jpeg;base64,AAAA"},
        }


class _ToolThenDoneLLM:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def chat(self, messages, **kwargs):
        del kwargs
        self.messages.append(messages)
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(
                content="drawing",
                reasoning_content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "visualize_frame", "arguments": "{}"},
                }],
            )
        return SimpleNamespace(content="done", reasoning_content="", tool_calls=[])


def test_agent_loop_injects_processed_image_into_context(tmp_path):
    from nova_agentos.memory import ImageMemory

    manager = SessionManager(tmp_path)
    session = manager.start()
    task = manager.create_task(session.session_id, "draw the frame")
    memory = ImageMemory(tmp_path / "memory")
    llm = _ToolThenDoneLLM()
    loop = AgentLoop(
        llm, _Skills(), _ImageAdapter(), session_manager=manager,
        image_memory_factory=lambda _sid: memory,
    )
    loop._running = True
    loop.queue.put((task.task_id, session.session_id, task.instruction))
    loop.queue.put(None)
    loop._run()

    assert len(llm.messages) >= 2
    # 工具返回图落盘为 processed,并在下一轮 context 中作为图像段注入
    assert len(memory.processed_records()) == 1
    second_round = llm.messages[1]
    image_parts = [
        part
        for message in second_round
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert image_parts, "expected a processed image injected into context"
    tool_messages = [m for m in second_round if isinstance(m, dict) and m.get("role") == "tool"]
    assert tool_messages and "data:image" not in tool_messages[0]["content"]


def test_agent_loop_injects_both_observation_providers(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.start()
    task = manager.create_task(session.session_id, "move the cup")
    llm = _CapturingLLM()
    loop = AgentLoop(
        llm,
        _Skills(),
        _Adapter(),
        session_manager=manager,
        observation_provider=lambda: {"role": "user", "content": "VISION_FRAME"},
        robot_state_provider=lambda: {"role": "user", "content": "ROBOT_STATE_FRAME"},
    )
    loop._running = True
    loop.queue.put((task.task_id, session.session_id, task.instruction))
    loop.queue.put(None)
    loop._run()

    dumped = [str(message) for message in llm.messages]
    assert any("VISION_FRAME" in text for text in dumped)
    assert any("ROBOT_STATE_FRAME" in text for text in dumped)



def _runner_with_memory(tmp_path):
    memory = ImageMemory(tmp_path)
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    memory.refresh_current({"camA": (frame, 100.0)})
    memory.save_history("camA", frame, 100.0)
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG")
    processed = memory.save_processed(buf.getvalue(), "camA", "visualize_grid", {}, "", 101.0)
    runner = object.__new__(TaskRunner)
    runner.images = memory
    runner.image_history_depth = 4
    runner.image_processed_depth = 3
    runner.frame_provider = None
    runner.task = SimpleNamespace(task_id="t1", add_event=lambda *a, **k: None)
    runner.manager = SimpleNamespace(save_task=lambda *a, **k: None)
    return runner, processed


def test_local_image_tools_list_and_fetch(tmp_path):
    runner, processed = _runner_with_memory(tmp_path)
    listing = runner._list_accessible_images({})
    assert processed.url in listing and "visualize_grid" in listing

    before = len(runner.images.processed_records())
    text, images = runner._fetch_history_image({"time": 100.0, "topic": "camA"})
    assert images == {}  # 不再走 runtime 注入,靠 processed 段
    records = runner.images.processed_records()
    assert len(records) == before + 1
    fetched = records[-1]
    assert fetched.tool == "fetch_history_image"
    assert "fetch_history_image" in fetched.note
    assert fetched.base  # 指向原历史图
    assert "fetch_history_image" in text


def test_fetch_image_from_url_promotes_record_to_processed(tmp_path):
    runner, processed = _runner_with_memory(tmp_path)
    before = len(runner.images.processed_records())
    text, images = runner._fetch_image_from_url({"ref": processed.url})
    assert images == {}
    records = runner.images.processed_records()
    assert len(records) == before + 1
    fetched = records[-1]
    assert fetched.tool == "fetch_image_from_url"
    assert fetched.base == processed.url
    assert "fetch_image_from_url" in text
    assert runner._fetch_image_from_url({"ref": "file://processed/missing.jpg"})[0].startswith("未找到图像")


class _AbortAdapter:
    def __init__(self, abort=True):
        self.abort = abort

    def execute(self, name, params, trace_id, timeout_sec=120.0, feedback_callback=None):
        return {"ok": True, "abort_previous_processed": self.abort}


def test_run_tool_abort_previous_processed_clears_processed(tmp_path):
    runner, processed = _runner_with_memory(tmp_path)
    path = runner.images.path_of(processed)
    runner.adapter = _AbortAdapter(abort=True)
    text, images = runner._run_tool("move_eef", {"waypoints": []})
    assert images == {}
    assert runner.images.processed_records() == []
    assert not path.exists()
    assert "abort_previous_processed" not in text


def test_run_tool_keeps_processed_when_abort_false(tmp_path):
    runner, processed = _runner_with_memory(tmp_path)
    runner.adapter = _AbortAdapter(abort=False)
    runner._run_tool("move_eef", {"waypoints": []})
    assert len(runner.images.processed_records()) == 1


def test_image_context_dedups_repeated_urls(tmp_path):
    runner, _ = _runner_with_memory(tmp_path)
    memory = runner.images
    history = memory.history_records("camA", 10)[0]
    with memory._lock:
        memory._history["camA"].extend([history, history])
    parts = runner._image_context()
    urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
    assert len(urls) == len(set(urls))
    assert urls.count(memory.data_url(history)) == 1


def test_image_context_limits_processed_render_depth(tmp_path):
    runner, _ = _runner_with_memory(tmp_path)
    memory = runner.images
    runner.image_processed_depth = 2
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    for i in range(2):
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG")
        memory.save_processed(buf.getvalue(), "camA", "visualize_grid", {}, "", 102.0 + i)
    processed = memory.processed_records()
    assert len(processed) == 3
    parts = runner._image_context()
    # current(1) + 最近 2 张 processed + history(1)
    assert len([p for p in parts if p.get("type") == "image_url"]) == 4
    summary = "\n".join(
        p["text"] for p in parts if p.get("type") == "text" and "更早的工具返回图" in p["text"]
    )
    assert processed[0].url in summary
    assert processed[1].url not in summary
    assert processed[2].url not in summary
    # 摘要中的旧图仍可被解析引用(继续叠画)
    assert memory.find(processed[0].url) is not None


def test_activate_session_preheats_image_memory(tmp_path):
    memory = ImageMemory(tmp_path)
    loop = AgentLoop(
        _LLM(), _Skills(), _Adapter(), session_manager=SessionManager(tmp_path),
        image_memory_factory=lambda _sid: memory,
    )
    assert loop.current_memory() is None
    loop.activate_session("sess_x")
    assert loop.current_memory() is memory
    loop.deactivate_session("sess_x")
    assert loop.current_memory() is None
