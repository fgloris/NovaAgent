from types import SimpleNamespace

import pytest

from nova_agentos.agent_loop import AgentLoop
from nova_agentos.memory import ContextBuilder, SessionManager


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


def test_agent_loop_injects_tool_images_as_one_user_message(tmp_path):
    manager = SessionManager(tmp_path)
    session = manager.start()
    task = manager.create_task(session.session_id, "draw the frame")
    llm = _ToolThenDoneLLM()
    loop = AgentLoop(llm, _Skills(), _ImageAdapter(), session_manager=manager)
    loop._running = True
    loop.queue.put((task.task_id, session.session_id, task.instruction))
    loop.queue.put(None)
    loop._run()

    assert len(llm.messages) >= 2
    second_round = llm.messages[1]
    image_parts = [
        part
        for message in second_round
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert image_parts, "expected an injected image user message"
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

