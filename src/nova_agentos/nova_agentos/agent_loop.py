"""Serial AgentOS task runtime."""
from __future__ import annotations

import json
import queue
import threading
from typing import Callable

from nova_common.llm_client import LLMClient

from nova_agentos.mcp_adapter import McpAdapter, to_llm_tools
from nova_agentos.memory import Compactor, ContextBuilder, SessionManager, TaskMemory
from nova_agentos.skill_store import SkillStore

MAX_STEPS_PER_TASK = 20
MAX_TOOL_FAILS = 3

SYSTEM_BASE = (
    "你是具身机器人 NovaAgent 的 VLM agent。理解用户指令和当前多摄像机画面，"
    "通过调用工具逐步执行任务。每次调用一个工具，根据工具返回结果决定下一步，"
    "任务完成时必须调用 finish 并给出总结。"
)

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "宣告当前任务完成,给出最终总结",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "任务完成总结"}},
            "required": ["summary"],
        },
    },
}

LOAD_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": "加载指定 skill 的领域经验正文",
        "parameters": {
            "type": "object",
            "properties": {"skill": {"type": "string", "description": "skill 名称"}},
            "required": ["skill"],
        },
    },
}

# 运行单个任务
class TaskRunner:
    def __init__(
        self,
        task: TaskMemory,
        session_manager: SessionManager,
        llm: LLMClient,
        skills: SkillStore,
        adapter: McpAdapter,
        context_builder: ContextBuilder,
        on_state: Callable[[str, str, str, bool, str, str], None] | None,
        observation_provider: Callable[[], dict | None] | None,
        robot_context_provider: Callable[[], dict | None] | None = None,
    ) -> None:
        self.task = task
        self.manager = session_manager
        self.llm = llm
        self.skills = skills
        self.adapter = adapter
        self.context_builder = context_builder
        self.on_state = on_state
        self.observation_provider = observation_provider
        self.robot_context_provider = robot_context_provider
        self.runtime_messages: list[dict] = []

    def run(self) -> None:
        self._event("status", "working", f"收到指令: {self.task.instruction}")
        try:
            descriptors = self.adapter.fetch_tools()
            tools = [LOAD_SKILL_TOOL, FINISH_TOOL] + to_llm_tools(descriptors)
            fails = 0
            for round_no in range(1, MAX_STEPS_PER_TASK + 1):
                # 构造上下文： 会话 = 工具 + 先前任务上下文 + 系统提示 + Runtime Message
                session = self.manager.get(self.task.session_id)
                previous = self.manager.tasks(self.task.session_id)
                session.context["_tasks"] = [
                    item.to_dict() for item in previous if item.task_id != self.task.task_id
                ]
                session.context["system_prompt"] = SYSTEM_BASE
                messages = self.context_builder.build(
                    session,
                    self.task,
                    self._observation(),
                    tools,
                    skill_index=self.skills.index_text(),
                    robot_context=self.robot_context_provider() if self.robot_context_provider else None,
                )
                messages.extend(self.runtime_messages)
                self.manager.save_context(self.task.session_id, session.context)
                result = self.llm.chat(
                    messages,
                    tools=tools,
                    task_id=self.task.task_id,
                    session_id=self.task.session_id,
                )
                self.task.add_event(
                    "assistant_text",
                    content=result.content or "",
                    reasoning_content=result.reasoning_content or "",
                    round=round_no,
                )
                self.manager.save_task(self.task)
                self.runtime_messages.append(
                    {
                        "role": "assistant",
                        "content": result.content or "",
                        "tool_calls": result.tool_calls,
                        **(
                            {"reasoning_content": result.reasoning_content}
                            if result.reasoning_content
                            else {}
                        ),
                    }
                )
                if result.content:
                    self._event("text", "working", result.content)
                if not result.tool_calls:
                    self._finish("success", result.content or "模型返回文本，任务等待后续指令")
                    return
                
                # 处理 tool calls
                for tool_call in result.tool_calls:
                    name = tool_call["function"]["name"]
                    args = self._parse_args(tool_call)
                    if name == "finish":
                        summary = args.get("summary", "")
                        self.task.add_event(
                            "tool_result", tool_name="finish", success=True, summary=summary
                        )
                        self.manager.save_task(self.task)
                        self._finish("success", summary)
                        return
                    args_text = json.dumps(args, ensure_ascii=False)
                    self.task.add_event("tool_call", tool_name=name, params=args)
                    self.manager.save_task(self.task)
                    self._event("tool_call", "working", f"调用 {name}: {args_text}")
                    content = self._run_tool(name, args)
                    failed = content.startswith("工具执行失败")
                    self.task.add_event(
                        "tool_result",
                        tool_name=name,
                        success=not failed,
                        summary=content[:2000],
                    )
                    self.manager.save_task(self.task)
                    self.runtime_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": content,
                        }
                    )
                    self._event("tool_result", "working", f"{name} -> {content[:200]}")
                    if failed:
                        fails += 1
                        if fails >= MAX_TOOL_FAILS:
                            self._finish("failed", f"工具连续失败 {fails} 次: {content}")
                            return
                    else:
                        fails = 0
            self._finish("failed", f"超过单任务最大步数 {MAX_STEPS_PER_TASK}")
        except Exception as exc:
            self._finish("failed", f"agent loop 异常: {exc}")

    def _run_tool(self, name: str, args: dict) -> str:
        try:
            if name == "load_skill":
                skill = args.get("skill", "")
                contents = self.skills.load([skill])
                return contents.get(skill) or f"未找到 skill: {skill}"

            def feedback(status: str, message: str) -> None:
                self.task.add_event(
                    "tool_feedback", tool_name=name, status=status, message=message
                )
                self.manager.save_task(self.task)
                self._event("tool_feedback", "working", f"{name}: {message}")

            result = self.adapter.execute(
                name,
                args,
                trace_id=self.task.task_id,
                timeout_sec=300.0,
                feedback_callback=feedback,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return f"工具执行失败: {exc}"

    def _observation(self) -> dict | None:
        if self.observation_provider is None:
            return None
        try:
            return self.observation_provider()
        except Exception as exc:
            return {"role": "user", "content": f"# 当前环境视觉观测\n读取失败: {exc}"}

    @staticmethod
    def _parse_args(tool_call: dict) -> dict:
        try:
            return json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, TypeError, KeyError):
            return {}

    # 用于debug, 向 /agent_msg 发送话题
    def _event(self, kind: str, status: str, message: str, done: bool = False) -> None:
        if self.on_state:
            self.on_state(self.task.task_id, self.task.session_id, status, message, done, kind)

    # 结束此次任务
    def _finish(self, outcome: str, summary: str) -> None:
        self.task.finish(outcome, summary)
        self.manager.save_task(self.task)
        session = self.manager.get(self.task.session_id, allow_ended=True)
        session.context["system_prompt"] = SYSTEM_BASE
        session.context["_tasks"] = [
            item.to_dict() for item in self.manager.tasks(self.task.session_id)
        ]
        self.manager.save_context(self.task.session_id, session.context)
        self._event("status", "done" if outcome == "success" else "failed", summary, True)


class AgentLoop:
    # 全局任务 FIFO 队列

    def __init__(
        self,
        llm: LLMClient,
        skills: SkillStore,
        adapter: McpAdapter,
        session_manager: SessionManager | None = None,
        on_state: Callable[[str, str, str, bool, str, str], None] | None = None,
        observation_provider: Callable[[], dict | None] | None = None,
        robot_context_provider: Callable[[], dict | None] | None = None,
        context_budget_tokens: int = 12000,
        context_compaction_enabled: bool = True,
        max_recent_tasks: int = 8,
    ) -> None:
        self.llm = llm
        self.skills = skills
        self.adapter = adapter
        self.session_manager = session_manager or SessionManager()
        self.on_state = on_state
        self.observation_provider = observation_provider
        self.context_builder = ContextBuilder(
            Compactor(context_budget_tokens, context_compaction_enabled, max_recent_tasks)
        )
        self.queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.queue.put(None)
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)

    def submit(self, task_id: str, session_id: str, instruction: str) -> None:
        self.queue.put((task_id, session_id, instruction))

    def _run(self) -> None:
        while self._running:
            item = self.queue.get()
            if item is None:
                break
            task_id, session_id, _instruction = item
            try:
                task = self.session_manager.load_task(session_id, task_id)
                TaskRunner(
                    task,
                    self.session_manager,
                    self.llm,
                    self.skills,
                    self.adapter,
                    self.context_builder,
                    self.on_state,
                    self.observation_provider,
                    robot_context_provider=self.robot_context_provider,
                ).run()
            except Exception as exc:
                if self.on_state:
                    self.on_state(
                        task_id, session_id, "failed", f"任务启动失败: {exc}", True, "status"
                    )
