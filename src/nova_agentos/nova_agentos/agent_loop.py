# agent 循环:后台线程消费用户消息队列,上下文跨任务持久累积。
# 每轮让 LLM 发出函数调用(executor 工具 / load_skill / finish),执行后回填上下文继续;
# 直到 LLM 调 finish 完成任务,或只发纯文本(视为等待用户输入)。
# 不再有 DAG:单函数调用 = 一步执行,模型根据上一步结果决定下一步(闭环)。
import json
import queue
import threading
from typing import Callable

from nova_common.llm_client import LLMClient

from nova_agentos.mcp_adapter import McpAdapter, to_llm_tools
from nova_agentos.skill_store import SkillStore

# 单任务最大步数,防止模型一直调用工具不结束
MAX_STEPS_PER_TASK = 20
# 工具连续失败次数上限,超过则放弃当前任务
MAX_TOOL_FAILS = 3

SYSTEM_BASE = (
    "你是具身机器人 NovaAgent 的 agent。你需要理解用户指令,通过调用工具逐步执行任务,"
    "每次调用一个工具,根据工具返回结果决定下一步,直到任务完成。"
    "需要领域经验时先调用 load_skill 加载;任务完成时必须调用 finish 并给出总结。"
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
        "description": "加载指定 skill 的领域经验正文(名称只能来自 skill 索引),结果注入你的上下文",
        "parameters": {
            "type": "object",
            "properties": {"skill": {"type": "string", "description": "skill 名称"}},
            "required": ["skill"],
        },
    },
}


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        skills: SkillStore,
        adapter: McpAdapter,
        on_state: Callable[[str, str, str, bool, str], None] | None = None,
    ) -> None:
        # on_state(task_id, status, message, done, kind):任务消息回调,由调用方发布到话题
        self.llm = llm
        self.skills = skills
        self.adapter = adapter
        self.on_state = on_state
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_BASE}]
        self.queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.queue.put((None, None))

    def submit(self, task_id: str, instruction: str) -> None:
        self.queue.put((task_id, instruction))

    def _run(self) -> None:
        while self._running:
            task_id, instruction = self.queue.get()
            if instruction is None:
                break
            try:
                self._handle(task_id, instruction)
            except Exception as exc:
                self._emit(task_id, "failed", f"agent loop 异常: {exc}", True)

    def _repair_history(self) -> None:
        # 上一任务若因 finish 提前返回,可能留下"assistant 带 tool_calls 却缺 tool 响应"的畸形历史,
        # 下一任务第一次调 LLM 就会被 400 拒绝;这里补齐占位响应,保证每条 tool_call 都有应答。
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            answered = {
                m.get("tool_call_id")
                for m in self.messages[i + 1:]
                if m.get("role") == "tool"
            }
            for tc in msg["tool_calls"]:
                if tc.get("id") not in answered:
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "该工具调用未执行(finish 或历史遗留)",
                        }
                    )
            break

    def _handle(self, task_id: str, instruction: str) -> None:
        self._repair_history()
        skill_index = self.skills.index_text()
        descriptors = self.adapter.fetch_tools()
        tools_text = "\n".join(f"- {d.name}: {d.description}" for d in descriptors)
        context = (
            f"# 当前可用 skill 索引\n{skill_index or '(无)'}\n\n"
            f"# 当前可用工具\n{tools_text or '(无)'}\n\n"
            f"用户指令: {instruction}"
        )
        self.messages.append({"role": "user", "content": context})
        self._emit(task_id, "working", f"收到指令: {instruction}", False, kind="status")
        tools = [LOAD_SKILL_TOOL, FINISH_TOOL] + to_llm_tools(descriptors)

        fails = 0
        for round_no in range(1, MAX_STEPS_PER_TASK + 1):
            result = self.llm.chat(self.messages, tools=tools)
            print(
                f"[agent_loop] task={task_id} round={round_no} "
                f"content={result.content!r} "
                f"tool_calls={[tc['function']['name'] + ' ' + tc['function']['arguments'] for tc in result.tool_calls]}",
                flush=True,
            )
            self.messages.append(
                {
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": result.tool_calls,
                    **({"reasoning_content": result.reasoning_content} if result.reasoning_content else {}),
                }
            )
            if result.content:
                # 规划消息:模型每轮的文本(思考/说明),发布给 UI
                self._emit(task_id, "working", result.content, False, kind="text")
            if not result.tool_calls:
                # 纯文本回复:等待用户下一条消息,上下文保留
                self._emit(task_id, "working", result.content or "", False, kind="text")
                return
            for tc in result.tool_calls:
                name = tc["function"]["name"]
                args = self._parse_args(tc)
                if name == "finish":
                    # 也必须补 tool 响应,保持历史完整(否则下个任务会 400)
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": "finished"}
                    )
                    self._emit(task_id, "done", args.get("summary", ""), True, kind="status")
                    return
                args_text = json.dumps(args, ensure_ascii=False)
                self._emit(task_id, "working", f"调用 {name}: {args_text}", False, kind="tool_call")
                args = self._with_context(args)
                content = self._run_tool(name, args, task_id)
                print(f"[agent_loop] task={task_id} tool={name} result={content[:200]}", flush=True)
                if content.startswith("工具执行失败"):
                    fails += 1
                    if fails >= MAX_TOOL_FAILS:
                        self._emit(task_id, "failed", f"工具连续失败 {fails} 次: {content}", True, kind="status")
                        return
                else:
                    fails = 0
                self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
                self._emit(task_id, "working", f"{name} -> {content[:200]}", False, kind="tool_result")
        self._emit(task_id, "failed", f"超过单任务最大步数 {MAX_STEPS_PER_TASK}", True, kind="status")

    def _run_tool(self, name: str, args: dict, task_id: str) -> str:
        try:
            if name == "load_skill":
                skill = args.get("skill", "")
                contents = self.skills.load([skill])
                return contents.get(skill) or f"未找到 skill: {skill}"
            result = self.adapter.execute(name, args, trace_id=task_id, timeout_sec=300.0)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return f"工具执行失败: {exc}"

    # 把最近对话历史序列化注入工具参数(_agent_context),供 executor 里的 VLM 工具继承上下文。
    # 只取最近 CONTEXT_TAIL 条消息,并限制总长度,避免每次工具调用都全量搬运历史。
    CONTEXT_TAIL = 30
    CONTEXT_MAX_CHARS = 8000

    def _with_context(self, args: dict) -> dict:
        recent = self.messages[-self.CONTEXT_TAIL:]
        try:
            text = json.dumps(recent, ensure_ascii=False)
            if len(text) > self.CONTEXT_MAX_CHARS:
                text = text[-self.CONTEXT_MAX_CHARS:]
            args = {**args, "_agent_context": text}
        except Exception:
            pass
        return args

    @staticmethod
    def _parse_args(tc) -> dict:
        try:
            return json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            return {}

    def _emit(self, task_id: str, status: str, message: str, done: bool, kind: str = "status") -> None:
        if self.on_state:
            self.on_state(task_id, status, message, done, kind)
