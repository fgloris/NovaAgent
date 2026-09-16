"""串行的 AgentOS 任务运行时。"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Callable

from nova_common import image_codec
from nova_common.llm_client import LLMClient
from nova_common.tool_result import split_images

from nova_agentos.mcp_adapter import McpAdapter, to_llm_tools
from nova_agentos.memory import Compactor, ContextBuilder, ImageMemory, SessionManager, TaskMemory
from nova_agentos.skill_store import SkillStore

MAX_STEPS_PER_TASK = 20
MAX_TOOL_FAILS = 3

SYSTEM_BASE = (
    "你是具身机器人 NovaAgent 的 VLM agent。理解用户指令和当前多摄像机画面，"
    "通过调用工具逐步执行任务。通过组合各类工具完成你的任务，"
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

LIST_IMAGES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_accessible_images",
        "description": "列出可访问的图像:所有工具返回图 + 每链路最近 N 张历史图,含 url/时间/相机/来源描述",
        "parameters": {
            "type": "object",
            "properties": {
                "history_depth": {
                    "type": "integer",
                    "description": "每条链路返回的历史图数量,默认取配置",
                }
            },
        },
    },
}

FETCH_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_history_image",
        "description": "按时间取某链路最接近的历史图像并注入上下文,返回其描述与 url",
        "parameters": {
            "type": "object",
            "properties": {
                "time": {"type": "number", "description": "目标时间(epoch 秒)"},
                "topic": {"type": "string", "description": "相机/链路名"},
            },
            "required": ["time", "topic"],
        },
    },
}

LOCAL_TOOLS = [LOAD_SKILL_TOOL, LIST_IMAGES_TOOL, FETCH_HISTORY_TOOL]

class TaskRunner:
    """执行单个任务的 ReAct 循环:构造上下文 -> LLM 决策 -> 调工具,直到 finish 或超限。"""

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
        robot_state_provider: Callable[[], dict | None] | None = None,
        image_memory: ImageMemory | None = None,
        frame_provider: Callable[[], dict] | None = None,
        image_history_depth: int = 4,
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
        self.robot_state_provider = robot_state_provider
        self.images = image_memory
        self.frame_provider = frame_provider
        self.image_history_depth = max(1, int(image_history_depth))
        self.runtime_messages: list[dict] = []

    def run(self) -> None:
        """任务主循环:每轮把最新上下文与观测喂给 LLM,执行其请求的工具调用。"""
        self._event("status", "working", f"收到指令: {self.task.instruction}")
        try:
            descriptors = self.adapter.fetch_tools()
            tools = LOCAL_TOOLS + [FINISH_TOOL] + to_llm_tools(descriptors)
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
                    self._observations(),
                    tools,
                    skill_index=self.skills.index_text(),
                    robot_context=self.robot_context_provider() if self.robot_context_provider else None,
                )
                messages.extend(self.runtime_messages)
                # 动态图像段(current -> processed -> history)与时间戳放在最末,保留稳定前缀
                image_parts = self._image_context()
                if image_parts:
                    messages.append({"role": "user", "content": image_parts})
                messages.append(self._timestamp_message())
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
                
                # 依次处理本轮的所有 tool calls
                round_images: list[tuple[str, str, str]] = []
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
                    content, images = self._run_tool(name, args)
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
                    for image_id, url in images.items():
                        round_images.append((name, image_id, url))
                    if failed:
                        fails += 1
                        if fails >= MAX_TOOL_FAILS:
                            self._finish("failed", f"工具连续失败 {fails} 次: {content}")
                            return
                    else:
                        fails = 0
                # 本轮工具返回的图像合并成一条 user 多模态消息,下一轮 VLM 可见
                if round_images:
                    parts: list[dict] = []
                    for tool_name, image_id, url in round_images:
                        parts.append({"type": "text", "text": f"工具 {tool_name} 返回图像 {image_id}"})
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                    self.runtime_messages.append({"role": "user", "content": parts})
                    self.task.add_event(
                        "tool_images",
                        images=[{"tool": t, "image_id": i} for t, i, _ in round_images],
                    )
                    self.manager.save_task(self.task)
            self._finish("failed", f"超过单任务最大步数 {MAX_STEPS_PER_TASK}")
        except Exception as exc:
            self._finish("failed", f"agent loop 异常: {exc}")

    def _run_tool(self, name: str, args: dict) -> tuple[str, dict[str, str]]:
        """执行一次工具调用,返回 (给模型的文本, 待注入的图像 {url: data_url})。"""
        try:
            if name == "load_skill":
                skill = args.get("skill", "")
                contents = self.skills.load([skill])
                return contents.get(skill) or f"未找到 skill: {skill}", {}
            if name == "list_accessible_images":
                return self._list_accessible_images(args), {}
            if name == "fetch_history_image":
                return self._fetch_history_image(args)

            call_args = self._prepare_executor_args(args)

            def feedback(status: str, message: str) -> None:
                self.task.add_event(
                    "tool_feedback", tool_name=name, status=status, message=message
                )
                self.manager.save_task(self.task)
                self._event("tool_feedback", "working", f"{name}: {message}")

            result = self.adapter.execute(
                name,
                call_args,
                trace_id=self.task.task_id,
                timeout_sec=300.0,
                feedback_callback=feedback,
            )
            stripped, images = split_images(result)
            if self.images is not None and images:
                stripped["images"] = self._store_processed(name, call_args, stripped, images)
            return json.dumps(stripped, ensure_ascii=False), {}
        except Exception as exc:
            return f"工具执行失败: {exc}", {}

    # ---------- 图像记忆 ----------

    def _prepare_executor_args(self, args: dict) -> dict:
        """给 perception 调用注入隐藏 image_root,并把相机名/别名解析成 file:// 引用。"""
        if self.images is None or "image" not in args:
            return args
        call_args = dict(args)
        call_args["image"] = self._resolve_image_arg(args.get("image"))
        call_args["image_root"] = str(self.images.root)
        return call_args

    def _resolve_image_arg(self, value) -> str:
        """把 image 参数解析为可用的 file:// 引用:优先按已有记录,其次按相机名映射最新 current 图。"""
        text = str(value)
        record = self.images.find(text)
        if record is not None:
            return record.url
        for current in self.images.current_records():
            if current.camera == text:
                return current.url
        return text

    def _store_processed(self, tool: str, call_args: dict, stripped: dict, images: dict) -> list[dict]:
        """把工具返回的 data URL 图落盘为 processed,返回给模型看的描述列表。"""
        camera = str(stripped.get("camera", "") or "")
        base_url = str(call_args.get("image", "") or "")
        params = {k: v for k, v in call_args.items() if k not in {"image", "image_root"}}
        now = time.time()
        described: list[dict] = []
        for url in images.values():
            try:
                data = image_codec.data_url_to_bytes(url)
            except Exception:
                continue
            record = self.images.save_processed(data, camera, tool, params, base_url, now)
            described.append(record.describe())
        return described

    def _list_accessible_images(self, args: dict) -> str:
        if self.images is None:
            return "图像记忆未启用"
        try:
            depth = int(args.get("history_depth", self.image_history_depth))
        except (TypeError, ValueError):
            depth = self.image_history_depth
        depth = max(1, depth)
        payload = {
            "processed": [record.describe() for record in self.images.processed_records()],
            "history": {
                camera: [record.describe() for record in self.images.history_records(camera, depth)]
                for camera in self.images.all_history_cameras()
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _fetch_history_image(self, args: dict) -> tuple[str, dict[str, str]]:
        if self.images is None:
            return "图像记忆未启用", {}
        camera = str(args.get("topic", ""))
        try:
            ts = float(args.get("time"))
        except (TypeError, ValueError):
            return "time 必须是数字", {}
        record = self.images.nearest_history(camera, ts)
        if record is None:
            return f"未找到链路 {camera} 的历史图像", {}
        return json.dumps(record.describe(), ensure_ascii=False), {record.url: self.images.data_url(record)}

    def _image_context(self) -> list[dict]:
        """构建动态图像段:current -> processed -> history(每链路最近 N 张)。"""
        if self.images is None:
            return []
        if self.frame_provider is not None:
            try:
                frames = self.frame_provider()
            except Exception:
                frames = {}
            if frames:
                self.images.refresh_current(frames)
        parts: list[dict] = []
        for record in self.images.current_records():
            self._append_image(parts, "current", record)
        for record in self.images.processed_records():
            self._append_image(parts, "processed", record)
        for camera in self.images.all_history_cameras():
            for record in self.images.history_records(camera, self.image_history_depth):
                self._append_image(parts, "history", record)
        return parts

    def _append_image(self, parts: list[dict], label: str, record) -> None:
        try:
            url = self.images.data_url(record)
        except Exception:
            return
        parts.append(
            {"type": "text", "text": f"# {label} image\n{json.dumps(record.describe(), ensure_ascii=False)}"}
        )
        parts.append({"type": "image_url", "image_url": {"url": url}})

    @staticmethod
    def _timestamp_message() -> dict:
        now = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int((now % 1) * 1000):03d}Z"
        return {"role": "user", "content": f"# 当前时间\n{iso} (epoch {now:.3f})"}

    def _observations(self) -> list[dict]:
        """收集各观测 provider(视觉、机器人状态)的消息;单个 provider 失败不影响其它。"""
        out: list[dict] = []
        for label, provider in (
            ("环境视觉观测", self.observation_provider),
            ("机器人状态", self.robot_state_provider),
        ):
            if provider is None:
                continue
            try:
                item = provider()
            except Exception as exc:
                item = {"role": "user", "content": f"# {label}\n读取失败: {exc}"}
            if item:
                out.append(item)
        return out

    @staticmethod
    def _parse_args(tool_call: dict) -> dict:
        """解析 tool_call 的 JSON 参数;解析失败返回空 dict。"""
        try:
            return json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, TypeError, KeyError):
            return {}

    def _event(self, kind: str, status: str, message: str, done: bool = False) -> None:
        """通过 on_state 回调上报一次状态事件(用于 debug 与向 /agent_msg 发话题)。"""
        if self.on_state:
            self.on_state(self.task.task_id, self.task.session_id, status, message, done, kind)

    def _finish(self, outcome: str, summary: str) -> None:
        """结束本次任务:写入结果与总结,把任务上下文回存到 session,并上报完成事件。"""
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
    """单后台线程消费全局任务 FIFO 队列,逐个交给 TaskRunner 串行执行。"""

    def __init__(
        self,
        llm: LLMClient,
        skills: SkillStore,
        adapter: McpAdapter,
        session_manager: SessionManager | None = None,
        on_state: Callable[[str, str, str, bool, str, str], None] | None = None,
        observation_provider: Callable[[], dict | None] | None = None,
        robot_context_provider: Callable[[], dict | None] | None = None,
        robot_state_provider: Callable[[], dict | None] | None = None,
        context_budget_tokens: int = 12000,
        context_compaction_enabled: bool = True,
        max_recent_tasks: int = 8,
        image_memory_factory: Callable[[str], ImageMemory] | None = None,
        frame_provider: Callable[[], dict] | None = None,
        image_history_depth: int = 4,
    ) -> None:
        self.llm = llm
        self.skills = skills
        self.adapter = adapter
        self.session_manager = session_manager or SessionManager()
        self.on_state = on_state
        self.observation_provider = observation_provider
        self.robot_context_provider = robot_context_provider
        self.robot_state_provider = robot_state_provider
        self.image_memory_factory = image_memory_factory
        self.frame_provider = frame_provider
        self.image_history_depth = max(1, int(image_history_depth))
        self.context_builder = ContextBuilder(
            Compactor(context_budget_tokens, context_compaction_enabled, max_recent_tasks)
        )
        self.queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._current_memory: ImageMemory | None = None
        self._active_session_id = ""

    def current_memory(self) -> ImageMemory | None:
        """返回当前活动 session 的图像记忆(供采样器使用)。"""
        return self._current_memory

    def activate_session(self, session_id: str) -> None:
        """session 激活/恢复时预热图像记忆,使采样在任务开始前即可运行。"""
        if self.image_memory_factory and session_id:
            self._current_memory = self.image_memory_factory(session_id)
            self._active_session_id = session_id

    def deactivate_session(self, session_id: str) -> None:
        """session 结束时停止其图像采样(文件保留)。"""
        if session_id and session_id == self._active_session_id:
            self._current_memory = None
            self._active_session_id = ""

    def start(self) -> None:
        """启动后台任务消费线程。"""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止消费线程并等待其退出(最多 3 秒)。"""
        self._running = False
        self.queue.put(None)
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)

    def submit(self, task_id: str, session_id: str, instruction: str) -> None:
        """把任务加入队列,交由后台线程执行。"""
        self.queue.put((task_id, session_id, instruction))

    def _run(self) -> None:
        """后台线程主体:不断从队列取任务并运行,取到 None 哨兵时退出。"""
        while self._running:
            item = self.queue.get()
            if item is None:
                break
            task_id, session_id, _instruction = item
            try:
                task = self.session_manager.load_task(session_id, task_id)
                memory = (
                    self.image_memory_factory(session_id) if self.image_memory_factory else None
                )
                self._current_memory = memory
                self._active_session_id = session_id
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
                    robot_state_provider=self.robot_state_provider,
                    image_memory=memory,
                    frame_provider=self.frame_provider,
                    image_history_depth=self.image_history_depth,
                ).run()
            except Exception as exc:
                if self.on_state:
                    self.on_state(
                        task_id, session_id, "failed", f"任务启动失败: {exc}", True, "status"
                    )
