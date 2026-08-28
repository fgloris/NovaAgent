# 规划器:将用户指令转为 DAG 任务图。
# agent loop:维护对话上下文记忆,按需 load_skill 加载领域经验,
# 直到模型发出 submit_dag 的 function calling 才结束;结束后上下文随本次调用丢弃。
import json

from nova_common.llm_client import LLMClient

from nova_agentos.skill_store import SkillStore

# agent loop 最大轮数,防止模型不提交时死循环
MAX_ROUNDS = 6

# LLM 通过 function calling 提交 DAG
PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_dag",
        "description": "把用户指令转化为工具调用 DAG 任务图,并给出需要用到的 skill 名称",
        "parameters": {
            "type": "object",
            "properties": {
                "selected_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "选中的 skill 名称(仅来自 skill 索引)",
                },
                "description": {"type": "string", "description": "整体任务概括"},
                "nodes": {
                    "type": "array",
                    "description": "任务节点,每个节点是一次工具调用",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "唯一节点 id"},
                            "type": {"type": "string", "enum": ["tool"], "description": "当前固定 tool"},
                            "tool_name": {"type": "string", "description": "可用工具名"},
                            "params_json": {
                                "type": "string",
                                "description": '工具参数(JSON 字符串),参数值可为 "$ref:<节点id>" 引用前序节点结果',
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "依赖的前序节点 id,无依赖则为空数组",
                            },
                            "goal": {"type": "string", "description": "该节点目标描述"},
                        },
                        "required": ["id", "tool_name", "depends_on", "goal"],
                    },
                },
            },
            "required": ["selected_skills", "description", "nodes"],
        },
    },
}

# 按需加载 skill 全文,结果作为 tool 消息回填上下文
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

SYSTEM_BASE = (
    "你是具身机器人 NovaAgent 的规划器。你的任务:把用户指令分解为 DAG 任务图,"
    "图中每个节点是一次 executor 工具调用(type 固定为 tool)。"
    "用 depends_on 表达节点间依赖,无依赖的节点可以并行。"
    "参数放在 params_json(JSON 字符串);参数可以引用前序节点结果,把对应值写成 \"$ref:<节点id>\"。"
    "只使用下面提供的工具。规划时可先调用 load_skill 加载相关 skill 的领域经验作为参考,"
    "最后必须调用 submit_dag 提交结果;不要用文本代替 submit_dag。"
)


class Planner:
    def __init__(self, llm: LLMClient, skill_store: SkillStore) -> None:
        self.llm = llm
        self.skills = skill_store

    def plan(self, instruction: str, tools_text: str) -> dict:
        skill_index = self.skills.index_text()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_BASE + self._context(skill_index, tools_text)},
            {"role": "user", "content": f"指令:{instruction}"},
        ]
        for round_no in range(1, MAX_ROUNDS + 1):
            result = self.llm.chat(messages, tools=[PLAN_TOOL, LOAD_SKILL_TOOL])
            dag = self._step(messages, result)
            if dag is not None:
                print(f"[planner] 输出DAG: {json.dumps(dag, ensure_ascii=False)}", flush=True)
                return dag
            print(
                f"[planner]: content={result.content!r} "
                f"tool_calls={[tc['function']['name'] for tc in result.tool_calls]}",
                flush=True,
            )
        # 兜底:从最后一条文本里尝试解析 DAG
        content = messages[-1].get("content") or ""
        start, end = content.find("{"), content.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"规划器 {MAX_ROUNDS} 轮内未提交 DAG,最后输出: {content[:500]}")

    # 处理一轮 LLM 返回:更新上下文;返回 DAG 表示本轮以 submit_dag 结束,否则继续
    def _step(self, messages: list[dict], result) -> dict | None:
        if not result.tool_calls:
            messages.append({"role": "assistant", "content": result.content or ""})
            return None
        messages.append(
            {"role": "assistant", "content": result.content or "", "tool_calls": result.tool_calls}
        )
        for tc in result.tool_calls:
            if tc["function"]["name"] == "submit_dag":
                return json.loads(tc["function"]["arguments"])
        # 其它工具调用(load_skill):执行并回填 tool 消息
        for tc in result.tool_calls:
            content = self._run_tool(tc)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
        return None

    def _run_tool(self, tc) -> str:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            return "工具参数非法,请重试"
        if name == "load_skill":
            skill = args.get("skill", "")
            contents = self.skills.load([skill])
            return contents.get(skill) or f"未找到 skill: {skill}"
        return f"未知工具: {name}"

    @staticmethod
    def _context(skill_index: str, tools_text: str) -> str:
        return (
            f"\n\n# Skill 索引\n{skill_index or '(无)'}\n\n# 可用工具\n{tools_text or '(无)'}"
        )
