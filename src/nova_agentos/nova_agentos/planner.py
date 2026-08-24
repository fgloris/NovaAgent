# 规划器:将用户指令转为 DAG 任务图。
# 两阶段:
#   阶段1 注入 skill 索引 + 可用工具,LLM 输出 {selected_skills, dag 草稿}
#   阶段2 注入选中 skill 全文,LLM 依据领域经验补全节点参数。
import json

from nova_common.llm_client import LLMClient

from nova_agentos.skill_store import SkillStore

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

SYSTEM_BASE = (
    "你是具身机器人 NovaAgent 的规划器。你的任务:把用户指令分解为 DAG 任务图,"
    "图中每个节点是一次 executor 工具调用(type 固定为 tool)。"
    "用 depends_on 表达节点间依赖,无依赖的节点可以并行。"
    "参数放在 params_json(JSON 字符串);参数可以引用前序节点结果,把对应值写成 \"$ref:<节点id>\"。"
    "只使用下面提供的工具。最后调用 submit_dag 提交结果。"
)


class Planner:
    def __init__(self, llm: LLMClient, skill_store: SkillStore) -> None:
        self.llm = llm
        self.skills = skill_store

    def plan(self, instruction: str, tools_text: str) -> dict:
        skill_index = self.skills.index_text()

        # 阶段1:选 skill + 产出 DAG 草稿
        sys1 = SYSTEM_BASE + self._context(skill_index, tools_text)
        m1 = self.llm.chat(
            [{"role": "system", "content": sys1}, {"role": "user", "content": f"指令:{instruction}"}],
            tools=[PLAN_TOOL],
        )
        draft = self._extract_dag(m1)

        # 阶段2:注入 skill 全文,补全参数
        contents = self.skills.load(draft.get("selected_skills", []))
        skill_text = "\n\n".join(f"# Skill: {k}\n{v}" for k, v in contents.items()) or "(未选择 skill)"
        sys2 = (
            SYSTEM_BASE
            + self._context(skill_index, tools_text)
            + f"\n\n# 已注入的 Skill 领域经验(严格参考它细化每个节点的参数)\n{skill_text}"
        )
        m2 = self.llm.chat(
            [
                {"role": "system", "content": sys2},
                {
                    "role": "user",
                    "content": f"指令:{instruction}\n\n草稿:{json.dumps(draft, ensure_ascii=False)}",
                },
            ],
            tools=[PLAN_TOOL],
        )
        final = self._extract_dag(m2)
        return final

    @staticmethod
    def _context(skill_index: str, tools_text: str) -> str:
        return (
            f"\n\n# Skill 索引\n{skill_index or '(无)'}\n\n# 可用工具\n{tools_text or '(无)'}"
        )

    @staticmethod
    def _extract_dag(result) -> dict:
        if result.tool_calls:
            return json.loads(result.tool_calls[0]["function"]["arguments"])
        # 容错:从纯文本里提取 JSON 块
        content = result.content
        start = content.find("{")
        end = content.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"规划器未能解析 DAG,原始输出: {content[:500]}")
