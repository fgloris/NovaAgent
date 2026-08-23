"""AgentOS 上下文构建:技能 XML 摘要 + system prompt + 观测渲染。"""

from __future__ import annotations

import json
from typing import Any

IDENTITY = """你是 NovoAgent 的 AgentOS,一个 LLM 驱动的机器人技能调度大脑。
你的目标:把自然语言指令拆解成可执行的技能序列,逐个下发,直到任务完成。
决策必须只依赖下方提供的可用技能,不得虚构不存在的技能。"""


def escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_skills_summary(skills: list[dict]) -> str:
    """把 ListSkills 返回的技能列表渲染成 XML 摘要,供 LLM 选择。"""
    if not skills:
        return "<skills/>"
    lines = ["<skills>"]
    for skill in skills:
        lines.append(f"  <skill available=\"{'true' if skill['available'] else 'false'}\">")
        lines.append(f"    <name>{escape_xml(skill['name'])}</name>")
        lines.append(f"    <description>{escape_xml(skill['description'])}</description>")
        if not skill["available"] and skill["requires"]:
            lines.append(f"    <requires>{escape_xml(skill['requires'])}</requires>")
        lines.append("  </skill>")
    lines.append("</skills>")
    return "\n".join(lines)


SYSTEM_TEMPLATE = """{identity}

# 可用技能

下面是当前注册的技能摘要。执行前必须用技能名+参数(JSON)下发命令。
如果某技能提供了详细 SKILL.md(通过 get/load 服务读取),先阅读它再决定参数。

{skills_summary}

# 输出格式

每次决策只输出一个 JSON 对象(不要输出其他文字):

- 继续执行: {{"done": false, "skill_id": "<技能名>", "params": {{...}}, "goal": "<这一步做什么>"}}
- 任务完成: {{"done": true, "summary": "<任务结果总结>"}}

约束:
- skill_id 必须来自可用技能列表。
- params 里的参数名必须与技能 SKILL.md 一致。
- 一次只选一个技能,不要在 goal 里塞多步任务。"""


def build_user_message(
    instruction: str,
    observation: dict[str, Any],
    history: list[dict],
    skill_details: dict[str, str] | None = None,
) -> str:
    lines = [f"# 当前任务\n{instruction}\n"]
    if observation:
        lines.append("# 当前观测\n```json\n" + json.dumps(observation, ensure_ascii=False) + "\n```")
    if history:
        lines.append("# 已执行步骤\n")
        for entry in history:
            lines.append(
                f"- skill={entry['skill_id']} goal={entry['goal']} -> "
                f"{entry['status']} ({entry['info']})"
            )
    if skill_details:
        lines.append("# 技能详情(执行前已读取的 SKILL.md)\n")
        for name, content in skill_details.items():
            lines.append(f"## {name}\n{content}")
    return "\n".join(lines)
