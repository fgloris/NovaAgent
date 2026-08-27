# skill 存储:管理任务型领域经验(skill 是纯文本,不涉及执行)。
# 目录结构:skills/<name>/SKILL.yaml(元数据) + skills/<name>/SKILL.md(经验正文)。
# future annotations:避免类作用域里 list 方法遮蔽内置 list,导致 load 注解报错
from __future__ import annotations
from pathlib import Path
import yaml


class SkillStore:
    def __init__(self, skills_dir: str | Path) -> None:
        self.root = Path(skills_dir)

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        result = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            meta_file = d / "SKILL.yaml"
            if not meta_file.exists():
                continue
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            result.append(
                {
                    "name": meta.get("name", d.name),
                    "description": meta.get("description", ""),
                    "tags": meta.get("tags") or [],
                    "requires_tools": meta.get("requires_tools") or [],
                    "dir": str(d),
                }
            )
        return result

    # 紧凑索引,注入 LLM 阶段1,只给名称/描述/标签/推荐工具
    def index_text(self) -> str:
        lines = []
        for s in self.list():
            lines.append(
                f"- {s['name']}: {s['description']} "
                f"(tags: {', '.join(s['tags']) or '无'}; 推荐工具: {', '.join(s['requires_tools']) or '无'})"
            )
        return "\n".join(lines)

    # 按名称加载正文,注入 LLM 阶段2
    def load(self, names: list[str]) -> dict[str, str]:
        contents = {}
        for s in self.list():
            if s["name"] in names:
                md = Path(s["dir"]) / "SKILL.md"
                contents[s["name"]] = md.read_text(encoding="utf-8") if md.exists() else ""
        return contents
