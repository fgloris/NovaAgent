"""skill 存储:管理任务型领域经验(skill 是纯文本,不涉及执行)。

目录结构:skills/<name>/SKILL.yaml(元数据) + skills/<name>/SKILL.md(经验正文)。
future annotations:避免类作用域里 list 方法遮蔽内置 list,导致 load 注解报错。
"""
from __future__ import annotations
from pathlib import Path
import yaml


class SkillStore:
    """扫描 skills 目录,提供 skill 元数据列表、索引文本与正文加载。"""

    def __init__(self, skills_dir: str | Path) -> None:
        self.root = Path(skills_dir)

    def list(self) -> list[dict]:
        """扫描目录,返回每个含 SKILL.yaml 的 skill 的元数据列表。"""
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

    def index_text(self) -> str:
        """生成紧凑的 skill 索引(名称/描述/标签/推荐工具),注入 LLM 阶段1。"""
        lines = []
        for s in self.list():
            lines.append(
                f"- {s['name']}: {s['description']} "
                f"(tags: {', '.join(s['tags']) or '无'}; 推荐工具: {', '.join(s['requires_tools']) or '无'})"
            )
        return "\n".join(lines)

    def load(self, names: list[str]) -> dict[str, str]:
        """按名称加载 skill 正文(SKILL.md),注入 LLM 阶段2。"""
        contents = {}
        for s in self.list():
            if s["name"] in names:
                md = Path(s["dir"]) / "SKILL.md"
                contents[s["name"]] = md.read_text(encoding="utf-8") if md.exists() else ""
        return contents
