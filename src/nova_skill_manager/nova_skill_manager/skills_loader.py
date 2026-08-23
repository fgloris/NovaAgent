"""技能注册表加载器:发现 skills/<name>/SKILL.md 并做可用性检查。"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import yaml


class SkillRecord:
    """一个已发现技能的名字、文件路径与来源。"""

    def __init__(self, name: str, path: Path, source: str) -> None:
        self.name = name
        self.path = path
        self.source = source

    @property
    def content(self) -> str:
        return self.path.read_text(encoding="utf-8")


def parse_frontmatter(content: str) -> dict:
    if not content.startswith("---"):
        return {}
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group(1)) or {}
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return {}


def nova_meta(frontmatter: dict) -> dict:
    """从 frontmatter 的 metadata 字段提取 nova 命名空间的元数据。"""
    raw = frontmatter.get("metadata")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        data = raw.get("nova", raw)
    else:
        try:
            data = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            return {}
        data = data.get("nova", data) if isinstance(data, dict) else {}
    return data if isinstance(data, dict) else {}


class SkillsLoader:
    """扫描 workspace 与 builtin 两个目录,workspace 同名覆盖 builtin。"""

    def __init__(self, builtin_dir: Path, workspace_dir: Path) -> None:
        self.builtin_dir = Path(builtin_dir)
        self.workspace_dir = Path(workspace_dir)

    def _scan(self, base: Path) -> list[SkillRecord]:
        records: list[SkillRecord] = []
        if not base.exists():
            return records
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "SKILL.md").exists():
                source = "workspace" if base == self.workspace_dir else "builtin"
                records.append(SkillRecord(child.name, child / "SKILL.md", source))
        return records

    def list(self) -> list[SkillRecord]:
        records: dict[str, SkillRecord] = {}
        for record in self._scan(self.workspace_dir):
            records[record.name] = record
        for record in self._scan(self.builtin_dir):
            records.setdefault(record.name, record)
        return list(records.values())

    def get(self, name: str) -> SkillRecord | None:
        for record in self.list():
            if record.name == name:
                return record
        return None

    def frontmatter(self, name: str) -> dict:
        record = self.get(name)
        if record is None:
            return {}
        return parse_frontmatter(record.content)

    def description(self, name: str) -> str:
        meta = self.frontmatter(name).get("description")
        if isinstance(meta, str):
            return meta.strip()
        if isinstance(meta, list):
            return " ".join(str(x) for x in meta).strip()
        return name

    def _requires(self, name: str) -> dict:
        return nova_meta(self.frontmatter(name)).get("requires") or {}

    def is_available(self, name: str) -> bool:
        meta = nova_meta(self.frontmatter(name))
        available = meta.get("available", True)
        if isinstance(available, str):
            if available.strip().lower() in {"false", "0", "no", "off"}:
                return False
        elif available is False:
            return False
        requires = self._requires(name)
        for binary in requires.get("bins", []):
            if not shutil.which(binary):
                return False
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True

    def missing_requirements(self, name: str) -> str:
        missing: list[str] = []
        requires = self._requires(name)
        for binary in requires.get("bins", []):
            if not shutil.which(binary):
                missing.append(f"bin:{binary}")
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"env:{env}")
        return ", ".join(missing)

    def info(self, record: SkillRecord) -> dict:
        return {
            "name": record.name,
            "description": self.description(record.name),
            "available": self.is_available(record.name),
            "requires": self.missing_requirements(record.name),
            "source": record.source,
            "location": str(record.path),
        }

    def load(self, name: str) -> str | None:
        record = self.get(name)
        return record.content if record is not None else None
