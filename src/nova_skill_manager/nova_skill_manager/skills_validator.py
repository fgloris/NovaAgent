"""SKILL.md 结构校验,规则取自 HoloAgent 的 validate_skills.py。"""

from __future__ import annotations

from pathlib import Path

from nova_skill_manager.skills_loader import parse_frontmatter


def validate_skill(record) -> list[str]:
    errors: list[str] = []
    if not record.path.exists():
        errors.append(f"{record.name}: missing SKILL.md")
        return errors

    content = record.content
    frontmatter = parse_frontmatter(content)

    if not frontmatter.get("name"):
        errors.append(f"{record.name}: SKILL.md missing YAML frontmatter name")
    if not frontmatter.get("description"):
        errors.append(f"{record.name}: SKILL.md missing description")
    if "**Use this skill when:**" not in content:
        errors.append(f"{record.name}: SKILL.md missing trigger guidance")
    if "## Workflow" not in content:
        errors.append(f"{record.name}: SKILL.md missing workflow section")
    if "## Safety Rules" not in content:
        errors.append(f"{record.name}: SKILL.md missing safety rules section")
    return errors


def validate_directory(skills_dir: Path, records) -> str:
    if not skills_dir.exists():
        return f"[ERROR] skills directory not found: {skills_dir}"
    if not records:
        return "[ERROR] no skill directories found"

    errors: list[str] = []
    warnings: list[str] = []
    for record in records:
        errors.extend(validate_skill(record))
        readme = record.path.parent / "README.md"
        if not readme.exists():
            warnings.append(f"{record.name}: missing README.md")

    lines: list[str] = []
    if errors:
        lines.append("Skill validation failed:")
        lines.extend(f"- {error}" for error in errors)
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in warnings)
        return "\n".join(lines)

    lines.append("Skill validation passed.")
    lines.extend(f"- {record.name}" for record in records)
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)
