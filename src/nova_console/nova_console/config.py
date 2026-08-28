#!/usr/bin/env python3
# nova_console 配置:加载/校验 sessions.yaml(profiles + sessions)。
# 会话字段:id / name / venv? / workdir? / env? / pre[] / command / depends_on[] / wait_for? / wait_timeout_sec?
from __future__ import annotations

import os
from pathlib import Path

import yaml

REQUIRED_FIELDS = ("id", "command")


class ConfigError(Exception):
    pass


def find_config_path() -> Path | None:
    env = os.environ.get("NOVA_CONSOLE_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "config" / "sessions.yaml",  # src/nova_console/config
    ]
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory("nova_console")) / "config" / "sessions.yaml"
        )
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            return c
    return None


def load_profiles(config_path: str | None = None) -> dict[str, dict]:
    path = Path(config_path).expanduser() if config_path else find_config_path()
    if path is None:
        raise ConfigError("找不到 sessions.yaml(可设置环境变量 NOVA_CONSOLE_CONFIG 或先构建 nova_console)")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    profiles = data.get("profiles") or {}
    result: dict[str, dict] = {}
    for name, pf in profiles.items():
        sessions = pf.get("sessions") or []
        ids = [s.get("id") for s in sessions]
        for s in sessions:
            for field in REQUIRED_FIELDS:
                if not s.get(field):
                    raise ConfigError(f"profile {name} 的会话缺少字段 '{field}'")
            for dep in s.get("depends_on") or []:
                if dep not in ids:
                    raise ConfigError(f"profile {name} 会话 {s['id']} 的依赖不存在: {dep}")
        result[name] = {"name": name, "sessions": sessions}
    return result
