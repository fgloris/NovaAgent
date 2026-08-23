"""内置技能执行器基类。"""

from __future__ import annotations

from typing import Any


class ExecContext:
    """一次技能调用所需的运行时上下文。"""

    def __init__(self, node, task_id: str, goal: str, params: dict[str, Any]) -> None:
        self.node = node
        self.task_id = task_id
        self.goal = goal
        self.params = params


class ExecResult:
    def __init__(self, success: bool, info: str = "") -> None:
        self.success = success
        self.info = info


class BaseExecutor:
    """执行器在 skill_manager 节点的线程池里运行,可用节点做同步服务调用。"""

    def __init__(self, node) -> None:
        self.node = node

    def run(self, ctx: ExecContext) -> ExecResult:
        raise NotImplementedError
