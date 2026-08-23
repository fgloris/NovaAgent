from nova_skill_manager.executors.base import BaseExecutor, ExecContext, ExecResult
from nova_skill_manager.executors.move_base import MoveBaseExecutor
from nova_skill_manager.executors.reset import ResetExecutor
from nova_skill_manager.executors.wait import WaitExecutor

__all__ = [
    "BaseExecutor",
    "ExecContext",
    "ExecResult",
    "MoveBaseExecutor",
    "ResetExecutor",
    "WaitExecutor",
]

# 内置执行器注册表:技能名 -> 执行器类
BUILTIN_EXECUTORS = {
    "reset": ResetExecutor,
    "wait": WaitExecutor,
    "move_base": MoveBaseExecutor,
}
