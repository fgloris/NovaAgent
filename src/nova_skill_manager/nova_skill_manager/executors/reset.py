"""reset: 重置 robocasa 仿真环境。"""

from __future__ import annotations

import rclpy
from std_srvs.srv import Trigger

from nova_skill_manager.executors.base import BaseExecutor, ExecContext, ExecResult


class ResetExecutor(BaseExecutor):
    def __init__(self, node) -> None:
        super().__init__(node)
        self.client = node.create_client(Trigger, "/nova/robocasa/reset")

    def run(self, ctx: ExecContext) -> ExecResult:
        if not self.client.wait_for_service(timeout_sec=5.0):
            return ExecResult(False, "reset service not available")
        future = self.client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
        if future.result() is None:
            return ExecResult(False, "reset service call failed")
        response = future.result()
        return ExecResult(response.success, response.message)
