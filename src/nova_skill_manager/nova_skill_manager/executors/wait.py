"""wait: 等待若干秒,期间可发零动作让仿真继续推进。"""

from __future__ import annotations

import time

from std_msgs.msg import Float32MultiArray

from nova_skill_manager.executors.base import BaseExecutor, ExecContext, ExecResult


class WaitExecutor(BaseExecutor):
    def __init__(self, node) -> None:
        super().__init__(node)
        self.action_pub = node.create_publisher(Float32MultiArray, "/nova/robocasa/action_cmd", 10)

    def run(self, ctx: ExecContext) -> ExecResult:
        duration = float(ctx.params.get("duration", 1.0))
        publish_zero = bool(ctx.params.get("zero_action", True))
        deadline = time.monotonic() + max(0.0, duration)
        while time.monotonic() < deadline:
            if publish_zero:
                msg = Float32MultiArray()
                msg.data = [0.0] * 12
                self.action_pub.publish(msg)
            time.sleep(0.05)
        return ExecResult(True, f"waited {duration:.2f}s")
