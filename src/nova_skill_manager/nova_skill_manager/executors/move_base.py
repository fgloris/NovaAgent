"""move_base: 让 PandaOmron 底盘朝 (vx, vy, wz) 持续移动若干秒。

robocasa 12 维动作布局(见 bridge 的 action_vector_to_dict):
[0:3] 末端位置, [3:6] 末端姿态, [6] 夹爪, [7:11] base_motion[vx,vy,wz,torso], [11] control_mode。
"""

from __future__ import annotations

import time

from std_msgs.msg import Float32MultiArray

from nova_skill_manager.executors.base import BaseExecutor, ExecContext, ExecResult


class MoveBaseExecutor(BaseExecutor):
    def __init__(self, node) -> None:
        super().__init__(node)
        self.action_pub = node.create_publisher(Float32MultiArray, "/nova/robocasa/action_cmd", 10)

    def run(self, ctx: ExecContext) -> ExecResult:
        vx = float(ctx.params.get("vx", 0.0))
        vy = float(ctx.params.get("vy", 0.0))
        wz = float(ctx.params.get("wz", 0.0))
        duration = float(ctx.params.get("duration", 1.0))
        base_mode = float(ctx.params.get("base_mode", 1.0))

        values = [0.0] * 12
        values[7:11] = [vx, vy, wz, 0.0]
        values[11] = 1.0 if base_mode >= 0.5 else 0.0

        deadline = time.monotonic() + max(0.0, duration)
        while time.monotonic() < deadline:
            msg = Float32MultiArray()
            msg.data = values
            self.action_pub.publish(msg)
            time.sleep(0.05)

        return ExecResult(
            True,
            f"base moved (vx={vx}, vy={vy}, wz={wz}) for {duration:.2f}s",
        )
