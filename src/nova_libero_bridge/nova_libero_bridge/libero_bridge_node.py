#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
import numpy as np
import rclpy

from nova_common.env_bridge import EnvBridgeBase


class LiberoBridgeNode(EnvBridgeBase):
    def __init__(self) -> None:
        super().__init__("libero_bridge", action_dim_default=7)

    def _declare_custom_params(self) -> None:
        self.declare_parameter("benchmark", "libero_spatial")
        self.declare_parameter("task_id", 0)
        self.declare_parameter("seed", 0)
        self.benchmark = str(self.get_parameter("benchmark").value)
        self.task_id = int(self.get_parameter("task_id").value)
        self.seed = int(self.get_parameter("seed").value)

    def _build_reset_request(self) -> dict[str, Any]:
        return {
            "type": "reset",
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "seed": self.seed,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
        }

    def action_vector_to_native(self, values: np.ndarray):
        return values.tolist()

    def _extra_info(self) -> dict[str, Any]:
        info = dict(self.sim_info)
        info.setdefault("sim", "libero")
        return info


def main(args=None) -> int:
    rclpy.init(args=args)
    node = LiberoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
