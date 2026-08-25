#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import numpy as np
import rclpy

from nova_common.env_bridge import EnvBridgeBase
from nova_robocasa_bridge.robocasa_sim_server import action_vector_to_dict


class RoboCasaBridgeNode(EnvBridgeBase):
    def __init__(self) -> None:
        super().__init__("robocasa_bridge", action_dim_default=12)

    def _declare_custom_params(self) -> None:
        self.declare_parameter("env_id", "robocasa/PickPlaceCounterToCabinet")
        self.declare_parameter("seed", 0)
        self.env_id = str(self.get_parameter("env_id").value)
        self.seed = int(self.get_parameter("seed").value)

    def _build_reset_request(self) -> dict[str, Any]:
        return {
            "type": "reset",
            "env_id": self.env_id,
            "seed": self.seed,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
        }

    def action_vector_to_native(self, values: np.ndarray):
        return action_vector_to_dict(values)

    def _extra_info(self) -> dict[str, Any]:
        info = dict(self.sim_info)
        info.setdefault("sim", "robocasa")
        return info


def main(args=None) -> int:
    rclpy.init(args=args)
    node = RoboCasaBridgeNode()
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
