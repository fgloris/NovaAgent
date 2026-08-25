#!/usr/bin/env python3
from __future__ import annotations

import random

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class RandomActionClient(Node):
    def __init__(self) -> None:
        super().__init__("nova_random_action_client")
        self.pub = self.create_publisher(Float32MultiArray, "/nova/env/action_cmd", 10)
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("scale", 0.05)
        self.declare_parameter("action_dim", 12)
        rate_hz = float(self.get_parameter("rate_hz").value)
        self.scale = float(self.get_parameter("scale").value)
        self.dim = int(self.get_parameter("action_dim").value)
        self.timer = self.create_timer(1.0 / max(rate_hz, 0.1), self.tick)

    def tick(self) -> None:
        values = [random.uniform(-self.scale, self.scale) for _ in range(self.dim)]
        if self.dim > 6:
            values[6] = random.choice([0.0, 1.0])
        if self.dim > 11:
            values[11] = 0.0
        msg = Float32MultiArray()
        msg.data = values
        self.pub.publish(msg)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = RandomActionClient()
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
