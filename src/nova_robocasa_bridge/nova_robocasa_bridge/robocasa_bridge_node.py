#!/usr/bin/env python3
from __future__ import annotations
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from nova_common.env_bridge import EnvBridgeBase
from nova_robot_description import load_robot_description, to_json


class RoboCasaBridgeNode(EnvBridgeBase):
    def __init__(self) -> None:
        super().__init__("robocasa_bridge", action_dim_default=12)

    def _declare_custom_params(self) -> None:
        self.declare_parameter("env_id", "robocasa/PickPlaceCounterToCabinet")
        self.declare_parameter("seed", 0)
        self.env_id = str(self.get_parameter("env_id").value)
        self.seed = int(self.get_parameter("seed").value)
        self.declare_parameter("robot_id", "robot0")
        self.declare_parameter("publish_robot_state", True)
        self.robot_id = str(self.get_parameter("robot_id").value)
        self.publish_robot_state = bool(self.get_parameter("publish_robot_state").value)
        self.robot_description = load_robot_description("panda_omron")
        self.eef_pub = self.create_publisher(PoseStamped, f"/{self.robot_id}/eef_pose", 10)
        self.joint_pub = self.create_publisher(JointState, f"/{self.robot_id}/joint_states", 10)
        self.gripper_pub = self.create_publisher(Float32MultiArray, f"/{self.robot_id}/gripper_state", 10)

    def _build_reset_request(self) -> dict[str, Any]:
        return {
            "type": "reset",
            "env_id": self.env_id,
            "seed": self.seed,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
        }

    def action_vector_to_native(self, values: np.ndarray):
        return values

    def _extra_info(self) -> dict[str, Any]:
        info = dict(self.sim_info)
        info.setdefault("sim", "robocasa")
        info["robot"] = to_json(self.robot_description)
        return info

    def _publish_observation(self) -> None:
        super()._publish_observation()
        if not self.publish_robot_state or self.obs is None:
            return
        p = self.obs.get(f"state.{self.robot_id}_base_to_eef_pos", self.obs.get(f"{self.robot_id}_base_to_eef_pos"))
        q = self.obs.get(f"state.{self.robot_id}_base_to_eef_quat", self.obs.get(f"{self.robot_id}_base_to_eef_quat"))
        if p is not None and q is not None:
            msg = PoseStamped(); msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = f"{self.robot_id}_base"
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [float(x) for x in p[:3]]
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = [float(x) for x in q[:4]]
            self.eef_pub.publish(msg)
        g = self.obs.get(f"state.{self.robot_id}_gripper_qpos", self.obs.get(f"{self.robot_id}_gripper_qpos"))
        if g is not None:
            gm = Float32MultiArray(); gm.data = [float(x) for x in np.asarray(g).reshape(-1)]; self.gripper_pub.publish(gm)
        j = self.obs.get(f"state.{self.robot_id}_joint_pos", self.obs.get(f"{self.robot_id}_joint_pos"))
        if j is not None:
            jm = JointState(); jm.header.stamp = self.get_clock().now().to_msg(); jm.name = [f"{self.robot_id}_joint_{i}" for i in range(len(np.asarray(j).reshape(-1)))]; jm.position = [float(x) for x in np.asarray(j).reshape(-1)]; self.joint_pub.publish(jm)


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
