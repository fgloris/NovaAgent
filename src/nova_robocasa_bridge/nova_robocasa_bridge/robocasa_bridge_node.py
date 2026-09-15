#!/usr/bin/env python3
"""RoboCasa ROS 桥:在 EnvBridgeBase 基础上补齐机器人状态发布与 EEF action 执行。"""
from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from nova_common.env_bridge import EnvBridgeBase
from nova_interfaces.action import EEFExecute
from nova_robot_description import load_robot_description, to_json, to_markdown


class RoboCasaBridgeNode(EnvBridgeBase):
    """RoboCasa 环境桥:额外提供 EEF 轨迹 action 与 /{robot}/eef_pose 等状态话题。"""

    def __init__(self) -> None:
        super().__init__("robocasa_bridge", action_dim_default=12)

    def _declare_custom_params(self) -> None:
        self._request_lock = threading.RLock()
        self._trajectory_lock = threading.Lock()
        self._control_active = threading.Event()
        self.robot_state: dict[str, Any] | None = None
        self.declare_parameter("env_id", "robocasa/PickPlaceCounterToCabinet")
        self.declare_parameter("seed", 0)
        self.env_id = str(self.get_parameter("env_id").value)
        self.seed = int(self.get_parameter("seed").value)
        self.declare_parameter("robot_id", "robot0")
        self.declare_parameter("publish_robot_state", True)
        self.declare_parameter("eef_action_name", "")
        self.declare_parameter("feedback_stride", 5)
        self.robot_id = str(self.get_parameter("robot_id").value)
        self.publish_robot_state = bool(self.get_parameter("publish_robot_state").value)
        self.eef_action_name = str(self.get_parameter("eef_action_name").value) or f"/nova/{self.robot_id}/eef_execute"
        self.feedback_stride = max(1, int(self.get_parameter("feedback_stride").value))
        self.robot_description = load_robot_description("panda_omron")
        self.eef_pub = self.create_publisher(PoseStamped, f"/{self.robot_id}/eef_pose", 10)
        self.joint_pub = self.create_publisher(JointState, f"/{self.robot_id}/joint_states", 10)
        self.gripper_pub = self.create_publisher(
            Float32MultiArray, f"/{self.robot_id}/gripper_state", 10
        )
        self._eef_action_server = ActionServer(
            self,
            EEFExecute,
            self.eef_action_name,
            execute_callback=self._execute_eef,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

    def _build_reset_request(self) -> dict[str, Any]:
        """构造 RoboCasa 的 reset 请求(env_id/seed/相机尺寸)。"""
        return {
            "type": "reset",
            "env_id": self.env_id,
            "seed": self.seed,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
        }

    def action_vector_to_native(self, values: np.ndarray):
        """RoboCasa 直接使用规范动作向量,不做额外转换。"""
        return values

    def _extra_info(self) -> dict[str, Any]:
        """在 sim 信息上补充机器人描述(JSON/Markdown/哈希)与状态话题。"""
        info = dict(self.sim_info)
        info.setdefault("sim", "robocasa")
        robot = to_json(self.robot_description)
        robot.update(
            {
                "description_name": "panda_omron",
                "context_schema": "robot_context_v1",
                "context_json": to_json(self.robot_description),
                "context_markdown": to_markdown(self.robot_description),
                "description_sha256": self.robot_description.source.get("sha256", ""),
                "state_topics": {
                    "eef_pose": f"/{self.robot_id}/eef_pose",
                    "joint_states": f"/{self.robot_id}/joint_states",
                    "gripper_state": f"/{self.robot_id}/gripper_state",
                },
                "tf_enabled": False,
            }
        )
        info["robot"] = robot
        return info

    def _observation_info(self) -> dict[str, Any]:
        """只挑 sim/robots/controller/env_id 作为每帧观测元数据。"""
        return {
            key: self.sim_info[key]
            for key in ("sim", "robots", "controller", "env_id")
            if key in self.sim_info
        }

    def _absorb_response(self, response: dict[str, Any]) -> None:
        """吸收响应时顺带更新缓存的机器人状态。"""
        super()._absorb_response(response)
        state = response.get("robot_state")
        if isinstance(state, dict):
            self.robot_state = state

    def _reset_env(self) -> None:
        """加请求锁执行 reset,避免与 EEF 控制并发访问 sim server。"""
        with self._request_lock:
            super()._reset_env()

    def _step_once(self) -> None:
        """加请求锁执行普通步进;EEF 控制激活时跳过,避免动作互相干扰。"""
        if self._control_active.is_set():
            return
        with self._request_lock:
            if self._control_active.is_set():
                return
            super()._step_once()

    def action_callback(self, msg: Float32MultiArray) -> None:
        """接收普通动作;EEF 控制进行中时忽略,防止抢占。"""
        if self._control_active.is_set():
            self.get_logger().warn("Ignoring /nova/env/action_cmd while EEF control is active")
            return
        super().action_callback(msg)

    def _publish_observation(self) -> None:
        """发布观测后,把缓存的机器人状态转成 eef_pose/joint_states/gripper_state 话题。"""
        super()._publish_observation()
        if not self.publish_robot_state or not self.robot_state:
            return
        state = self.robot_state
        stamp = self.get_clock().now().to_msg()
        eef = state.get("eef") or {}
        position = eef.get("position")
        orientation = eef.get("orientation")
        if position is not None and orientation is not None:
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = str(state.get("base_frame") or f"{self.robot_id}_base")
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, position)
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ) = map(float, orientation)
            self.eef_pub.publish(msg)
        joints = state.get("joints") or {}
        names = [str(name) for name in joints.get("name") or []]
        positions = [float(value) for value in joints.get("position") or []]
        velocities = [float(value) for value in joints.get("velocity") or []]
        if names and len(names) == len(positions):
            msg = JointState()
            msg.header.stamp = stamp
            msg.name = names
            msg.position = positions
            if len(velocities) == len(names):
                msg.velocity = velocities
            self.joint_pub.publish(msg)
        gripper = (state.get("gripper") or {}).get("position")
        if gripper is not None:
            msg = Float32MultiArray()
            msg.data = [float(value) for value in gripper]
            self.gripper_pub.publish(msg)

    @staticmethod
    def _waypoint_dict(waypoint) -> dict[str, Any]:
        """把 EEFWaypoint 消息转成 sim server 需要的 dict(可选带 gripper)。"""
        item = {
            "position": [float(value) for value in waypoint.position],
            "orientation": [float(value) for value in waypoint.orientation],
        }
        if waypoint.has_gripper:
            item["gripper"] = float(waypoint.gripper)
        return item

    def _execute_eef(self, goal_handle):
        """执行 EEF 轨迹 action:先校验轨迹,再逐点 step_eef 并持续发布反馈。"""
        result = EEFExecute.Result()
        goal = goal_handle.request
        if goal.robot_id and goal.robot_id != self.robot_id:
            result.error = f"unknown_robot:{goal.robot_id}"
            goal_handle.abort()
            return result
        base_frame = self.robot_description.base_frame
        if goal.frame_id and goal.frame_id != base_frame:
            result.error = f"frame_unavailable:{goal.frame_id}"
            goal_handle.abort()
            return result
        waypoints = [self._waypoint_dict(item) for item in goal.waypoints]
        if not waypoints:
            result.error = "empty_waypoints"
            goal_handle.abort()
            return result
        if not self._trajectory_lock.acquire(blocking=False):
            result.error = "control_busy"
            goal_handle.abort()
            return result
        self._control_active.set()
        try:
            with self._request_lock:
                validation = self.client.request(
                    {
                        "type": "validate_eef_trajectory",
                        "robot_id": self.robot_id,
                        "frame_id": base_frame,
                        "waypoints": waypoints,
                    }
                )
                if not validation.get("valid", False):
                    raise RuntimeError(validation.get("error", "trajectory_rejected"))
                total = len(waypoints)
                last_response: dict[str, Any] = validation
                for index, waypoint in enumerate(waypoints):
                    if goal_handle.is_cancel_requested:
                        result.error = "cancelled"
                        result.result_json = json.dumps(
                            {"success": False, "error": "cancelled", "executed_points": index}
                        )
                        goal_handle.canceled()
                        return result
                    last_response = self.client.request(
                        {"type": "step_eef", "robot_id": self.robot_id, "waypoint": waypoint}
                    )
                    self._absorb_response(last_response)
                    self.step_count += 1
                    self._ensure_camera_publishers(self.obs_spec.get("cameras", {}))
                    self._publish_observation()
                    # 反馈只报进度,机器人状态走独立话题/观测通道;按 stride 节流
                    if index % self.feedback_stride == 0 or index == total - 1:
                        feedback = EEFExecute.Feedback()
                        feedback.current_waypoint = index + 1
                        feedback.total_waypoints = total
                        feedback.status = "running"
                        feedback.message = json.dumps(
                            {"waypoint": index + 1, "total": total},
                            ensure_ascii=False,
                        )
                        goal_handle.publish_feedback(feedback)
                payload = {
                    "success": True,
                    "executed_points": total,
                    "robot_state": last_response.get("robot_state"),
                }
                result.success = True
                result.result_json = json.dumps(payload, ensure_ascii=False)
                goal_handle.succeed()
                return result
        except Exception as exc:
            result.success = False
            result.error = str(exc)
            result.result_json = json.dumps({"success": False, "error": str(exc)})
            goal_handle.abort()
            return result
        finally:
            self.latest_action = self._zero_action()
            self._control_active.clear()
            self._trajectory_lock.release()


def main(args=None) -> int:
    """用多线程 executor 运行 RoboCasa 桥节点。"""
    rclpy.init(args=args)
    node = RoboCasaBridgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
