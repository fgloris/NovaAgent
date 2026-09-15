"""ROS action 后端:把通用 EEF 轨迹转发给机器人 bridge 执行。"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable

from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState

from nova_interfaces.action import EEFExecute
from nova_interfaces.msg import EEFWaypoint

from .trajectory import Pose


class RosEEFBackend:
    """通过 EEFExecute action 与 /{robot}/eef_pose、/joint_states 话题操作机器人。

    缓存最新 EEF 位姿与关节状态,供相对轨迹合成与校验使用。
    """

    def __init__(
        self,
        node,
        action_name: str = "/nova/robot0/eef_execute",
        robot_id: str = "robot0",
    ) -> None:
        self.node = node
        self.action_name = action_name
        self.robot_id = robot_id
        self._lock = threading.Lock()
        self._pose: Pose | None = None
        self._joints: dict = {"name": [], "position": [], "velocity": []}
        group = ReentrantCallbackGroup()
        self.client = ActionClient(node, EEFExecute, action_name, callback_group=group)
        node.create_subscription(
            PoseStamped, f"/{robot_id}/eef_pose", self._pose_callback, 10, callback_group=group
        )
        node.create_subscription(
            JointState, f"/{robot_id}/joint_states", self._joint_callback, 10, callback_group=group
        )

    def _pose_callback(self, msg: PoseStamped) -> None:
        """缓存最新 EEF 位姿(位置 + XYZW 四元数)。"""
        with self._lock:
            self._pose = Pose(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
            )

    def _joint_callback(self, msg: JointState) -> None:
        """缓存最新关节状态(名称/位置/速度)。"""
        with self._lock:
            self._joints = {
                "name": list(msg.name),
                "position": list(msg.position),
                "velocity": list(msg.velocity),
            }

    def describe(self) -> dict:
        """返回机器人能力描述(基座坐标系、速度上限、工作空间等)。"""
        return {
            "robot_id": self.robot_id,
            "robot_base": f"{self.robot_id}_base",
            "supports_gripper": True,
            "max_linear_speed": 0.2,
            "max_angular_speed": 1.0,
            "workspace": {"min": [-0.8, -0.8, 0.0], "max": [0.8, 0.8, 1.2]},
        }

    def get_current_pose(self, robot_id: str) -> Pose:
        """读取缓存中的当前 EEF 位姿;无缓存或后端未就绪时抛异常。"""
        if robot_id != self.robot_id:
            raise ValueError(f"unknown_robot:{robot_id}")
        with self._lock:
            pose = self._pose
        if pose is None:
            if not self.client.server_is_ready():
                raise RuntimeError("backend_unavailable")
            raise RuntimeError("eef_state_unavailable")
        return Pose(list(pose.position), list(pose.orientation), pose.gripper)

    def transform_pose(self, pose: Pose, source: str, target: str) -> Pose:
        """坐标系转换占位实现:当前仅支持同一坐标系,否则抛 frame_unavailable。"""
        if source != target:
            raise ValueError(f"frame_unavailable:{source}")
        return pose

    def validate_trajectory(self, trajectory: list[Pose], constraints: dict) -> str | None:
        """轨迹预校验:后端不可用时返回错误码,否则返回 None。"""
        del trajectory, constraints
        if not self.client.server_is_ready():
            return "backend_unavailable"
        return None

    @staticmethod
    def _wait(future, timeout: float | None, cancel_callback: Callable[[], bool] | None = None):
        """轮询等待 future 完成;超时或被取消返回 None,否则返回结果。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not future.done():
            if cancel_callback and cancel_callback():
                return None
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.01)
        return future.result()

    def execute_trajectory(
        self,
        trajectory: list[Pose],
        feedback_callback: Callable,
        cancel_callback: Callable[[], bool],
        constraints: dict | None = None,
    ) -> dict:
        """把轨迹打包成 EEFExecute goal 发送并等待结果,支持取消与反馈回调。"""
        if not self.client.wait_for_server(timeout_sec=0.1):
            return {"success": False, "error": "backend_unavailable"}
        constraints = constraints or {}
        goal = EEFExecute.Goal()
        goal.robot_id = self.robot_id
        goal.frame_id = f"{self.robot_id}_base"
        goal.max_linear_speed = float(constraints.get("max_linear_speed", 0.0))
        goal.max_angular_speed = float(constraints.get("max_angular_speed", 0.0))
        for pose in trajectory:
            waypoint = EEFWaypoint()
            waypoint.position = [float(value) for value in pose.position]
            waypoint.orientation = [float(value) for value in pose.orientation]
            waypoint.has_gripper = pose.gripper is not None
            waypoint.gripper = float(pose.gripper or 0.0)
            goal.waypoints.append(waypoint)

        def on_feedback(message) -> None:
            feedback = message.feedback
            feedback_callback(
                max(0, int(feedback.current_waypoint) - 1),
                int(feedback.total_waypoints),
                feedback.status,
                feedback.message,
            )

        send_future = self.client.send_goal_async(goal, feedback_callback=on_feedback)
        goal_ref = self._wait(send_future, 5.0, cancel_callback)
        if goal_ref is None:
            return {"success": False, "error": "cancelled" if cancel_callback() else "backend_timeout"}
        if not goal_ref.accepted:
            return {"success": False, "error": "backend_rejected"}
        result_future = goal_ref.get_result_async()
        while not result_future.done():
            if cancel_callback():
                self.client.cancel_goal_async(goal_ref)
                self._wait(result_future, 5.0)
                return {"success": False, "error": "cancelled"}
            time.sleep(0.01)
        response = result_future.result().result
        if response.result_json:
            try:
                payload = json.loads(response.result_json)
            except json.JSONDecodeError:
                payload = {"result": response.result_json}
        else:
            payload = {}
        payload.setdefault("success", bool(response.success))
        if response.error:
            payload.setdefault("error", response.error)
        return payload
