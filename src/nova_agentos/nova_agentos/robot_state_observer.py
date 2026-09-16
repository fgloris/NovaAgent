#!/usr/bin/env python3
# AgentOS 机器人状态缓存:从 /nova/env/info 的 robot.state_topics 发现状态话题,
# 常驻订阅并缓存最新一帧,每轮规划作为独立观测注入(tool-independent)。
from __future__ import annotations

import json
import threading
import time

from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from nova_interfaces.srv import EnvInfo

DEFAULT_ROBOT_ID = "robot0"


class RobotStateObserver:
    """订阅机器人 EEF/关节/夹爪状态话题,缓存最新帧并生成 VLM 观测消息。"""

    def __init__(
        self,
        node: Node,
        env_ns: str = "/nova/env",
        robot_id: str = DEFAULT_ROBOT_ID,
        refresh_sec: float = 2.0,
    ) -> None:
        self.node = node
        self.env_ns = env_ns.rstrip("/")
        self.robot_id = robot_id
        self._lock = threading.Lock()
        self._eef: tuple | None = None
        self._joints: tuple | None = None
        self._gripper: tuple | None = None
        self._subs: dict[str, bool] = {}
        self._info_cg = MutuallyExclusiveCallbackGroup()
        self._info_client = node.create_client(
            EnvInfo, f"{self.env_ns}/info", callback_group=self._info_cg
        )
        node.create_timer(max(0.5, float(refresh_sec)), self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """异步查询 /nova/env/info,从中发现机器人状态话题。"""
        if not self._info_client.service_is_ready():
            return
        future = self._info_client.call_async(EnvInfo.Request())
        future.add_done_callback(self._info_done)

    def _info_done(self, future) -> None:
        try:
            response = future.result()
            if not response or not response.success:
                return
            info = json.loads(response.spec_json or "{}")
            robot = info.get("robot") or {}
            topics = robot.get("state_topics") or {}
            robot_id = str(robot.get("robot_id") or self.robot_id)
        except Exception as exc:
            self.node.get_logger().warn(f"刷新机器人状态话题失败: {exc}")
            return
        self.robot_id = robot_id
        self._ensure_sub("eef_pose", topics.get("eef_pose") or f"/{robot_id}/eef_pose", PoseStamped)
        self._ensure_sub("joint_states", topics.get("joint_states") or f"/{robot_id}/joint_states", JointState)
        self._ensure_sub(
            "gripper_state", topics.get("gripper_state") or f"/{robot_id}/gripper_state", Float32MultiArray
        )

    def _ensure_sub(self, key: str, topic: str, msg_type) -> None:
        with self._lock:
            if key in self._subs:
                return
            self._subs[key] = True
        self.node.create_subscription(msg_type, str(topic), self._make_cb(key), 10)
        self.node.get_logger().info(f"AgentOS 订阅机器人状态: {topic}")

    def _make_cb(self, key: str):
        def cb(msg) -> None:
            now = time.time()
            with self._lock:
                if key == "eef_pose":
                    self._eef = (
                        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                        [
                            msg.pose.orientation.x,
                            msg.pose.orientation.y,
                            msg.pose.orientation.z,
                            msg.pose.orientation.w,
                        ],
                        now,
                    )
                elif key == "joint_states":
                    self._joints = (list(msg.name), list(msg.position), now)
                else:
                    self._gripper = (list(msg.data), now)

        return cb

    def gripper_state(self) -> list[float] | None:
        """返回最新夹爪关节位置;无数据时返回 None(供图像去重门控使用)。"""
        with self._lock:
            gripper = self._gripper
        return list(gripper[0]) if gripper is not None else None

    def snapshot_message(self) -> dict | None:
        """返回一条机器人状态观测消息;三路状态均未到达时返回 None。"""
        with self._lock:
            eef, joints, gripper = self._eef, self._joints, self._gripper
        if eef is None and joints is None and gripper is None:
            return None
        now = time.time()
        payload: dict = {"robot_id": self.robot_id}
        if eef is not None:
            payload["eef"] = {
                "position": eef[0],
                "orientation": eef[1],
                "age_sec": round(now - eef[2], 3),
            }
        if joints is not None:
            payload["joints"] = {
                "name": joints[0],
                "position": joints[1],
                "age_sec": round(now - joints[2], 3),
            }
        if gripper is not None:
            payload["gripper"] = {"position": gripper[0], "age_sec": round(now - gripper[1], 3)}
        return {"role": "user", "content": "# 机器人状态\n" + json.dumps(payload, ensure_ascii=False)}
