#!/usr/bin/env python3
"""EEF raw executor 手动测试客户端。

读取当前 EEF 位姿,向 nova_executor_raw 的 move_eef / move_eef_relative action
发送一条轨迹 goal,打印反馈与结果,并回显执行后的 EEF 位姿。
"""
from __future__ import annotations

import argparse
import json
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node

from nova_interfaces.action import MCPExecute


class EefTestClient(Node):
    def __init__(self, robot_id: str, action_name: str) -> None:
        super().__init__("eef_test_client")
        self.action_name = action_name
        self._pose = None
        self.create_subscription(PoseStamped, f"/{robot_id}/eef_pose", self._on_pose, 10)
        self.client = ActionClient(self, MCPExecute, action_name)

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg.pose

    def wait_pose(self, timeout: float = 5.0):
        deadline = time.time() + timeout
        while rclpy.ok() and self._pose is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self._pose

    @staticmethod
    def _pose_values(pose) -> dict:
        return {
            "position": [pose.position.x, pose.position.y, pose.position.z],
            "orientation": [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
        }

    def send(self, tool: str, params: dict, trace_id: str, timeout: float) -> int:
        if not self.client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"action server 不可用: {self.action_name}")
        goal = MCPExecute.Goal()
        goal.tool_name = tool
        goal.params_json = json.dumps(params, ensure_ascii=False)
        goal.trace_id = trace_id
        self.get_logger().info(f"发送 {tool} -> {self.action_name}: {goal.params_json}")

        send_future = self.client.send_goal_async(goal, feedback_callback=self._on_feedback)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        goal_ref = send_future.result()
        if goal_ref is None or not goal_ref.accepted:
            self.get_logger().error("goal 被拒绝")
            return 1
        result_future = goal_ref.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().error("执行超时")
            return 1
        result = result_future.result().result
        self.get_logger().info(
            f"结果 success={result.success} error={result.error!r} result_json={result.result_json}"
        )
        return 0 if result.success else 1

    def _on_feedback(self, message) -> None:
        feedback = message.feedback
        self.get_logger().info(f"[feedback] {feedback.status}: {feedback.message}")


def _build_params(client: EefTestClient, args) -> dict:
    current = client.wait_pose()
    if current is None:
        raise RuntimeError("未收到 /eef_pose,先确认 bridge 已 reset 并发布状态")
    pose = client._pose_values(current)
    if args.tool == "move_eef_relative":
        waypoint = {"position_delta": list(args.delta), "rotation_delta": list(args.rotation)}
    else:
        position = args.position or [pose["position"][i] + args.delta[i] for i in range(3)]
        waypoint = {"position": position, "orientation": pose["orientation"]}
    if args.gripper is not None:
        waypoint["gripper"] = args.gripper
    return {"waypoints": [waypoint]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", default="robot0")
    parser.add_argument("--tool", default="move_eef_relative", choices=["move_eef_relative", "move_eef"])
    parser.add_argument("--delta", type=float, nargs=3, default=[0.0, 0.0, 0.05], help="相对位移/绝对目标增量(m)")
    parser.add_argument("--rotation", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0], help="相对旋转四元数 xyzw")
    parser.add_argument("--position", type=float, nargs=3, default=None, help="move_eef 绝对目标位置")
    parser.add_argument("--gripper", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    rclpy.init()
    client = EefTestClient(args.robot_id, f"/nova_executor_raw/{args.tool}/execute")
    code = 1
    try:
        before = client.wait_pose()
        if before is None:
            raise RuntimeError("未收到 /eef_pose,先确认 bridge 已 reset 并发布状态")
        client.get_logger().info(f"执行前 EEF: {client._pose_values(before)}")
        params = _build_params(client, args)
        code = client.send(args.tool, params, "eef_test", args.timeout)
        deadline = time.time() + 0.5
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(client, timeout_sec=0.05)
        client.get_logger().info(f"执行后 EEF: {client._pose_values(client._pose) if client._pose else None}")
    except Exception as exc:
        client.get_logger().error(str(exc))
    finally:
        client.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
