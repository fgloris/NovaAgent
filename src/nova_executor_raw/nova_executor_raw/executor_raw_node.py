#!/usr/bin/env python3
import json, time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from nova_interfaces.action import MCPExecute
from nova_interfaces.msg import ExecutorHeartbeat, ToolDescriptor
from .trajectory import Pose, validate_waypoints, compose, interpolate, normalize

TOOLS = {
    "move_eef": (
        "Execute an absolute Cartesian EEF trajectory",
        {
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "frame_id": {"type": "string"},
                "waypoints": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "array", "minItems": 3, "maxItems": 3},
                            "orientation": {
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "gripper": {"type": "number"},
                        },
                        "required": ["position", "orientation"],
                    },
                },
                "max_linear_speed": {"type": "number"},
                "max_angular_speed": {"type": "number"},
            },
            "required": ["waypoints"],
        },
    ),
    "move_eef_relative": (
        "Execute waypoints relative to pose at invocation start",
        {
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "frame_id": {"type": "string"},
                "waypoints": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["position_delta", "rotation_delta"],
                    },
                },
            },
            "required": ["waypoints"],
        },
    ),
}


class DefaultBackend:
    def describe(self):
        """返回后备 backend 的能力和安全限制。"""
        return {
            "robot_id": "robot0",
            "robot_base": "robot0_base",
            "supports_gripper": False,
            "max_linear_speed": 0.2,
            "max_angular_speed": 1.0,
            "workspace": {"min": [-0.8, -0.8, 0.0], "max": [0.8, 0.8, 1.2]},
        }

    def get_current_pose(self, robot_id):
        """返回指定机器人的当前 EEF 位姿。"""
        return Pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    def transform_pose(self, pose, source, target):
        """在坐标系之间转换位姿；后备实现保持位姿不变。"""
        return pose

    def validate_trajectory(self, traj, constraints):
        """执行 backend 专属的轨迹执行前检查。"""
        return None

    def execute_trajectory(self, traj, feedback_callback, cancel_callback):
        """执行轨迹，同时发布反馈并响应取消请求。"""
        for i, _ in enumerate(traj):
            if cancel_callback():
                return {"success": False, "error": "cancelled"}
            feedback_callback(i, len(traj), "running", "executing")
            time.sleep(0.001)
        return {"success": True, "executed_points": len(traj)}


class ExecutorRawNode(Node):
    def __init__(self):
        """创建 MCP action server 并启动能力心跳。"""
        super().__init__("nova_executor_raw")
        self.backend = DefaultBackend()
        self._servers = []
        for name in TOOLS:
            self._servers.append(
                ActionServer(
                    self,
                    MCPExecute,
                    f"/{self.get_name()}/{name}/execute",
                    self._callback(name),
                )
            )
        self.pub = self.create_publisher(
            ExecutorHeartbeat, "/nova/executors/heartbeat", 10
        )
        self.create_timer(1.0, self._heartbeat)
        self._heartbeat()

    def _heartbeat(self):
        """发布当前 EEF 工具的描述信息。"""
        hb = ExecutorHeartbeat()
        hb.executor_name = self.get_name()
        for n, (d, s) in TOOLS.items():
            t = ToolDescriptor()
            t.name = n
            t.description = d
            t.params_schema_json = json.dumps(s)
            t.action_server_name = f"/{self.get_name()}/{n}/execute"
            hb.tools.append(t)
        self.pub.publish(hb)

    def _callback(self, name):
        """构造绑定指定工具名称的 action 回调。"""
        def callback(goal_handle):
            """校验、插值、预检查并执行一个 action goal。"""
            try:
                p = json.loads(goal_handle.request.params_json or "{}")
                b = self.backend.describe()
                rid = p.get("robot_id", b.get("robot_id"))
                frame = p.get("frame_id", b.get("robot_base"))
                if name == "move_eef":
                    if any(
                        "gripper" in w for w in p.get("waypoints", [])
                    ) and not b.get("supports_gripper", False):
                        raise ValueError("gripper_unsupported")
                    poses = validate_waypoints(p.get("waypoints"))
                    start = None
                else:
                    start = self.backend.get_current_pose(rid)
                    raw = []
                    for w in p.get("waypoints", []):
                        pos = w.get("position_delta")
                        rot = w.get("rotation_delta")
                        if not isinstance(pos, list) or len(pos) != 3:
                            raise ValueError("invalid_pose")
                        raw.append(
                            compose(
                                start, Pose([float(x) for x in pos], normalize(rot))
                            )
                        )
                    poses = raw
                poses = [
                    self.backend.transform_pose(x, frame, b.get("robot_base", frame))
                    for x in poses
                ]
                ws = b.get("workspace")
                if ws:
                    for x in poses:
                        if any(
                            x.position[i] < ws["min"][i] or x.position[i] > ws["max"][i]
                            for i in range(3)
                        ):
                            raise ValueError("workspace_violation")
                traj = interpolate(
                    poses,
                    min(float(p.get("max_linear_speed", b.get("max_linear_speed", 0.2))),b.get("max_linear_speed", 0.2)),
                    min(float(p.get("max_angular_speed", b.get("max_angular_speed", 1.0))),b.get("max_angular_speed", 1.0)),
                )
                vr = self.backend.validate_trajectory(traj, p)
                if vr:
                    raise ValueError(str(vr))

                def feedback(i, total, status, msg):
                    f = MCPExecute.Feedback()
                    f.status = status
                    f.message = f"{msg} ({i + 1}/{total})"
                    goal_handle.publish_feedback(f)

                out = self.backend.execute_trajectory(
                    traj, feedback, lambda: goal_handle.is_cancel_requested
                )
                r = MCPExecute.Result()
                r.success = out.get("success", False)
                r.error = out.get("error", "")
                r.result_json = json.dumps(out)
                goal_handle.canceled() if r.error == "cancelled" else (
                    goal_handle.succeed() if r.success else goal_handle.abort()
                )
                return r
            except Exception as e:
                r = MCPExecute.Result()
                r.success = False
                r.error = str(e)
                goal_handle.abort()
                return r

        return callback


def main(args=None):
    """初始化 ROS、运行 executor 节点并安全关闭。"""
    rclpy.init(args=args)
    n = ExecutorRawNode()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()
