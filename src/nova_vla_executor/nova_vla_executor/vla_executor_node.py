#!/usr/bin/env python3
# VLA executor bridge:本地 ROS2 侧,连接远程 pi0 推理服务。
# 注册 pi0_policy 工具(声明 obs_bindings),由 AgentOS 注入 session 命名空间;
# 执行期间订阅 session 相机/state,调用远程 HTTP 推理,把动作发回 <ns>/action_cmd。
import json
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from nova_interfaces.action import MCPExecute
from nova_interfaces.msg import ExecutorHeartbeat, ToolDescriptor

from nova_vla_executor.pi0_bridge import RemotePi0Client

HEARTBEAT_TOPIC = "/nova/executors/heartbeat"
_CAM_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


def _image_to_numpy(msg: Image) -> np.ndarray:
    return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))


class VLAExecutorNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_vla_executor")
        self.declare_parameter("server_url", "http://127.0.0.1:8001")
        self.declare_parameter("request_timeout_sec", 60.0)
        self.declare_parameter("heartbeat_rate_hz", 1.0)
        self.declare_parameter("camera_names", ["agentview", "robot0_eye_in_hand"])
        self.declare_parameter("state_keys", ["robot0_eef_pos"])
        self.declare_parameter("default_duration_sec", 10.0)

        self.server_url = str(self.get_parameter("server_url").value)
        self.request_timeout = float(self.get_parameter("request_timeout_sec").value)
        rate = float(self.get_parameter("heartbeat_rate_hz").value)
        self.camera_names = list(self.get_parameter("camera_names").value)
        self.state_keys = list(self.get_parameter("state_keys").value)
        self.default_duration = float(self.get_parameter("default_duration_sec").value)

        self.client = RemotePi0Client(self.server_url, self.request_timeout)
        self._bindings = {"cameras": self.camera_names, "state": self.state_keys}

        self._action_server = ActionServer(
            self, MCPExecute, f"/{self.get_name()}/pi0_policy/execute", self._execute_cb
        )
        self._heartbeat_pub = self.create_publisher(ExecutorHeartbeat, HEARTBEAT_TOPIC, 1)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_heartbeat)
        self._publish_heartbeat()
        self.get_logger().info(
            f"VLA executor 就绪,远程推理: {self.server_url},"
            f" cameras={self.camera_names}, state_keys={self.state_keys}"
        )

    def _publish_heartbeat(self) -> None:
        hb = ExecutorHeartbeat()
        hb.executor_name = self.get_name()
        tool = ToolDescriptor()
        tool.name = "pi0_policy"
        tool.description = (
            "远程 pi0 VLA 策略:订阅 AgentOS 注入的 session 相机/state,"
            "调用远程推理并把动作回灌到仿真"
        )
        tool.params_schema_json = json.dumps(
            {
                "type": "object",
                "properties": {
                    "topic_namespace": {"type": "string", "description": "AgentOS 注入的 session 命名空间,无需用户提供"},
                    "duration_sec": {"type": "number", "description": "策略运行秒数"},
                    "instruction": {"type": "string", "description": "可选的指令覆盖(默认取 state JSON 的 instruction)"},
                },
                "required": ["topic_namespace"],
            }
        )
        tool.action_server_name = f"/{self.get_name()}/pi0_policy/execute"
        tool.obs_bindings = json.dumps(self._bindings)
        hb.tools.append(tool)
        self._heartbeat_pub.publish(hb)

    def _execute_cb(self, goal_handle):
        goal = goal_handle.request
        try:
            params = json.loads(goal.params_json) if goal.params_json else {}
            result_json = self._run_policy(params)
            result = MCPExecute.Result()
            result.success = True
            result.result_json = json.dumps(result_json, ensure_ascii=False)
            result.error = ""
            goal_handle.succeed()
            self.get_logger().info(f"pi0_policy 完成: {result_json}")
            return result
        except Exception as exc:
            result = MCPExecute.Result()
            result.success = False
            result.result_json = ""
            result.error = str(exc)
            goal_handle.abort()
            self.get_logger().error(f"pi0_policy 失败: {exc}")
            return result

    def _run_policy(self, params: dict) -> dict:
        ns = str(params.get("topic_namespace", "")).rstrip("/")
        duration_sec = float(params.get("duration_sec", self.default_duration))
        instruction_override = str(params.get("instruction", "")).strip() or None
        if not ns:
            return {"ok": False, "executed": False, "error": "缺少 topic_namespace(需 AgentOS 注入)"}
        base = ns + "/"

        buf = {"frames": {}, "doc": None, "step": None, "dim": None}
        subs = []
        for cam in self.camera_names:
            sub = self.create_subscription(
                Image, f"{base}camera/{cam}/image_raw", self._make_cam_cb(buf, cam), _CAM_QOS
            )
            subs.append(sub)
        sub_state = self.create_subscription(
            String, f"{base}state", self._make_state_cb(buf), 10
        )
        subs.append(sub_state)
        pub_action = self.create_publisher(Float32MultiArray, f"{base}action_cmd", 10)
        try:
            # 等全部相机帧与 state 都就绪(容忍动态 topic discovery 延迟)
            deadline = time.time() + 5.0
            while time.time() < deadline and (
                len(buf["frames"]) < len(self.camera_names) or buf["doc"] is None
            ):
                rclpy.spin_once(self, timeout_sec=0.2)
            if not buf["frames"] or buf["doc"] is None:
                self.get_logger().warn(f"未收到完整观测({ns}),跳过")
                return {"ok": False, "executed": False, "error": "未收到相机/state 帧"}

            last_step = -1
            start = time.time()
            n_infer = 0
            n_error = 0
            while time.time() - start < duration_sec:
                rclpy.spin_once(self, timeout_sec=0.05)
                if buf["step"] is None or buf["step"] == last_step:
                    continue
                last_step = buf["step"]
                try:
                    state = self._build_state_vector(buf)
                    action = self.client.predict(
                        dict(buf["frames"]),
                        instruction_override or (buf["doc"].get("instruction") or ""),
                        state,
                    )
                    dim = buf["dim"]
                    if dim and action.size != dim:
                        self.get_logger().warn(
                            f"动作维度不匹配 remote={action.size} expected={dim},丢弃"
                        )
                        continue
                    msg = Float32MultiArray()
                    msg.data = action.tolist()
                    pub_action.publish(msg)
                    n_infer += 1
                except Exception as exc:
                    n_error += 1
                    self.get_logger().error(f"远程推理失败: {exc}")
            return {"ok": True, "executed": True, "infer_steps": n_infer, "errors": n_error}
        finally:
            for sub in subs:
                self.destroy_subscription(sub)
            self.destroy_publisher(pub_action)

    # 按 state_keys 从 obs JSON 的 state 摘取并拼接 state 向量;键缺失则跳过
    def _build_state_vector(self, buf: dict) -> np.ndarray | None:
        state = (buf["doc"] or {}).get("state") or {}
        values = []
        for key in self.state_keys:
            value = state.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                values.extend(float(x) for x in value)
            elif isinstance(value, (int, float)):
                values.append(float(value))
        return np.asarray(values, dtype=np.float32) if values else None

    @staticmethod
    def _make_cam_cb(buf: dict, cam: str):
        def cb(msg):
            buf["frames"][cam] = _image_to_numpy(msg)

        return cb

    @staticmethod
    def _make_state_cb(buf: dict):
        def cb(msg):
            try:
                doc = json.loads(msg.data)
                buf["doc"] = doc
                buf["step"] = int(doc.get("step_count", 0))
                spec = doc.get("action_spec") or {}
                dim = spec.get("dim")
                buf["dim"] = int(dim) if dim else None
            except Exception:
                pass

        return cb


def main(args=None) -> int:
    rclpy.init(args=args)
    node = VLAExecutorNode()
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
