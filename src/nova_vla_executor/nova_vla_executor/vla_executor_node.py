#!/usr/bin/env python3
# VLA executor bridge:本地 ROS2 侧,连接远程 pi0 推理服务。
# 静态绑定 /nova/env/*:启动时常驻订阅 env 相机/state,滚动缓存最新帧;
# pi0_policy 被调用时直接用缓存推理,把动作回灌到 /nova/env/action_cmd。
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
ENV_NS = "/nova/env"
_CAM_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


def _image_to_numpy(msg: Image) -> np.ndarray:
    return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))


class VLAExecutorNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_vla_executor")
        self.declare_parameter("server_url", "http://127.0.0.1:8767")
        self.declare_parameter("request_timeout_sec", 60.0)
        self.declare_parameter("heartbeat_rate_hz", 1.0)
        # 静态绑定 /nova/env/* 的相机/state 键,必须与运行中的 env 一致(robocasa groot fork)。
        # state_keys 顺序 = 模型训练时 groot_openpi_dataset 的 state 拼接顺序(eef 在前,base 在后)
        self.declare_parameter("camera_names", ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"])
        self.declare_parameter("state_keys", ["body.end_effector_position_relative", "body.end_effector_rotation_relative", "body.base_position", "body.base_rotation", "hand.gripper_qpos"])
        self.declare_parameter("default_duration_sec", 10.0)

        self.server_url = str(self.get_parameter("server_url").value)
        self.request_timeout = float(self.get_parameter("request_timeout_sec").value)
        rate = float(self.get_parameter("heartbeat_rate_hz").value)
        self.camera_names = list(self.get_parameter("camera_names").value)
        self.state_keys = list(self.get_parameter("state_keys").value)
        self.default_duration = float(self.get_parameter("default_duration_sec").value)

        self.client = RemotePi0Client(self.server_url, self.request_timeout)

        # 常驻订阅 + 滚动缓存:被调用时直接用最新帧,无 discovery 等待
        self._buf = {"frames": {}, "doc": None, "step": None, "dim": None}
        for cam in self.camera_names:
            self.create_subscription(
                Image,
                f"{ENV_NS}/camera/{cam}/image_raw",
                self._make_cam_cb(cam),
                _CAM_QOS,
            )
        self.create_subscription(String, f"{ENV_NS}/obs", self._make_obs_cb(), 10)
        self._action_pub = self.create_publisher(Float32MultiArray, f"{ENV_NS}/action_cmd", 10)

        self._action_server = ActionServer(
            self, MCPExecute, f"/{self.get_name()}/pi0_policy/execute", self._execute_cb
        )
        self._heartbeat_pub = self.create_publisher(ExecutorHeartbeat, HEARTBEAT_TOPIC, 1)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_heartbeat)
        self._publish_heartbeat()
        self.get_logger().info(
            f"VLA executor 就绪,远程推理: {self.server_url},"
            f" cameras={self.camera_names}, state_keys={self.state_keys}, env={ENV_NS}"
        )

    def _publish_heartbeat(self) -> None:
        hb = ExecutorHeartbeat()
        hb.executor_name = self.get_name()
        tool = ToolDescriptor()
        tool.name = "pi0_policy"
        tool.description = (
            "远程 pi0 VLA 策略:读取 /nova/env/* 最新相机/state,"
            "调用远程推理并把动作回灌到仿真,直到任务成功或超时"
        )
        tool.params_schema_json = json.dumps(
            {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "任务语言指令(可选,缺省用环境提供的 task_description)"},
                },
            }
        )
        tool.action_server_name = f"/{self.get_name()}/pi0_policy/execute"
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
        instruction_override = str(params.get("instruction", "")).strip() or None
        buf = self._buf
        if not buf["frames"] or buf["doc"] is None:
            self.get_logger().warn(f"尚未收到完整观测({ENV_NS}),跳过")
            return {"ok": False, "executed": False, "error": "未收到相机/state 帧"}

        last_step = -1
        start = time.time()
        n_infer = 0
        n_error = 0
        while time.time() - start < self.default_duration:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (buf["doc"] or {}).get("success"):
                break
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
                self._action_pub.publish(msg)
                n_infer += 1
            except Exception as exc:
                n_error += 1
                self.get_logger().error(f"远程推理失败: {exc}")
        return {"ok": True, "executed": True, "infer_steps": n_infer, "errors": n_error}

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
        if values:
            return np.asarray(values, dtype=np.float32)
        dim = buf["dim"]
        if dim:
            self.get_logger().warn(
                f"state 键 {self.state_keys} 缺失,可用键={list(state.keys())},使用零向量(dim={dim})"
            )
            return np.zeros(dim, dtype=np.float32)
        return None

    def _make_cam_cb(self, cam: str):
        def cb(msg):
            self._buf["frames"][cam] = _image_to_numpy(msg)

        return cb

    def _make_obs_cb(self):
        def cb(msg):
            try:
                doc = json.loads(msg.data)
                self._buf["doc"] = doc
                self._buf["step"] = int(doc.get("step_count", 0))
                spec = doc.get("action_spec") or {}
                dim = spec.get("dim")
                self._buf["dim"] = int(dim) if dim else None
            except Exception:
                pass

        return cb

    def destroy_node(self) -> bool:
        self.client.close()
        return super().destroy_node()


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
