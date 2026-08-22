#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import socket
import traceback
from typing import Any

os.environ.setdefault("ROS_LOG_DIR", "/tmp/novaagent_ros_logs")
os.makedirs(os.environ["ROS_LOG_DIR"], exist_ok=True)

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from std_srvs.srv import Trigger


ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)


# 12 维动作向量 → RoboCasa 动作 dict(缺省补零并截断到 [-1,1])
def action_vector_to_dict(values: np.ndarray) -> dict[str, list[float]]:
    values = np.ravel(values).astype(np.float32)
    if values.size > 12:
        raise ValueError(f"expected at most 12 action values, got {values.size}")
    if values.size < 12:
        values = np.pad(values, (0, 12 - values.size))
    values = np.clip(values, -1.0, 1.0)

    return {
        "action.end_effector_position": values[0:3].tolist(),
        "action.end_effector_rotation": values[3:6].tolist(),
        "action.gripper_close": [1.0 if values[6] >= 0.5 else 0.0],
        "action.base_motion": values[7:11].tolist(),
        "action.control_mode": [1.0 if values[11] >= 0.5 else 0.0],
    }


def zero_action_dict() -> dict[str, list[float]]:
    return action_vector_to_dict(np.zeros(12, dtype=np.float32))


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    data = base64.b64decode(payload["data"])
    array = np.frombuffer(data, dtype=np.dtype(payload["dtype"]))
    return array.reshape(payload["shape"])


# 递归还原观测:把 base64 编码的 __ndarray__ 解码回 numpy 数组
def decode_observation(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        return decode_array(value)
    if isinstance(value, dict):
        return {str(k): decode_observation(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_observation(v) for v in value]
    return value


# 压缩观测值用于 topic 发布,大数组只保留 shape/dtype/min/max
def summarize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size > 32:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(np.nanmin(value)) if value.size else 0.0,
                "max": float(np.nanmax(value)) if value.size else 0.0,
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): summarize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize_value(v) for v in value]
    return value


class JsonLineClient:
    def __init__(self, host: str, port: int, timeout_sec: float) -> None:
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.sock: socket.socket | None = None
        self.file = None

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    # 发送一行 JSON 请求并读取一行响应,失败时关闭连接
    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._connect()
        assert self.sock is not None
        assert self.file is not None
        try:
            line = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            self.sock.sendall(line)
            response_line = self.file.readline()
            if not response_line:
                raise ConnectionError("sim server closed the connection")
            response = json.loads(response_line.decode("utf-8"))
        except Exception:
            self.close()
            raise

        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "sim server request failed"))
        return response

    def _connect(self) -> None:
        if self.sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
        sock.settimeout(self.timeout_sec)
        self.sock = sock
        self.file = sock.makefile("rb")


class RoboCasaBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("robocasa_bridge")

        self.declare_parameter("env_id", "robocasa/PickPlaceCounterToCabinet")
        self.declare_parameter("seed", 0)
        self.declare_parameter("camera_width", 256)
        self.declare_parameter("camera_height", 256)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("auto_reset", True)
        self.declare_parameter("zero_action_on_start", True)
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 8766)
        self.declare_parameter("request_timeout_sec", 30.0)

        self.env_id = str(self.get_parameter("env_id").value)
        self.seed = int(self.get_parameter("seed").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.auto_reset = bool(self.get_parameter("auto_reset").value)
        self.zero_action_on_start = bool(self.get_parameter("zero_action_on_start").value)
        self.server_host = str(self.get_parameter("server_host").value)
        self.server_port = int(self.get_parameter("server_port").value)
        self.request_timeout_sec = float(self.get_parameter("request_timeout_sec").value)

        self.client = JsonLineClient(
            host=self.server_host,
            port=self.server_port,
            timeout_sec=self.request_timeout_sec,
        )
        self.get_logger().info(
            f"Using RoboCasa sim server at {self.server_host}:{self.server_port}"
        )
        self.obs: dict[str, Any] | None = None
        self.latest_action = zero_action_dict()
        self.last_reward = 0.0
        self.last_success = False
        self.last_done = False
        self.step_count = 0
        self.camera_publishers: dict[str, Any] = {}

        self.create_subscription(
            Float32MultiArray,
            "/nova/robocasa/action_cmd",
            self.action_callback,
            10,
        )
        self.state_pub = self.create_publisher(String, "/nova/robocasa/state", 10)
        self.reward_pub = self.create_publisher(Float32, "/nova/robocasa/reward", 10)
        self.success_pub = self.create_publisher(Bool, "/nova/robocasa/success", 10)

        self.create_service(Trigger, "/nova/robocasa/reset", self.reset_callback)
        self.create_service(Trigger, "/nova/robocasa/step_zero", self.step_zero_callback)

        if self.auto_reset:
            self._reset_env()

        period = 1.0 / max(self.publish_rate_hz, 0.1)
        self.timer = self.create_timer(period, self.timer_callback)

    # 重置仿真环境并发布初始观测
    def _reset_env(self) -> None:
        response = self._server_request(
            {
                "type": "reset",
                "env_id": self.env_id,
                "seed": self.seed,
                "camera_width": self.camera_width,
                "camera_height": self.camera_height,
            }
        )
        self.obs = decode_observation(response["obs"])
        info = response.get("info", {})

        self.latest_action = zero_action_dict()
        self.last_reward = 0.0
        self.last_success = bool(info.get("success", False))
        self.last_done = False
        self.step_count = 0
        self._ensure_camera_publishers(self.obs)
        self._publish_observation()
        self.get_logger().info("RoboCasa env reset")

    def _server_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None
        return self.client.request(payload)

    def _ensure_camera_publishers(self, obs: dict[str, Any]) -> None:
        for key, value in obs.items():
            if not key.startswith("video.") or not isinstance(value, np.ndarray):
                continue
            if value.ndim != 3 or value.shape[-1] != 3:
                continue
            topic_suffix = key.removeprefix("video.").replace(".", "_")
            topic = f"/nova/robocasa/cameras/{topic_suffix}/image_raw"
            if key not in self.camera_publishers:
                self.camera_publishers[key] = self.create_publisher(Image, topic, 10)
                self.get_logger().info(f"Camera publisher: {topic}")

    def action_callback(self, msg: Float32MultiArray) -> None:
        try:
            self.latest_action = action_vector_to_dict(np.asarray(msg.data, dtype=np.float32))
        except Exception as exc:
            self.get_logger().error(f"Invalid action_cmd: {exc}")

    def reset_callback(self, request: Trigger.Request, response: Trigger.Response):
        del request
        try:
            self._reset_env()
            response.success = True
            response.message = "reset complete"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    # 给零动作，让仿真空推一帧
    def step_zero_callback(self, request: Trigger.Request, response: Trigger.Response):
        del request
        try:
            self.latest_action = zero_action_dict()
            self._step_once()
            response.success = True
            response.message = "zero action step complete"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    # 定时器驱动:每次调用向 sim server 发 step,推进仿真一帧
    def timer_callback(self) -> None:
        try:
            if self.zero_action_on_start and self.step_count == 0:
                self.latest_action = zero_action_dict()
            self._step_once()
        except Exception:
            self.get_logger().error(traceback.format_exc())

    # 单步推进仿真:发送当前动作,更新 reward/success/done 并发布观测
    def _step_once(self) -> None:
        response = self._server_request(
            {"type": "step", "action": self.latest_action}
        )
        self.obs = decode_observation(response["obs"])
        reward = response.get("reward", 0.0)
        done = response.get("terminated", False)
        truncated = response.get("truncated", False)
        info = response.get("info", {})

        self.last_reward = float(reward)
        self.last_success = bool(info.get("success", False))
        self.last_done = bool(done or truncated)
        self.step_count += 1
        self._ensure_camera_publishers(self.obs)
        self._publish_observation()

    # 发布 state/reward/success 及所有相机话题
    def _publish_observation(self) -> None:
        if self.obs is None:
            return

        stamp = self.get_clock().now().to_msg()
        state_payload = {
            "env_id": self.env_id,
            "step_count": self.step_count,
            "reward": self.last_reward,
            "success": self.last_success,
            "done": self.last_done,
            "instruction": self.obs.get("annotation.human.task_description", ""),
            "state": {
                key: summarize_value(value)
                for key, value in self.obs.items()
                if key.startswith("state.")
            },
        }
        state_msg = String()
        state_msg.data = json.dumps(state_payload, ensure_ascii=False)
        self.state_pub.publish(state_msg)

        reward_msg = Float32()
        reward_msg.data = self.last_reward
        self.reward_pub.publish(reward_msg)

        success_msg = Bool()
        success_msg.data = self.last_success
        self.success_pub.publish(success_msg)

        for key, publisher in self.camera_publishers.items():
            image = self.obs.get(key)
            if not isinstance(image, np.ndarray):
                continue
            image_msg = self._numpy_rgb_to_image_msg(image)
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = key.removeprefix("video.").replace(".", "_")
            publisher.publish(image_msg)

    def _numpy_rgb_to_image_msg(self, image: np.ndarray) -> Image:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)
        msg = Image()
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(image.shape[1] * 3)
        msg.data = image.tobytes()
        return msg

    def destroy_node(self) -> bool:
        if self.client is not None:
            self.client.close()
        return super().destroy_node()


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
