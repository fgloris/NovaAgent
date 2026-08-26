# 统一仿真桥节点基类。
# 订阅 /nova/env/action_cmd,发布 /nova/env/obs|reward|success 与 /nova/env/camera/*/image_raw,
# 提供 /nova/env/info service 做自发现。子类只填 sim 特定细节。
import json
import os
import traceback
from typing import Any

os.environ.setdefault("ROS_LOG_DIR", "/tmp/novaagent_ros_logs")
os.makedirs(os.environ["ROS_LOG_DIR"], exist_ok=True)

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from std_srvs.srv import Trigger

from nova_common.jsonline import JsonLineClient
from nova_common.obs_codec import summarize_value
from nova_interfaces.srv import EnvInfo

# 相机图帧量大且实时,用 best_effort 防背压;obs/reward/success/action 走 reliable
_CAMERA_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class EnvBridgeBase(Node):
    def __init__(self, node_name: str, action_dim_default: int = 7) -> None:
        super().__init__(node_name)
        self.declare_parameter("camera_width", 256)
        self.declare_parameter("camera_height", 256)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("auto_reset", True)
        self.declare_parameter("zero_action_on_start", True)
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 8767)
        self.declare_parameter("request_timeout_sec", 30.0)

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
        self.get_logger().info(f"Using sim server at {self.server_host}:{self.server_port}")

        self.obs: dict[str, Any] | None = None
        self.action_spec: dict[str, Any] = {"dim": action_dim_default, "meaning": []}
        self.obs_spec: dict[str, Any] = {"state": {}, "cameras": {}}
        self.sim_info: dict[str, Any] = {}
        self.latest_action: Any = None
        self.last_reward = 0.0
        self.last_success = False
        self.last_done = False
        self.step_count = 0
        self.camera_publishers: dict[str, Any] = {}

        self._declare_custom_params()

        self.create_subscription(
            Float32MultiArray, "/nova/env/action_cmd", self.action_callback, 10
        )
        self.obs_pub = self.create_publisher(String, "/nova/env/obs", 10)
        self.reward_pub = self.create_publisher(Float32, "/nova/env/reward", 10)
        self.success_pub = self.create_publisher(Bool, "/nova/env/success", 10)

        self.create_service(EnvInfo, "/nova/env/info", self.info_callback)
        self.create_service(Trigger, "/nova/env/reset", self.reset_callback)
        self.create_service(Trigger, "/nova/env/step_zero", self.step_zero_callback)

        if self.auto_reset:
            self._reset_env()

        period = 1.0 / max(self.publish_rate_hz, 0.1)
        self.timer = self.create_timer(period, self.timer_callback)

    # ---------- 子类必须实现 ----------
    def _declare_custom_params(self) -> None:
        # 声明并读取子类专属参数;必须在 _reset_env 之前调用
        raise NotImplementedError

    def _build_reset_request(self) -> dict[str, Any]:
        raise NotImplementedError

    def action_vector_to_native(self, values: np.ndarray):
        # 规范动作向量 -> 发送给 sim server 的 request["action"];返回 None 表示丢弃
        raise NotImplementedError

    # ---------- 可覆写 ----------
    def _extra_info(self) -> dict[str, Any]:
        return dict(self.sim_info)

    # ---------- 公共逻辑 ----------
    def _zero_action(self):
        return self.action_vector_to_native(np.zeros(self.action_spec["dim"], dtype=np.float32))

    def _absorb_response(self, response: dict[str, Any]) -> None:
        # 帧协议已把 obs/info 里的 numpy 数组还原为 ndarray,直接用
        self.obs = response["obs"]
        self.action_spec = response.get("action_spec") or self.action_spec
        self.obs_spec = response.get("obs_spec") or self.obs_spec
        self.sim_info = response.get("sim_info") or self.sim_info

    def _reset_env(self) -> None:
        response = self.client.request(self._build_reset_request())
        self._absorb_response(response)
        self.latest_action = self._zero_action()
        self.last_reward = 0.0
        self.last_success = bool(response.get("info", {}).get("success", False))
        self.last_done = False
        self.step_count = 0
        self._ensure_camera_publishers(self.obs_spec.get("cameras", {}))
        self._publish_observation()
        self.get_logger().info(f"env reset ({json.dumps(self._extra_info(), ensure_ascii=False)})")

    def _step_once(self) -> None:
        if self.latest_action is None:
            self.latest_action = self._zero_action()
        response = self.client.request({"type": "step", "action": self.latest_action})
        self._absorb_response(response)
        reward = response.get("reward", 0.0)
        done = response.get("terminated", False)
        truncated = response.get("truncated", False)
        info = response.get("info", {})

        self.last_reward = float(reward)
        self.last_success = bool(info.get("success", False))
        self.last_done = bool(done or truncated)
        self.step_count += 1
        self._ensure_camera_publishers(self.obs_spec.get("cameras", {}))
        self._publish_observation()

    def action_callback(self, msg: Float32MultiArray) -> None:
        try:
            values = np.asarray(msg.data, dtype=np.float32)
            dim = int(self.action_spec["dim"])
            if values.size != dim:
                raise ValueError(f"expected {dim} action values, got {values.size}")
            native = self.action_vector_to_native(values)
            if native is None:
                return
            self.latest_action = native
        except Exception as exc:
            self.get_logger().error(f"Invalid action_cmd: {exc}")

    def info_callback(self, request, response):
        del request
        try:
            payload = dict(self._extra_info())
            payload["action_spec"] = self.action_spec
            payload["obs_spec"] = self.obs_spec
            payload["instruction"] = (self.obs or {}).get("state.instruction", "")
            response.success = True
            response.message = ""
            response.spec_json = json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            response.spec_json = ""
        return response

    def reset_callback(self, request, response):
        del request
        try:
            self._reset_env()
            response.success = True
            response.message = "reset complete"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def step_zero_callback(self, request, response):
        del request
        try:
            self.latest_action = self._zero_action()
            self._step_once()
            response.success = True
            response.message = "zero action step complete"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def timer_callback(self) -> None:
        try:
            if self.zero_action_on_start and self.step_count == 0:
                self.latest_action = self._zero_action()
            self._step_once()
        except Exception:
            self.get_logger().error(traceback.format_exc())

    def _ensure_camera_publishers(self, cameras: dict[str, Any]) -> None:
        for name in cameras:
            if name in self.camera_publishers:
                continue
            topic = f"/nova/env/camera/{name}/image_raw"
            self.camera_publishers[name] = self.create_publisher(Image, topic, _CAMERA_QOS)
            self.get_logger().info(f"Camera publisher: {topic}")

    def _publish_observation(self) -> None:
        if self.obs is None:
            return

        stamp = self.get_clock().now().to_msg()
        payload = dict(self._extra_info())
        payload.update(
            {
                "step_count": self.step_count,
                "reward": self.last_reward,
                "success": self.last_success,
                "done": self.last_done,
                "instruction": self.obs.get("state.instruction", ""),
                "action_spec": self.action_spec,
                "state": {
                    key[len("state."):]: summarize_value(value)
                    for key, value in self.obs.items()
                    if key.startswith("state.")
                },
                "cameras": {
                    name: summarize_value(self.obs.get(f"video.{name}"))
                    for name in self.obs_spec.get("cameras", {})
                },
            }
        )
        obs_msg = String()
        obs_msg.data = json.dumps(payload, ensure_ascii=False)
        self.obs_pub.publish(obs_msg)

        reward_msg = Float32()
        reward_msg.data = self.last_reward
        self.reward_pub.publish(reward_msg)

        success_msg = Bool()
        success_msg.data = self.last_success
        self.success_pub.publish(success_msg)

        for name, publisher in self.camera_publishers.items():
            image = self.obs.get(f"video.{name}")
            if not isinstance(image, np.ndarray):
                continue
            image_msg = self._numpy_rgb_to_image_msg(image)
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = name
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
