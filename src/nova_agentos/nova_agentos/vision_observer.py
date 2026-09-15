#!/usr/bin/env python3
# AgentOS 视觉观测缓存:订阅 /nova/env/obs 与当前环境声明的多路相机帧,
# 在规划回合前生成 OpenAI 兼容的多模态 user message。
from __future__ import annotations

import base64
import io
import json
import threading
import time
from typing import Any

import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from nova_interfaces.srv import EnvInfo

_CAM_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class VisionObserver:
    def __init__(
        self,
        node: Node,
        env_ns: str = "/nova/env",
        camera_names: list[str] | None = None,
        max_images: int = 6,
        max_image_size: int = 768,
        jpeg_quality: int = 80,
    ) -> None:
        self.node = node
        self.env_ns = env_ns.rstrip("/")
        initial_cameras = list(camera_names or [])
        self.camera_names: list[str] = []
        self.max_images = max(1, int(max_images))
        self.max_image_size = max(64, int(max_image_size))
        self.jpeg_quality = min(95, max(20, int(jpeg_quality)))

        self._lock = threading.Lock()
        self._obs: dict[str, Any] = {}
        self._frames: dict[str, tuple[np.ndarray, float]] = {}
        self._robot_context: dict | None = None
        self._robot_context_hash = ""
        self._subs: dict[str, Any] = {}
        self._info_cg = MutuallyExclusiveCallbackGroup()
        self._info_client = node.create_client(
            EnvInfo, f"{self.env_ns}/info", callback_group=self._info_cg
        )

        node.create_subscription(String, f"{self.env_ns}/obs", self._obs_cb, 10)
        for cam in initial_cameras:
            self._ensure_camera_sub(cam)
        node.create_timer(2.0, self.refresh_cameras)

    def refresh_cameras(self) -> None:
        if not self._info_client.service_is_ready():
            return
        future = self._info_client.call_async(EnvInfo.Request())
        future.add_done_callback(self._info_done)

    def snapshot_message(self) -> dict | None:
        with self._lock:
            obs = dict(self._obs)
            frames = dict(self._frames)
            cameras = list(self.camera_names)
        if not obs and not frames:
            return None

        content: list[dict] = [{"type": "text", "text": self._snapshot_text(obs, frames, cameras)}]
        added = 0
        for cam in cameras:
            item = frames.get(cam)
            if item is None:
                continue
            frame, _stamp = item
            content.append(
                {
                    "type": "text",
                    "text": f"camera: {cam}; timestamp: {time.time():.6f}; shape: {list(frame.shape)}",
                }
            )
            content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(frame)}})
            added += 1
            if added >= self.max_images:
                break
        return {"role": "user", "content": content}

    def get_robot_context(self) -> dict | None:
        with self._lock:
            return dict(self._robot_context) if self._robot_context is not None else None

    def _obs_cb(self, msg: String) -> None:
        try:
            obs = json.loads(msg.data) if msg.data else {}
        except Exception:
            obs = {"raw": msg.data[:4000]}
        camera_names = sorted((obs.get("cameras") or {}).keys()) if isinstance(obs, dict) else []
        with self._lock:
            self._obs = obs
        for cam in camera_names:
            self._ensure_camera_sub(cam)

    def _info_done(self, future) -> None:
        try:
            response = future.result()
            if not response or not response.success:
                return
            info = json.loads(response.spec_json or "{}")
            robot = info.get("robot") or {}
            if robot.get("context_schema") == "robot_context_v1" and robot.get("context_markdown"):
                new_hash = str(robot.get("description_sha256", ""))
                with self._lock:
                    changed = new_hash != self._robot_context_hash
                    self._robot_context = dict(robot)
                    self._robot_context_hash = new_hash
                if changed:
                    self.node.get_logger().info(f"loaded robot context: {robot.get('robot_type')} sha256={new_hash}")
            camera_names = sorted((info.get("obs_spec") or {}).get("cameras", {}).keys())
            if not camera_names:
                camera_names = sorted((info.get("cameras") or {}).keys())
        except Exception as exc:
            self.node.get_logger().warn(f"刷新环境相机失败: {exc}")
            return
        for cam in camera_names:
            self._ensure_camera_sub(cam)

    def _ensure_camera_sub(self, cam: str) -> None:
        cam = str(cam).strip()
        if not cam:
            return
        with self._lock:
            if cam in self._subs:
                return
            self.camera_names.append(cam)
        topic = f"{self.env_ns}/camera/{cam}/image_raw"
        sub = self.node.create_subscription(Image, topic, self._make_cam_cb(cam), _CAM_QOS)
        with self._lock:
            self._subs[cam] = sub
        self.node.get_logger().info(f"AgentOS VLM 订阅相机: {topic}")

    def _make_cam_cb(self, cam: str):
        def cb(msg: Image) -> None:
            try:
                frame = self._image_msg_to_numpy(msg)
            except Exception as exc:
                self.node.get_logger().warn(f"相机 {cam} 图像解码失败: {exc}")
                return
            with self._lock:
                self._frames[cam] = (frame, time.time())

        return cb

    @staticmethod
    def _image_msg_to_numpy(msg: Image) -> np.ndarray:
        if msg.encoding not in ("rgb8", "bgr8"):
            raise ValueError(f"暂不支持 encoding={msg.encoding!r}, 需要 rgb8/bgr8")
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == "bgr8":
            frame = frame[:, :, ::-1]
        return np.ascontiguousarray(frame)

    def _image_to_data_url(self, image: np.ndarray) -> str:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        h, w = image.shape[:2]
        scale = min(1.0, self.max_image_size / max(h, w))
        if scale < 1.0:
            from PIL import Image as PILImage

            image = np.asarray(
                PILImage.fromarray(image, mode="RGB").resize(
                    (int(round(w * scale)), int(round(h * scale)))
                )
            )
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.fromarray(image, mode="RGB").save(buf, format="JPEG", quality=self.jpeg_quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _snapshot_text(obs: dict[str, Any], frames: dict[str, tuple[np.ndarray, float]], cameras: list[str]) -> str:
        slim = dict(obs)
        if "state" in slim:
            slim["state"] = _trim_jsonish(slim["state"], max_chars=2500)
        camera_lines = []
        now = time.time()
        for cam in cameras:
            item = frames.get(cam)
            if item is None:
                camera_lines.append(f"- {cam}: no frame yet")
                continue
            frame, stamp = item
            camera_lines.append(f"- {cam}: shape={list(frame.shape)}, age_sec={now - stamp:.2f}")
        return (
            "# 当前环境视觉观测\n"
            "你是 VLM 规划器。决策下一步工具调用时必须结合这些最新相机画面、环境摘要和用户目标。\n"
            "图像按 camera 名称逐张给出。\n\n"
            f"obs_summary:\n{json.dumps(slim, ensure_ascii=False)[:6000]}\n\n"
            "camera_status:\n" + "\n".join(camera_lines)
        )


def _trim_jsonish(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return value
    return text[:max_chars] + "...(truncated)"
