#!/usr/bin/env python3
"""nova_perception_executor:感知类 MCP executor。

常驻订阅 /nova/env/camera/* 滚动缓存最新帧,提供:
  - reproject_pixels:多视图像素位置 -> 三角化 3D 点 + 重投影误差;
  - visualize_frame / visualize_point / visualize_segment / visualize_ray:
    在图像上叠加 3D 几何标注(半透明圆柱+圆锥箭头,画家算法渲染);
  - visualize_pixels:按像素坐标在图像上画圈;
  - visualize_grid:在图像上叠加定位网格。
绘制结果发到绘制话题,并通过 result_json.images 注入 VLM。
"""
import json
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from nova_common import image_codec
from nova_interfaces.action import MCPExecute
from nova_interfaces.msg import ExecutorHeartbeat, ToolDescriptor
from nova_interfaces.srv import EnvInfo

from nova_perception_executor import vision_geometry as vg

HEARTBEAT_TOPIC = "/nova/executors/heartbeat"
_CAM_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

DEFAULT_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

_IMAGE_DESC = "源图:相机名(用该相机最新帧)或之前绘制工具返回的 image_id(可在其基础上继续叠加)"
_RADIUS_DESC = "箭头/杆半径(m,base 系投影前固定值,近大远小),默认取节点参数"
_COLOR_DESC = "颜色:名称(red/green/blue/yellow/cyan/magenta/orange/white)或 [r,g,b](0-255)"
# 渲染风格(alpha/描边/圆周分段/字号/注入尺寸)只走节点参数/yaml,不暴露给模型
# 像素坐标一律用 VLM 看到的显示分辨率;工具内部自动换算到原始分辨率。

REPROJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {
            "type": "object",
            "description": "各相机的目标像素坐标 {相机名: [u, v]}(显示分辨率,至少 2 个相机)",
            "additionalProperties": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        },
        "image_size": {
            "type": "object",
            "description": "可选:覆盖各相机像素坐标所属的显示分辨率 {相机名: [w, h]}",
            "additionalProperties": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        },
    },
    "required": ["points"],
}

VISUALIZE_PIXELS_SCHEMA = {
    "type": "object",
    "properties": {
        "image": {"type": "string", "description": _IMAGE_DESC},
        "points": {
            "description": "要画圈的像素坐标(显示分辨率):[[u,v],...] 或 {label:[u,v]}",
            "oneOf": [
                {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
                {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
            ],
        },
        "radius_px": {"type": "number", "description": "圆圈半径(显示像素),默认取节点参数"},
        "color": {"description": _COLOR_DESC},
        "label": {"type": "string", "description": "可选文本标签"},
        "image_size": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                       "description": "可选:覆盖像素坐标所属的显示分辨率 [w, h]"},
    },
    "required": ["image", "points"],
}

VISUALIZE_GRID_SCHEMA = {
    "type": "object",
    "properties": {
        "image": {"type": "string", "description": _IMAGE_DESC},
        "grid_size": {"type": "integer", "description": "网格划分大小(默认取节点参数,默认 8 即 8x8)"},
    },
    "required": ["image"],
}

VISUALIZE_FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "image": {"type": "string", "description": _IMAGE_DESC},
        "origin": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                   "description": "坐标系原点(base 系 x,y,z,m)"},
        "orientation": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                        "description": "坐标系姿态(base 系 xyzw 四元数)"},
        "axis_length": {"type": "number", "description": "坐标轴长度(m),默认取节点参数"},
        "radius": {"type": "number", "description": _RADIUS_DESC},
        "labels": {"type": "boolean", "description": "是否在轴端标注 x/y/z,默认 true"},
    },
    "required": ["image", "origin", "orientation"],
}

VISUALIZE_POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "image": {"type": "string", "description": _IMAGE_DESC},
        "point": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                  "description": "3D 点(base 系 x,y,z,m)"},
        "radius": {"type": "number", "description": "小球半径(m),默认取节点参数"},
        "color": {"description": _COLOR_DESC},
        "label": {"type": "string", "description": "可选文本标签"},
    },
    "required": ["image", "point"],
}

VISUALIZE_SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "image": {"type": "string", "description": _IMAGE_DESC},
        "points": {
            "type": "array", "minItems": 2, "maxItems": 2,
            "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "description": "线段两个端点(base 系)",
        },
        "radius": {"type": "number", "description": _RADIUS_DESC},
        "show_distance": {"type": "boolean", "description": "是否在中点标注 3D 距离,默认 true"},
        "color": {"description": _COLOR_DESC},
        "label": {"type": "string", "description": "可选文本标签"},
    },
    "required": ["image", "points"],
}

VISUALIZE_RAY_SCHEMA = {
    "type": "object",
    "properties": {
        "image": {"type": "string", "description": _IMAGE_DESC},
        "origin": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                   "description": "射线起点(base 系 x,y,z,m)"},
        "orientation": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                        "description": "射线方向:局部 +z 指向该方向的 xyzw 四元数"},
        "length": {"type": "number", "description": "箭头长度(m),默认 0.2"},
        "radius": {"type": "number", "description": _RADIUS_DESC},
        "color": {"description": _COLOR_DESC},
        "label": {"type": "string", "description": "可选文本标签"},
    },
    "required": ["image", "origin", "orientation"],
}

TOOLS = {
    "reproject_pixels": (
        "多视图像素重投影:输入各相机上目标的像素坐标(显示分辨率,至少 2 个相机),"
        "用相机投影矩阵 DLT 三角化出 3D 基座系坐标,并返回各视图的重投影像素与误差、是否收敛。",
        REPROJECT_SCHEMA,
    ),
    "visualize_frame": (
        "在图像上叠加一个 3D 坐标系(base 系):origin + orientation(xyzw),"
        "沿局部 x/y/z 画红/绿/蓝半透明箭头。用于观察夹爪/物体等坐标系的朝向。",
        VISUALIZE_FRAME_SCHEMA,
    ),
    "visualize_point": (
        "在图像上叠加一个 3D 点(base 系),画成半透明小球并可选标签。",
        VISUALIZE_POINT_SCHEMA,
    ),
    "visualize_segment": (
        "在图像上叠加一条 3D 线段(base 系两点),画成半透明圆柱,可标注 3D 距离。",
        VISUALIZE_SEGMENT_SCHEMA,
    ),
    "visualize_ray": (
        "在图像上叠加一条 3D 射线/方向(base 系 origin + orientation),画成半透明箭头。",
        VISUALIZE_RAY_SCHEMA,
    ),
    "visualize_pixels": (
        "按像素坐标在图像上画空心圆圈(显示分辨率),用于核对物体像素位置;可带文本标签。",
        VISUALIZE_PIXELS_SCHEMA,
    ),
    "visualize_grid": (
        "在图像上叠加 grid_size×grid_size 网格并标注 行-列 编号,用于粗定位。",
        VISUALIZE_GRID_SCHEMA,
    ),
}

_COLORS = {
    "red": (255, 70, 70),
    "green": (70, 200, 70),
    "blue": (70, 120, 255),
    "yellow": (255, 220, 0),
    "cyan": (0, 220, 220),
    "magenta": (230, 80, 230),
    "orange": (255, 150, 40),
    "white": (255, 255, 255),
}


def _image_to_numpy(msg: Image) -> np.ndarray:
    """把 ROS Image(rgb8)消息转成 HxWx3 的 uint8 numpy 数组。"""
    return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))


def _vec3(value, name: str) -> np.ndarray:
    """校验并返回 3 维向量。"""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 必须是 3 个有限数")
    return arr


def _vec4(value, name: str) -> list[float]:
    """校验并返回 4 元 xyzw 四元数。"""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 4 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 必须是 4 个有限数(xyzw)")
    return arr.tolist()


def _parse_color(value, default):
    """解析颜色:名称字符串或 [r,g,b];无法识别时返回 default。"""
    if value is None:
        return default
    if isinstance(value, str):
        return _COLORS.get(value.strip().lower(), default)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 3:
        return tuple(int(np.clip(c, 0, 255)) for c in arr)
    return default


class PerceptionExecutorNode(Node):
    """感知类 MCP executor:3D 定位 + 图像几何可视化。"""

    def __init__(self) -> None:
        super().__init__("nova_perception_executor")
        self.declare_parameter("camera_names", DEFAULT_CAMERAS)
        self.declare_parameter("image_topics", [], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("env_ns", "/nova/env")
        self.declare_parameter("heartbeat_rate_hz", 1.0)
        self.declare_parameter("grid_size", 8)
        self.declare_parameter("display_max_size", image_codec.DEFAULT_MAX_IMAGE_SIZE)
        # 绘制相关默认参数
        self.declare_parameter("draw_topic", "/nova/perception/draw")
        self.declare_parameter("draw_alpha", 0.45)
        self.declare_parameter("draw_supersample", 2)
        self.declare_parameter("draw_outline", True)
        self.declare_parameter("draw_outline_width", 2.0)
        self.declare_parameter("draw_outline_color", [0, 0, 0])
        self.declare_parameter("label_font_scale", 10.0)
        self.declare_parameter("inject_image_max_size", 0)
        self.declare_parameter("pixel_radius", 12.0)
        self.declare_parameter("arrow_radius", 0.006)
        self.declare_parameter("point_radius", 0.02)
        self.declare_parameter("axis_length", 0.1)
        self.declare_parameter("segments", 16)
        self.declare_parameter("image_cache_size", 20)

        rate = float(self.get_parameter("heartbeat_rate_hz").value)
        self.camera_names = list(self.get_parameter("camera_names").value)
        self.env_ns = str(self.get_parameter("env_ns").value).rstrip("/")
        topics = list(self.get_parameter("image_topics").value or [])
        if topics:
            if len(topics) != len(self.camera_names):
                raise RuntimeError(
                    f"image_topics({len(topics)}) 与 camera_names({len(self.camera_names)}) 数量不一致"
                )
            self.image_topics = dict(zip(self.camera_names, topics))
        else:
            self.image_topics = {
                cam: f"{self.env_ns}/camera/{cam}/image_raw" for cam in self.camera_names
            }
        self.env_info_srv = f"{self.env_ns}/info"
        self._grid_size = int(self.get_parameter("grid_size").value)
        self._display_max_size = int(self.get_parameter("display_max_size").value)

        self._draw_topic = str(self.get_parameter("draw_topic").value)
        self._alpha = float(self.get_parameter("draw_alpha").value)
        self._supersample = int(self.get_parameter("draw_supersample").value)
        self._outline = bool(self.get_parameter("draw_outline").value)
        self._outline_width = float(self.get_parameter("draw_outline_width").value)
        self._outline_color = _parse_color(self.get_parameter("draw_outline_color").value, (0, 0, 0))
        self._label_font_scale = float(self.get_parameter("label_font_scale").value)
        self._inject_image_max_size = int(self.get_parameter("inject_image_max_size").value)
        self._pixel_radius = float(self.get_parameter("pixel_radius").value)
        self._arrow_radius = float(self.get_parameter("arrow_radius").value)
        self._point_radius = float(self.get_parameter("point_radius").value)
        self._axis_length = float(self.get_parameter("axis_length").value)
        self._segments = int(self.get_parameter("segments").value)
        self._image_cache_size = max(1, int(self.get_parameter("image_cache_size").value))

        self._frames: dict[str, np.ndarray] = {}
        self._camera_subs: set[str] = set()
        for cam, topic in self.image_topics.items():
            self._camera_subs.add(cam)
            self.create_subscription(Image, topic, self._make_cam_cb(cam), _CAM_QOS)
        # 自动发现 /nova/env/obs 里声明的全部相机(含腕部相机等),便于在任意相机上绘制
        self.create_subscription(String, f"{self.env_ns}/obs", self._obs_cb, 10)

        self._image_cache: dict[str, dict] = {}
        self._image_counter = 0

        self._info_cg = MutuallyExclusiveCallbackGroup()
        self._info_client = self.create_client(
            EnvInfo, self.env_info_srv, callback_group=self._info_cg
        )
        self._draw_pub = self.create_publisher(Image, self._draw_topic, 5)

        action_cg = ReentrantCallbackGroup()
        self._servers = []
        for name in TOOLS:
            self._servers.append(
                ActionServer(
                    self,
                    MCPExecute,
                    f"/{self.get_name()}/{name}/execute",
                    self._execute_cb(name),
                    cancel_callback=lambda _goal: CancelResponse.ACCEPT,
                    callback_group=action_cg,
                )
            )
        self._heartbeat_pub = self.create_publisher(ExecutorHeartbeat, HEARTBEAT_TOPIC, 1)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_heartbeat)
        self._publish_heartbeat()
        self.get_logger().info(
            f"感知 executor 就绪,相机={self.camera_names}, 工具={list(TOOLS)}, "
            f"绘制话题={self._draw_topic}"
        )

    def _publish_heartbeat(self) -> None:
        """周期性发布心跳,声明本 executor 提供的全部工具。"""
        hb = ExecutorHeartbeat()
        hb.executor_name = self.get_name()
        for name, (description, schema) in TOOLS.items():
            tool = ToolDescriptor()
            tool.name = name
            tool.description = description
            tool.params_schema_json = json.dumps(schema, ensure_ascii=False)
            tool.action_server_name = f"/{self.get_name()}/{name}/execute"
            hb.tools.append(tool)
        self._heartbeat_pub.publish(hb)

    def _make_cam_cb(self, cam: str):
        """为指定相机生成订阅回调,滚动缓存最新一帧。"""
        def cb(msg):
            self._frames[cam] = _image_to_numpy(msg)

        return cb

    def _obs_cb(self, msg: String) -> None:
        """从 /nova/env/obs 的 cameras 键发现新相机并补订阅。"""
        try:
            doc = json.loads(msg.data) if msg.data else {}
        except Exception:
            return
        cameras = doc.get("cameras")
        if isinstance(cameras, dict):
            for name in cameras:
                self._ensure_camera_sub(str(name))

    def _ensure_camera_sub(self, cam: str) -> None:
        """为尚未订阅的相机创建 {env_ns}/camera/{cam}/image_raw 订阅。"""
        cam = str(cam).strip()
        if not cam or cam in self._camera_subs:
            return
        self._camera_subs.add(cam)
        topic = f"{self.env_ns}/camera/{cam}/image_raw"
        self.create_subscription(Image, topic, self._make_cam_cb(cam), _CAM_QOS)
        self.get_logger().info(f"新增相机订阅: {topic}")

    # ---------- /nova/env/info ----------

    def _fetch_projections(self) -> dict:
        """同步调用 /nova/env/info,取出各相机的 intrinsics/projection 字典。"""
        if not self._info_client.service_is_ready():
            if not self._info_client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"env info service {self.env_info_srv} 不可用")
        future = self._info_client.call_async(EnvInfo.Request())
        deadline = time.time() + 5.0
        while rclpy.ok() and not future.done():
            if time.time() > deadline:
                raise RuntimeError("查询 env info 超时")
            time.sleep(0.02)
        response = future.result()
        if not response.success:
            raise RuntimeError(f"env info 失败: {response.message}")
        info = json.loads(response.spec_json)
        return info.get("cameras", {})

    @staticmethod
    def _match_projection(cam: str, projections: dict) -> str | None:
        """把请求的相机名映射到投影矩阵的键:先精确匹配,再按包含关系兜底。"""
        if cam in projections:
            return cam
        for key in projections:
            if cam in key or key in cam:
                return key
        return None

    def _camera_calibration(self, cam: str, projections: dict) -> tuple[np.ndarray, np.ndarray]:
        """取指定相机的 (intrinsics 3x3, projection 3x4)。"""
        key = self._match_projection(cam, projections)
        if key is None:
            raise RuntimeError(f"相机 {cam} 无投影矩阵(可用:{list(projections)})")
        entry = projections[key]
        if not isinstance(entry, dict) or "intrinsics" not in entry or "projection" not in entry:
            raise RuntimeError(f"相机 {cam} 投影矩阵格式不合法")
        return (
            np.asarray(entry["intrinsics"], dtype=float).reshape(3, 3),
            np.asarray(entry["projection"], dtype=float).reshape(3, 4),
        )

    # ---------- 工具分发 ----------

    def _execute_cb(self, name: str):
        """构造绑定工具名的 action 回调:解析参数、执行、返回结果。"""
        def callback(goal_handle):
            try:
                params = json.loads(goal_handle.request.params_json or "{}")
                result_json = self._handle_tool(name, params)
                result = MCPExecute.Result()
                result.success = True
                result.result_json = json.dumps(result_json, ensure_ascii=False)
                result.error = ""
                goal_handle.succeed()
                self.get_logger().info(f"{name} 完成: {result_json.get('image_id') or result_json.get('position')}")
                return result
            except Exception as exc:
                result = MCPExecute.Result()
                result.success = False
                result.result_json = ""
                result.error = str(exc)
                goal_handle.abort()
                self.get_logger().error(f"{name} 失败: {exc}")
                return result

        return callback

    def _handle_tool(self, name: str, params: dict) -> dict:
        """把工具名分发到对应实现。"""
        if name == "reproject_pixels":
            return self._reproject_pixels(params)
        if name == "visualize_frame":
            return self._draw_frame(params)
        if name == "visualize_point":
            return self._draw_point(params)
        if name == "visualize_segment":
            return self._draw_segment(params)
        if name == "visualize_ray":
            return self._draw_ray(params)
        if name == "visualize_pixels":
            return self._draw_pixels(params)
        if name == "visualize_grid":
            return self._draw_grid(params)
        raise ValueError(f"未知工具: {name}")

    # ---------- 图像源与缓存 ----------

    def _resolve_image(self, source: str):
        """把相机名或 image_id 解析为 (图像, 相机名, K, P, 源标识)。"""
        source = str(source)
        projections = self._fetch_projections()
        if not projections:
            raise RuntimeError("env info 未提供相机投影矩阵")
        if source in self._frames:
            cam = source
            image = self._frames[cam]
        elif source in self._camera_subs:
            # 已订阅但还没收到帧:短暂等待,避免节点刚启动时的竞态
            deadline = time.time() + 3.0
            while source not in self._frames and time.time() < deadline:
                time.sleep(0.05)
            if source not in self._frames:
                raise RuntimeError(f"相机 {source} 已订阅但尚未收到帧")
            cam = source
            image = self._frames[cam]
        elif source in self._image_cache:
            entry = self._image_cache[source]
            cam = entry["camera"]
            image = entry["image"]
        else:
            raise RuntimeError(
                f"未知图像源 {source!r};可用相机={list(self._frames)} 或 image_id={list(self._image_cache)}"
            )
        K, P = self._camera_calibration(cam, projections)
        return image, cam, K, P, source

    def _store_image(self, image: np.ndarray, camera: str, source: str) -> str:
        """缓存绘制结果并返回新的 image_id(LRU 上限)。"""
        self._image_counter += 1
        image_id = f"viz_{self._image_counter}"
        self._image_cache[image_id] = {"image": image, "camera": camera, "source": source}
        while len(self._image_cache) > self._image_cache_size:
            self._image_cache.pop(next(iter(self._image_cache)), None)
        return image_id

    def _render(self, image, K, P, verts, faces, colors, params: dict) -> np.ndarray:
        """按节点默认/工具参数渲染网格(alpha、描边可逐次覆盖)。"""
        return vg.rasterize_mesh(
            image, verts, faces, colors, K, P,
            alpha=float(params.get("alpha", self._alpha)),
            supersample=self._supersample,
            outline=bool(params.get("outline", self._outline)),
            outline_width=float(params.get("outline_width", self._outline_width)),
            outline_color=_parse_color(params.get("outline_color"), self._outline_color),
        )

    def _label_font_px(self, K, Rt, anchor, radius) -> int:
        """标签字号:只跟箭头半径有关(anchor 处半径的投影像素长度 × 系数)。"""
        anchor = np.asarray(anchor, dtype=float)
        px = vg.projected_length(K, Rt, anchor, anchor + np.array([radius, 0.0, 0.0]))
        return int(np.clip(px * self._label_font_scale, 12, 64))

    def _encode_for_vlm(self, image: np.ndarray, params: dict) -> str:
        """把绘制图编码为注入 VLM 的 data URL;默认不缩放(与输入一致),可用参数调最长边。"""
        max_size = int(params.get("inject_image_max_size", self._inject_image_max_size) or 0)
        if max_size <= 0:
            max_size = max(int(image.shape[0]), int(image.shape[1]))
        return vg.encode_image(image, max_size=max_size)

    @staticmethod
    def _visibility_warnings(K, Rt, labeled_points, width: int, height: int) -> list[str]:
        """检查各关键点是否在相机前方且落在画面内,返回告警列表。"""
        warnings = []
        for label, point in labeled_points:
            projected = vg.project_point_safe(K, Rt, point)
            if projected is None:
                warnings.append(f"{label} 在相机后方")
            elif not (0 <= projected[0] < width and 0 <= projected[1] < height):
                warnings.append(f"{label}超出画面")
        return warnings

    @staticmethod
    def _status(warnings: list[str]) -> str:
        """无告警返回 drew successfully,否则返回 warning 文本。"""
        return "drew successfully" if not warnings else "warning: " + "; ".join(warnings)

    def _finish_draw(self, image, camera, source, tool, status, params=None) -> dict:
        """缓存并发布绘制结果,返回给模型的文本状态与待注入图像。"""
        image_id = self._store_image(image, camera, source)
        frame_id = f"{source}|{image_id}"
        self._draw_pub.publish(self._numpy_to_image_msg(image, frame_id))
        return {
            "ok": True,
            "tool": tool,
            "status": status,
            "image_id": image_id,
            "source": source,
            "camera": camera,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "topic": self._draw_topic,
            "images": {image_id: self._encode_for_vlm(image, params or {})},
        }

    # ---------- 各可视化工具 ----------

    def _draw_frame(self, params: dict) -> dict:
        """绘制一个 base 系坐标系(x/y/z 三色箭头)。"""
        image, cam, K, P, source = self._resolve_image(params["image"])
        origin = _vec3(params.get("origin"), "origin")
        orientation = _vec4(params.get("orientation"), "orientation")
        axis_length = float(params.get("axis_length", self._axis_length))
        radius = float(params.get("radius", self._arrow_radius))
        segments = int(params.get("segments", self._segments))
        R = vg.quat_to_matrix_xyzw(orientation)
        Rt = vg.decompose_projection(K, P)
        tips = [origin + R @ np.eye(3)[:, idx] * axis_length for idx in range(3)]
        verts, faces, colors = vg.build_frame(origin, orientation, axis_length, radius, segments)
        out = self._render(image, K, P, verts, faces, colors, params)
        if params.get("labels", True):
            font_px = self._label_font_px(K, Rt, origin, radius)
            for name, color, tip in zip("xyz", ((255, 70, 70), (70, 200, 70), (70, 120, 255)), tips):
                projected = vg.project_point_safe(K, Rt, tip)
                if projected is not None:
                    out = vg.draw_label(out, projected[:2], name, color, font_px=font_px)
        points = [("origin", origin)] + [(f"{name} 轴", tip) for name, tip in zip("xyz", tips)]
        warnings = self._visibility_warnings(K, Rt, points, out.shape[1], out.shape[0])
        return self._finish_draw(out, cam, source, "visualize_frame", self._status(warnings), params)

    def _draw_point(self, params: dict) -> dict:
        """绘制一个 base 系 3D 点(小球)。"""
        image, cam, K, P, source = self._resolve_image(params["image"])
        point = _vec3(params.get("point"), "point")
        radius = float(params.get("radius", self._point_radius))
        color = _parse_color(params.get("color"), (255, 220, 0))
        Rt = vg.decompose_projection(K, P)
        verts, faces, colors = vg.build_sphere(point, radius, color, self._segments, max(4, self._segments // 2))
        out = self._render(image, K, P, verts, faces, colors, params)
        label = params.get("label")
        if label:
            projected = vg.project_point_safe(K, Rt, point)
            if projected is not None:
                out = vg.draw_label(out, projected[:2], str(label), color,
                                    font_px=self._label_font_px(K, Rt, point, radius))
        warnings = self._visibility_warnings(K, Rt, [("point", point)], out.shape[1], out.shape[0])
        return self._finish_draw(out, cam, source, "visualize_point", self._status(warnings), params)

    def _draw_segment(self, params: dict) -> dict:
        """绘制一条 base 系 3D 线段(圆柱),可标注 3D 距离。"""
        image, cam, K, P, source = self._resolve_image(params["image"])
        points = params.get("points")
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError("points 必须是两个 3D 点")
        p0 = _vec3(points[0], "points[0]")
        p1 = _vec3(points[1], "points[1]")
        radius = float(params.get("radius", self._arrow_radius))
        color = _parse_color(params.get("color"), (80, 180, 255))
        verts, faces, colors = vg.build_cylinder(p0, p1, radius, self._segments, color)
        out = self._render(image, K, P, verts, faces, colors, params)
        Rt = vg.decompose_projection(K, P)
        texts = []
        if params.get("show_distance", True):
            texts.append(f"{float(np.linalg.norm(p1 - p0)) * 100:.1f}cm")
        if params.get("label"):
            texts.append(str(params["label"]))
        if texts:
            mid_point = (p0 + p1) / 2.0
            mid = vg.project_point_safe(K, Rt, mid_point)
            if mid is not None:
                out = vg.draw_label(out, mid[:2], " ".join(texts), color,
                                    font_px=self._label_font_px(K, Rt, mid_point, radius))
        warnings = self._visibility_warnings(K, Rt, [("起点", p0), ("终点", p1)], out.shape[1], out.shape[0])
        return self._finish_draw(out, cam, source, "visualize_segment", self._status(warnings), params)

    def _draw_ray(self, params: dict) -> dict:
        """绘制一条 base 系 3D 射线(沿四元数局部 +z 的箭头)。"""
        image, cam, K, P, source = self._resolve_image(params["image"])
        origin = _vec3(params.get("origin"), "origin")
        orientation = _vec4(params.get("orientation"), "orientation")
        length = float(params.get("length", 0.2))
        radius = float(params.get("radius", self._arrow_radius))
        color = _parse_color(params.get("color"), (255, 120, 40))
        Rt = vg.decompose_projection(K, P)
        tip = origin + vg.quat_to_matrix_xyzw(orientation) @ np.array([0.0, 0.0, length])
        verts, faces, colors = vg.build_arrow(origin, orientation, length, radius,
                                              segments=self._segments, color=color)
        out = self._render(image, K, P, verts, faces, colors, params)
        label = params.get("label")
        if label:
            projected = vg.project_point_safe(K, Rt, tip)
            if projected is not None:
                out = vg.draw_label(out, projected[:2], str(label), color,
                                    font_px=self._label_font_px(K, Rt, origin, radius))
        warnings = self._visibility_warnings(K, Rt, [("origin", origin), ("射线尖端", tip)],
                                             out.shape[1], out.shape[0])
        return self._finish_draw(out, cam, source, "visualize_ray", self._status(warnings), params)

    # ---------- 像素工具 ----------

    def _display_wh(self, source: str, native_wh: tuple[int, int], override=None) -> tuple[int, int]:
        """返回像素坐标所属的显示分辨率:优先 override,否则按来源推断。"""
        if override:
            return int(override[0]), int(override[1])
        if source in self._image_cache:
            max_size = self._inject_image_max_size or max(native_wh)
        else:
            max_size = self._display_max_size
        return image_codec.display_size(native_wh[0], native_wh[1], max_size)

    def _reproject_pixels(self, params: dict) -> dict:
        """多视图像素 -> 三角化 3D 点 + 重投影误差;输入/输出都用显示分辨率。"""
        points = params.get("points")
        if not isinstance(points, dict) or len(points) < 2:
            raise ValueError("points 必须是至少 2 个相机的 {相机名:[u,v]}")
        projections = self._fetch_projections()
        if not projections:
            raise RuntimeError("env info 未提供相机投影矩阵")
        sizes = params.get("image_size") or {}

        cams = list(points.keys())
        native_pts, native_projs, display_sizes, native_sizes = [], [], {}, {}
        for cam in cams:
            if cam not in self._frames:
                raise RuntimeError(f"相机 {cam} 未收到帧")
            native_wh = (self._frames[cam].shape[1], self._frames[cam].shape[0])
            display_wh = self._display_wh(cam, native_wh, sizes.get(cam))
            _, P = self._camera_calibration(cam, projections)
            native_pts.append(image_codec.convert_points(list(points[cam]), display_wh, native_wh))
            native_projs.append(P)
            display_sizes[cam] = display_wh
            native_sizes[cam] = native_wh

        position = vg.triangulate(native_pts, native_projs)
        errors_native = vg.reprojection_errors(native_pts, native_projs, position)
        reprojected, errors = {}, {}
        for cam, P, err in zip(cams, native_projs, errors_native):
            native_wh = native_sizes[cam]
            display_wh = display_sizes[cam]
            uu, vv = vg.project_point(P, position)
            reprojected[cam] = image_codec.convert_points([uu, vv], native_wh, display_wh)
            errors[cam] = round(float(err) * (display_wh[0] / native_wh[0]), 3)
        mean_error = sum(errors.values()) / len(errors)
        return {
            "ok": True,
            "position": position,
            "reprojected": reprojected,
            "errors_px": errors,
            "mean_error_px": round(mean_error, 3),
            "display_size": {cam: list(display_sizes[cam]) for cam in cams},
            "native_size": {cam: list(native_sizes[cam]) for cam in cams},
        }

    def _draw_pixels(self, params: dict) -> dict:
        """按像素坐标在图像上画空心圆圈(输入显示分辨率,内部换算到原始分辨率)。"""
        image, cam, K, P, source = self._resolve_image(params["image"])
        native_wh = (image.shape[1], image.shape[0])
        display_wh = self._display_wh(source, native_wh, params.get("image_size"))
        raw = params.get("points")
        if isinstance(raw, dict):
            labeled = [(str(key), value) for key, value in raw.items()]
        elif isinstance(raw, list):
            labeled = [(None, value) for value in raw]
        else:
            raise ValueError("points 必须是 [[u,v],...] 或 {label:[u,v]}")
        radius_native = float(params.get("radius_px", self._pixel_radius)) * (native_wh[0] / display_wh[0])
        color = _parse_color(params.get("color"), (255, 80, 80))
        out = image
        warnings: list[str] = []
        for label, uv in labeled:
            if not isinstance(uv, (list, tuple)) or len(uv) != 2:
                raise ValueError("每个点必须是 [u, v]")
            if not (0 <= float(uv[0]) < display_wh[0] and 0 <= float(uv[1]) < display_wh[1]):
                warnings.append(f"点 {label or uv} 超出画面")
            native_px = image_codec.convert_points(list(uv), display_wh, native_wh)
            out = vg.draw_marker(out, native_px, color, radius=radius_native, label=label)
        return self._finish_draw(out, cam, source, "visualize_pixels", self._status(warnings), params)

    def _draw_grid(self, params: dict) -> dict:
        """在图像上叠加 grid_size×grid_size 网格并返回显示分辨率下的格子尺寸。"""
        image, cam, K, P, source = self._resolve_image(params["image"])
        native_wh = (image.shape[1], image.shape[0])
        display_wh = self._display_wh(source, native_wh, params.get("image_size"))
        grid_size = max(2, int(params.get("grid_size", self._grid_size)))
        out = vg.draw_grid(image, grid_size)
        result = self._finish_draw(out, cam, source, "visualize_grid", "drew successfully", params)
        result.update(
            {
                "grid_size": grid_size,
                "display_size": list(display_wh),
                "cell_px": [display_wh[0] / grid_size, display_wh[1] / grid_size],
            }
        )
        return result

    @staticmethod
    def _numpy_to_image_msg(image: np.ndarray, frame_id: str) -> Image:
        """numpy RGB 图 -> ROS Image 消息。"""
        image = np.ascontiguousarray(image.astype(np.uint8))
        msg = Image()
        msg.header.frame_id = frame_id
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(image.shape[1] * 3)
        msg.data = image.tobytes()
        return msg

    def destroy_node(self) -> bool:
        return super().destroy_node()


def main(args=None) -> int:
    """启动多线程 executor 并运行感知节点。"""
    rclpy.init(args=args)
    node = PerceptionExecutorNode()
    executor = MultiThreadedExecutor()
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
