#!/usr/bin/env python3
"""nova_perception_executor:感知类 MCP executor。

常驻订阅 /nova/env/camera/* 滚动缓存最新帧,提供:
  - locate_object_3d:多视图 VLM 3D 定位(base 系坐标);
  - visualize_frame / visualize_point / visualize_segment / visualize_ray:
    在图像上叠加 3D 几何标注(半透明圆柱+圆锥箭头,画家算法渲染),结果发到绘制话题。
"""
import json
import time
from typing import Any

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

from nova_common.llm_client import LLMClient
from nova_interfaces.action import MCPExecute
from nova_interfaces.msg import ExecutorHeartbeat, ToolDescriptor
from nova_interfaces.srv import EnvInfo

from nova_perception_executor import vision_geometry as vg
from nova_perception_executor.vlm_loop import VlmLocator

HEARTBEAT_TOPIC = "/nova/executors/heartbeat"
_CAM_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

DEFAULT_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right"]

LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string", "description": "要定位的物体描述,如 '红色杯子'"},
        "camera_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "参与三角化的相机名(默认取节点 camera_names 参数)",
        },
        "grid_size": {"type": "integer", "description": "网格划分大小,默认取节点参数(默认 8,即 8x8)"},
        "max_rounds": {"type": "integer", "description": "最大调整轮数,默认取节点参数(默认 5)"},
        "max_restarts": {"type": "integer", "description": "重投影不一致时的最大重来次数,默认取节点参数(默认 2)"},
        "max_reproj_error_px": {"type": "number", "description": "重投影误差阈值(像素),默认取节点参数(默认 25)"},
    },
    "required": ["object"],
}

_IMAGE_DESC = "源图:相机名(用该相机最新帧)或之前绘制工具返回的 image_id(可在其基础上继续叠加)"
_RADIUS_DESC = "箭头/杆半径(m,base 系投影前固定值,近大远小),默认取节点参数"
_COLOR_DESC = "颜色:名称(red/green/blue/yellow/cyan/magenta/orange/white)或 [r,g,b](0-255)"
_STYLE_PROPS = {
    "alpha": {"type": "number", "description": "箭头不透明度 0~1,默认取节点参数"},
    "outline": {"type": "boolean", "description": "是否给箭头描边,默认取节点参数"},
    "outline_width": {"type": "number", "description": "描边宽度(像素),默认取节点参数"},
    "outline_color": {"description": "描边颜色:名称或 [r,g,b],默认黑"},
}

VISUALIZE_FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        **_STYLE_PROPS,
        "image": {"type": "string", "description": _IMAGE_DESC},
        "origin": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                   "description": "坐标系原点(base 系 x,y,z,m)"},
        "orientation": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                        "description": "坐标系姿态(base 系 xyzw 四元数)"},
        "axis_length": {"type": "number", "description": "坐标轴长度(m),默认取节点参数"},
        "radius": {"type": "number", "description": _RADIUS_DESC},
        "segments": {"type": "integer", "description": "圆周分段数,默认取节点参数"},
        "labels": {"type": "boolean", "description": "是否在轴端标注 x/y/z,默认 true"},
    },
    "required": ["image", "origin", "orientation"],
}

VISUALIZE_POINT_SCHEMA = {
    "type": "object",
    "properties": {
        **_STYLE_PROPS,
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
        **_STYLE_PROPS,
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
        **_STYLE_PROPS,
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
    "locate_object_3d": (
        "多视图 VLM 3D 定位:读取 /nova/env/* 最新相机帧与投影矩阵,"
        "先让 VLM 在各图上用网格单元粗定位并 DLT 三角化,再把结果画回调试图,"
        "以像素偏移量迭代微调,返回物体 3D 基座系坐标(x,y,z,与 /robot0/eef_pose 同系)。",
        LOCATE_SCHEMA,
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
        self.declare_parameter("max_rounds", 5)
        self.declare_parameter("max_restarts", 2)
        self.declare_parameter("max_reproj_error_px", 25.0)
        # 绘制相关默认参数
        self.declare_parameter("draw_topic", "/nova/perception/draw")
        self.declare_parameter("draw_alpha", 0.45)
        self.declare_parameter("draw_supersample", 2)
        self.declare_parameter("draw_outline", True)
        self.declare_parameter("draw_outline_width", 2.0)
        self.declare_parameter("draw_outline_color", [0, 0, 0])
        self.declare_parameter("label_font_scale", 10.0)
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
        self._max_rounds = int(self.get_parameter("max_rounds").value)
        self._max_restarts = int(self.get_parameter("max_restarts").value)
        self._max_reproj_error_px = float(self.get_parameter("max_reproj_error_px").value)

        self._draw_topic = str(self.get_parameter("draw_topic").value)
        self._alpha = float(self.get_parameter("draw_alpha").value)
        self._supersample = int(self.get_parameter("draw_supersample").value)
        self._outline = bool(self.get_parameter("draw_outline").value)
        self._outline_width = float(self.get_parameter("draw_outline_width").value)
        self._outline_color = _parse_color(self.get_parameter("draw_outline_color").value, (0, 0, 0))
        self._label_font_scale = float(self.get_parameter("label_font_scale").value)
        self._arrow_radius = float(self.get_parameter("arrow_radius").value)
        self._point_radius = float(self.get_parameter("point_radius").value)
        self._axis_length = float(self.get_parameter("axis_length").value)
        self._segments = int(self.get_parameter("segments").value)
        self._image_cache_size = max(1, int(self.get_parameter("image_cache_size").value))

        self._frames: dict[str, np.ndarray] = {}
        for cam, topic in self.image_topics.items():
            self.create_subscription(Image, topic, self._make_cam_cb(cam), _CAM_QOS)

        self._image_cache: dict[str, dict] = {}
        self._image_counter = 0

        # debug 话题配置:off=不发布;sub=有订阅者才发布;on=总是发布
        self.declare_parameter("vlm_debug_mode", "on")
        mode = str(self.get_parameter("vlm_debug_mode").value).strip().lower()
        if mode not in ("off", "sub", "on"):
            raise RuntimeError(f"vlm_debug_mode 只支持 off|sub|on, 收到 {mode!r}")
        self._debug_mode = mode
        self.declare_parameter("vlm_round_topic", "/nova/perception/vlm_round")
        self._round_topic = str(self.get_parameter("vlm_round_topic").value).strip()
        self.declare_parameter("vlm_input_topic", "/nova/perception/vlm_input")
        self._input_base = str(self.get_parameter("vlm_input_topic").value).strip().rstrip("/")

        self._info_cg = MutuallyExclusiveCallbackGroup()
        self._info_client = self.create_client(
            EnvInfo, self.env_info_srv, callback_group=self._info_cg
        )
        self._llm = LLMClient(vision=True)
        self._round_pub = None
        self._cam_pubs: dict[str, Any] = {}
        if self._debug_mode != "off":
            if self._round_topic:
                self._round_pub = self.create_publisher(String, self._round_topic, 1)
            if self._input_base:
                for cam in self.camera_names:
                    topic = f"{self._input_base}/{cam}"
                    self._cam_pubs[cam] = self.create_publisher(Image, topic, 5)

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
            f"绘制话题={self._draw_topic}, vlm_debug_mode={self._debug_mode}"
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
                if name == "locate_object_3d":
                    result_json = self._locate(params, task_id=goal_handle.request.trace_id)
                else:
                    result_json = self._handle_visualize(name, params)
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

    def _handle_visualize(self, name: str, params: dict) -> dict:
        """把可视化工具名分发到对应绘制逻辑。"""
        if name == "visualize_frame":
            return self._draw_frame(params)
        if name == "visualize_point":
            return self._draw_point(params)
        if name == "visualize_segment":
            return self._draw_segment(params)
        if name == "visualize_ray":
            return self._draw_ray(params)
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

    def _finish_draw(self, image: np.ndarray, camera: str, source: str, tool: str) -> dict:
        """缓存并发布绘制结果,返回工具结果元数据。"""
        image_id = self._store_image(image, camera, source)
        frame_id = f"{source}|{image_id}"
        self._draw_pub.publish(self._numpy_to_image_msg(image, frame_id))
        return {
            "ok": True,
            "tool": tool,
            "image_id": image_id,
            "source": source,
            "camera": camera,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "topic": self._draw_topic,
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
        verts, faces, colors = vg.build_frame(origin, orientation, axis_length, radius, segments)
        out = self._render(image, K, P, verts, faces, colors, params)
        if params.get("labels", True):
            R = vg.quat_to_matrix_xyzw(orientation)
            Rt = vg.decompose_projection(K, P)
            font_px = self._label_font_px(K, Rt, origin, radius)
            for idx, (name, color) in enumerate(zip("xyz", ((255, 70, 70), (70, 200, 70), (70, 120, 255)))):
                tip = origin + R @ np.eye(3)[:, idx] * axis_length
                projected = vg.project_point_safe(K, Rt, tip)
                if projected is not None:
                    out = vg.draw_label(out, projected[:2], name, color, font_px=font_px)
        return self._finish_draw(out, cam, source, "visualize_frame")

    def _draw_point(self, params: dict) -> dict:
        """绘制一个 base 系 3D 点(小球)。"""
        image, cam, K, P, source = self._resolve_image(params["image"])
        point = _vec3(params.get("point"), "point")
        radius = float(params.get("radius", self._point_radius))
        color = _parse_color(params.get("color"), (255, 220, 0))
        verts, faces, colors = vg.build_sphere(point, radius, color, self._segments, max(4, self._segments // 2))
        out = self._render(image, K, P, verts, faces, colors, params)
        label = params.get("label")
        if label:
            Rt = vg.decompose_projection(K, P)
            projected = vg.project_point_safe(K, Rt, point)
            if projected is not None:
                out = vg.draw_label(out, projected[:2], str(label), color,
                                    font_px=self._label_font_px(K, Rt, point, radius))
        return self._finish_draw(out, cam, source, "visualize_point")

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
        return self._finish_draw(out, cam, source, "visualize_segment")

    def _draw_ray(self, params: dict) -> dict:
        """绘制一条 base 系 3D 射线(沿四元数局部 +z 的箭头)。"""
        image, cam, K, P, source = self._resolve_image(params["image"])
        origin = _vec3(params.get("origin"), "origin")
        orientation = _vec4(params.get("orientation"), "orientation")
        length = float(params.get("length", 0.2))
        radius = float(params.get("radius", self._arrow_radius))
        color = _parse_color(params.get("color"), (255, 120, 40))
        verts, faces, colors = vg.build_arrow(origin, orientation, length, radius,
                                              segments=self._segments, color=color)
        out = self._render(image, K, P, verts, faces, colors, params)
        label = params.get("label")
        if label:
            Rt = vg.decompose_projection(K, P)
            tip = origin + vg.quat_to_matrix_xyzw(orientation) @ np.array([0.0, 0.0, length])
            projected = vg.project_point_safe(K, Rt, tip)
            if projected is not None:
                out = vg.draw_label(out, projected[:2], str(label), color,
                                    font_px=self._label_font_px(K, Rt, origin, radius))
        return self._finish_draw(out, cam, source, "visualize_ray")

    # ---------- locate_object_3d ----------

    def _locate(self, params: dict, task_id: str = "") -> dict:
        """收集相机帧与投影矩阵,调用 VlmLocator 完成多视图 3D 定位。"""
        object_desc = str(params.get("object", "")).strip()
        if not object_desc:
            return {"ok": False, "error": "缺少 object 参数"}

        projections = self._fetch_projections()
        if not projections:
            return {"ok": False, "error": "env info 未提供相机投影矩阵(需升级 robocasa_sim_server)"}

        cams = params.get("camera_names") or self.camera_names
        images, projs = {}, {}
        for cam in cams:
            key = self._match_projection(cam, projections)
            frame = self._frames.get(cam)
            if frame is None:
                self.get_logger().warn(f"相机 {cam} 未收到帧")
                continue
            if key is None:
                self.get_logger().warn(f"相机 {cam} 无投影矩阵(可用:{list(projections)})")
                continue
            images[cam] = frame
            entry = projections[key]
            if isinstance(entry, dict) and "projection" in entry:
                entry = entry["projection"]
            projs[cam] = entry

        locator = VlmLocator(
            self._llm,
            grid_size=int(params.get("grid_size", self._grid_size)),
            max_rounds=int(params.get("max_rounds", self._max_rounds)),
            max_restarts=int(params.get("max_restarts", self._max_restarts)),
            max_reproj_error_px=float(params.get("max_reproj_error_px", self._max_reproj_error_px)),
        )
        return locator.locate(
            object_desc,
            images,
            projs,
            agent_context=str(params.get("_agent_context", "")),
            task_id=task_id,
            on_round=self._publish_vlm_round,
            on_images=self._publish_vlm_images,
        )

    # ---------- debug 发布 ----------

    def _want_debug_pub(self, pub) -> bool:
        """debug 话题是否该发:on 恒发;sub 有订阅者才发;off 时 publisher 未创建。"""
        if pub is None:
            return False
        if self._debug_mode == "on":
            return True
        if self._debug_mode == "sub":
            return pub.get_subscription_count() > 0
        return False

    def _publish_vlm_round(self, payload: dict) -> None:
        """VlmLocator 每轮回调:绘制图按相机发到各自话题,回合文本发到 vlm_round。"""
        task_id = str(payload.get("task_id", ""))
        round_tag = str(payload.get("round", ""))
        if self._debug_mode != "off":
            prompt = str(payload.get("prompt", ""))
            reply = str(payload.get("reply", ""))
            self.get_logger().info(
                f"[vlm] task={task_id} round={round_tag}\nprompt: {prompt}\nreply: {reply}"
            )
        self._publish_vlm_images(payload)
        if not self._want_debug_pub(self._round_pub):
            return
        try:
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._round_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"发布 vlm_round 失败: {exc}")

    def _publish_vlm_images(self, payload: dict) -> None:
        """只把标注图发到各相机 debug 话题(不写 vlm_round/日志),用于"发给模型前"推送。"""
        task_id = str(payload.get("task_id", ""))
        round_tag = str(payload.get("round", ""))
        for cam, url in (payload.get("images") or {}).items():
            pub = self._cam_pubs.get(cam)
            if not self._want_debug_pub(pub):
                continue
            try:
                frame_id = f"{task_id}|{round_tag}|{cam}"
                pub.publish(self._data_url_to_image_msg(url, frame_id))
            except Exception as exc:
                self.get_logger().warn(f"发布 vlm 输入图 {cam} 失败: {exc}")

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

    @staticmethod
    def _data_url_to_image_msg(data_url: str, frame_id: str) -> Image:
        """把 data:image/jpeg;base64 URL 解码并转成 ROS Image 消息。"""
        import base64
        import io

        from PIL import Image as PILImage

        b64 = data_url.split(",", 1)[1]
        jpeg = base64.b64decode(b64)
        image = np.asarray(PILImage.open(io.BytesIO(jpeg)).convert("RGB"))
        return PerceptionExecutorNode._numpy_to_image_msg(image, frame_id)

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
