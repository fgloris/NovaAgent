import numpy as np
import pytest

from nova_common import image_codec
from nova_perception_executor import vision_geometry as vg
from nova_perception_executor.perception_executor_node import (
    MODEL_PARAM_DEFAULTS,
    TOOLS,
    PerceptionExecutorNode,
    _parse_color,
    _vec3,
    _vec4,
)


def test_visibility_warnings_and_status():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    Rt = np.hstack([np.eye(3), np.array([[0.0], [0.0], [1.0]])])

    ok = PerceptionExecutorNode._visibility_warnings(K, Rt, [("point", [0.0, 0.0, 0.0])], 640, 480)
    assert ok == []
    assert PerceptionExecutorNode._status(ok) == "drew successfully"

    behind = PerceptionExecutorNode._visibility_warnings(K, Rt, [("point", [0.0, 0.0, -5.0])], 640, 480)
    assert behind and "相机后方" in behind[0]
    assert PerceptionExecutorNode._status(behind).startswith("warning:")

    outside = PerceptionExecutorNode._visibility_warnings(K, Rt, [("point", [10.0, 0.0, 0.0])], 640, 480)
    assert outside and "超出画面" in outside[0]



def test_vec3_validates_size():
    assert list(_vec3([1, 2, 3], "origin")) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        _vec3([1, 2], "origin")


def test_vec4_validates_quaternion():
    assert _vec4([0, 0, 0, 1], "orientation") == [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        _vec4([0, 0, 1], "orientation")


def test_parse_color_name_list_and_fallback():
    assert _parse_color("red", (0, 0, 0)) == (255, 70, 70)
    assert _parse_color([10, 20, 30], (0, 0, 0)) == (10, 20, 30)
    assert _parse_color(None, (1, 2, 3)) == (1, 2, 3)
    assert _parse_color("nope", (1, 2, 3)) == (1, 2, 3)
    # 默认值本身是颜色名时也要解析(yaml 里 color: red)
    assert _parse_color(None, "red") == (255, 70, 70)
    assert _parse_color(None, "RED") == (255, 70, 70)
    assert _parse_color("blue", "red") == (70, 120, 255)
    assert _parse_color(None, "nope") == (255, 255, 255)


def _bare_node():
    node = PerceptionExecutorNode.__new__(PerceptionExecutorNode)
    node._display_max_size = 100
    node._inject_image_max_size = 0
    node._image_cache = {}
    node._frames = {
        "camA": np.zeros((200, 200, 3), dtype=np.uint8),
        "camB": np.zeros((200, 200, 3), dtype=np.uint8),
    }
    K = np.array([[200.0, 0.0, 100.0], [0.0, 200.0, 100.0], [0.0, 0.0, 1.0]])
    PA = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    PB = K @ np.hstack([np.eye(3), np.array([[0.2], [0.0], [0.0]])])
    node._fetch_projections = lambda: {
        "camA": {"intrinsics": K, "projection": PA},
        "camB": {"intrinsics": K, "projection": PB},
    }
    return node, {"camA": PA, "camB": PB}


def test_font_px_falls_back_to_common():
    node = PerceptionExecutorNode.__new__(PerceptionExecutorNode)
    node._common_font_px = 32
    assert node._font_px(16) == 16
    assert node._font_px(0) == 32
    assert node._font_px(-1) == 32


def _point_node():
    node = PerceptionExecutorNode.__new__(PerceptionExecutorNode)
    node._segments = 16
    node._alpha = 0.6
    node._supersample = 2
    node._outline = True
    node._outline_width = 2.0
    node._outline_color = (0, 0, 0)
    node._light_dir = [0.3, -0.5, 1.0]
    node._shade_min = 0.5
    node._inject_image_max_size = 0
    node._image_cache = {}
    node._image_counter = 0
    node._image_cache_size = 20
    node._draw_topic = "/nova/perception/draw"
    node._font_path = ""
    node._font_index = 2
    node._stroke_width = 2
    node._stroke_color = (0, 0, 0)
    node._common_font_px = 32
    node._point = {"radius": 0.02, "alpha": 1.0, "shade": False, "color": "red",
                   "label_font_px": 32, "label_offset_px": 4}

    class _Pub:
        def publish(self, _msg):
            pass

    node._draw_pub = _Pub()
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    P = K @ np.hstack([np.eye(3), np.array([[0.0], [0.0], [1.0]])])
    node._resolve_image = lambda src: (np.zeros((480, 640, 3), dtype=np.uint8), "cam", K, P, src)
    return node


def test_draw_point_accepts_named_default_color():
    # 回归:yaml 默认 color 是颜色名 "red" 时,不应把字符串漏进 build_sphere
    node = _point_node()
    out = node._draw_point({"image": "cam", "point": [0.0, 0.0, 0.0], "label": "目标点"})
    assert out["ok"] is True
    assert out["image_id"].startswith("viz_")


def test_tool_schema_injects_model_defaults_without_mutating_tools():
    node = PerceptionExecutorNode.__new__(PerceptionExecutorNode)
    node._model_defaults = {
        "visualize_point": {"radius": 0.02, "color": "red"},
        "visualize_pixels": {"radius_px": 12.0, "color": [255, 80, 80]},
    }
    point = node._tool_schema("visualize_point")
    assert point["properties"]["radius"]["default"] == 0.02
    assert point["properties"]["color"]["default"] == "red"
    # 命名差异:模型参数 radius_px 来自 visualize_pixels.radius
    pixels = node._tool_schema("visualize_pixels")
    assert pixels["properties"]["radius_px"]["default"] == 12.0
    assert pixels["properties"]["color"]["default"] == [255, 80, 80]
    # 深拷贝:不得污染模块级 TOOLS
    assert "default" not in TOOLS["visualize_point"][1]["properties"]["radius"]


def test_model_param_defaults_cover_schema_properties():
    for tool, params in MODEL_PARAM_DEFAULTS.items():
        properties = TOOLS[tool][1]["properties"]
        for param in params:
            assert param in properties, f"{tool}.{param} 不在 schema 中"


def test_reproject_pixels_recovers_point_across_display_native():
    node, projections = _bare_node()
    point = np.array([0.05, -0.03, 0.5])
    display_points = {}
    for cam, P in projections.items():
        u, v = vg.project_point(P, point)  # 原始分辨率像素
        display_points[cam] = image_codec.convert_points([u, v], (200, 200), (100, 100))

    out = node._reproject_pixels({"points": display_points})
    assert np.allclose(out["position"], point, atol=1e-3)
    assert out["display_size"]["camA"] == [100, 100]
    assert max(out["errors_px"].values()) < 1e-3
