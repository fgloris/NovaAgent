import numpy as np
import pytest

from nova_perception_executor.perception_executor_node import (
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
