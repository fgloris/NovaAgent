import pytest

from nova_perception_executor.perception_executor_node import _parse_color, _vec3, _vec4


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
