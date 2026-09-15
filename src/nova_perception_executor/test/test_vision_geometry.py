import numpy as np

from nova_perception_executor import vision_geometry as vg


def _camera():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    Rt = np.hstack([np.eye(3), np.array([[0.0], [0.0], [1.0]])])
    return K, K @ Rt


def test_decompose_projection_roundtrip():
    K, P = _camera()
    Rt = np.hstack([np.eye(3), np.array([[0.0], [0.0], [1.0]])])
    assert np.allclose(vg.decompose_projection(K, P), Rt)


def test_project_point_safe_behind_camera_returns_none():
    K, P = _camera()
    Rt = vg.decompose_projection(K, P)
    assert vg.project_point_safe(K, Rt, [0.0, 0.0, -5.0]) is None
    projected = vg.project_point_safe(K, Rt, [0.0, 0.0, 0.0])
    assert projected is not None
    assert projected[0] == 320.0 and projected[1] == 240.0


def test_axis_to_quat_maps_local_z_to_axis():
    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 1.0, 0.0]):
        quat = vg._axis_to_quat(axis)
        mapped = vg.quat_to_matrix_xyzw(quat) @ np.array([0.0, 0.0, 1.0])
        expected = np.asarray(axis, dtype=float)
        expected /= np.linalg.norm(expected)
        assert np.allclose(mapped, expected, atol=1e-6)


def test_build_arrow_rasterizes_onto_image():
    K, P = _camera()
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    verts, faces, colors = vg.build_arrow([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 0.1, 0.01)
    assert len(verts) > 0 and len(faces) > 0
    out = vg.rasterize_mesh(image, verts, faces, colors, K, P, alpha=0.5, supersample=2)
    assert out.shape == image.shape
    assert np.any(out != image)


def test_build_frame_and_cylinder_shapes():
    fv, ff, _ = vg.build_frame([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 0.1, 0.006)
    assert len(fv) > 0 and len(ff) > 0
    cv, cf, _ = vg.build_cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 0.1], 0.005)
    assert len(cv) > 0 and len(cf) > 0
