"""回归:SimError 序列化为 JSON,携带错误码与结构化明细,便于跨层解析。"""
import json

from nova_robocasa_bridge.robocasa_sim_server import SimError


def test_sim_error_serializes_code_and_details():
    error = SimError(
        "ik_failed",
        index=3,
        pos_err=0.0523,
        rot_err=0.031,
        iters=100,
        target_pos=[0.3, 0.1, 0.8],
        target_quat=[0.0, 0.0, 0.0, 1.0],
    )
    payload = json.loads(str(error))
    assert payload["code"] == "ik_failed"
    assert payload["index"] == 3
    assert payload["pos_err"] == 0.0523
    assert payload["rot_err"] == 0.031
    assert payload["iters"] == 100
    assert payload["target_pos"] == [0.3, 0.1, 0.8]
    assert payload["target_quat"] == [0.0, 0.0, 0.0, 1.0]


def test_sim_error_serializes_jump_without_errors():
    payload = json.loads(str(SimError("joint_jump_violation", index=0, jump=0.73)))
    assert payload["code"] == "joint_jump_violation"
    assert payload["jump"] == 0.73
    assert "pos_err" not in payload
    assert "iters" not in payload
