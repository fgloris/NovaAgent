import json
import threading

from nova_executor_raw.ros_backend import RosEEFBackend
from nova_executor_raw.trajectory import Pose, compose, describe_failure, interpolate, validate_waypoints


class _UnavailableClient:
    def server_is_ready(self):
        return False


def test_backend_unavailable_is_explicit():
    backend = RosEEFBackend.__new__(RosEEFBackend)
    backend.client = _UnavailableClient()
    assert backend.validate_trajectory([], {}) == "backend_unavailable"


def test_gripper_survives_validation_and_interpolation():
    poses = validate_waypoints(
        [{"position": [0, 0, 0], "orientation": [0, 0, 0, 1], "gripper": -1.0}]
    )
    trajectory, _ = interpolate([Pose([0, 0, 0], [0, 0, 0, 1]), poses[0]])
    assert trajectory[-1].gripper == -1.0


def test_gripper_holds_when_later_waypoint_omits_it():
    # 起点/中间指定夹爪后,后续不带夹爪的路径点应保持该值(修复"提起时松开")
    poses = [Pose([0, 0, 0], [0, 0, 0, 1], 1.0), Pose([0, 0, 0.1], [0, 0, 0, 1])]
    trajectory, _ = interpolate(poses)
    assert all(pose.gripper == 1.0 for pose in trajectory)


def test_pure_gripper_change_gets_its_own_duration():
    # 位置不变、仅夹爪变化时,应给出足够帧数让夹爪闭合,而不是瞬间结束
    poses = [Pose([0, 0, 0], [0, 0, 0, 1], 1.0), Pose([0, 0, 0], [0, 0, 0, 1], -1.0)]
    trajectory, _ = interpolate(poses, gripper_speed=1.0, hz=20)
    assert len(trajectory) > 20
    assert trajectory[-1].gripper == -1.0


def test_gripper_closes_after_reaching_target_position():
    # 先到位再闭合:夹爪开始变化时,位置已到达段终点
    poses = [Pose([0, 0, 0], [0, 0, 0, 1], 1.0), Pose([0, 0, 0.2], [0, 0, 0, 1], -1.0)]
    trajectory, _ = interpolate(poses, linear_speed=0.2, gripper_speed=1.0)
    grips = [pose.gripper for pose in trajectory]
    assert grips[0] == 1.0
    assert all(a >= b for a, b in zip(grips, grips[1:]))
    assert grips[-1] == -1.0
    assert any(g not in (1.0, -1.0) for g in grips)
    first_change = next(pose for pose in trajectory if pose.gripper != 1.0)
    assert first_change.position == [0.0, 0.0, 0.2]


def test_relative_compose_inherits_base_gripper():
    base = Pose([0, 0, 0], [0, 0, 0, 1], 1.0)
    delta = Pose([0, 0, 0.1], [0, 0, 0, 1])
    assert compose(base, delta).gripper == 1.0


def test_interpolate_segments_cover_trajectory():
    # 段索引与轨迹等长、单调不减,首点属第 0 段、末点属最后一段
    poses = [
        Pose([0, 0, 0], [0, 0, 0, 1], 1.0),
        Pose([0, 0, 0.1], [0, 0, 0, 1], 1.0),
        Pose([0, 0.1, 0.1], [0, 0, 0, 1], -1.0),
    ]
    trajectory, segments = interpolate(poses, linear_speed=0.2, gripper_speed=1.0)
    assert len(segments) == len(trajectory)
    assert segments == sorted(segments)
    assert segments[0] == 0
    assert segments[-1] == len(poses) - 2


def test_describe_failure_maps_ik_error_to_waypoint():
    poses = [
        Pose([0, 0, 0], [0, 0, 0, 1], 1.0),
        Pose([0, 0, 0.1], [0, 0, 0, 1], 1.0),
        Pose([0.3, 0.1, 0.8], [0, 0, 0, 1], -1.0),
    ]
    trajectory, segments = interpolate(poses, linear_speed=0.2, gripper_speed=1.0)
    index = len(trajectory) - 1
    detail = describe_failure(
        json.dumps(
            {
                "code": "ik_failed",
                "index": index,
                "pos_err": 0.0523,
                "rot_err": 0.031,
                "iters": 100,
                "target_pos": [0.3, 0.1, 0.8],
                "target_quat": [0.0, 0.0, 0.0, 1.0],
            }
        ),
        segments,
        poses,
    )
    assert detail["failed_waypoint"] == 2
    assert detail["failed_segment"] == segments[index]
    assert "第2个waypoint" in detail["message"]
    assert "[0.300, 0.100, 0.800]" in detail["message"]
    assert "pos_err=0.0523" in detail["message"]
    assert detail["diagnostics"]["code"] == "ik_failed"


def test_describe_failure_handles_joint_jump_and_unknown():
    poses = [Pose([0, 0, 0], [0, 0, 0, 1]), Pose([0, 0, 0.1], [0, 0, 0, 1])]
    _, segments = interpolate(poses)
    detail = describe_failure(
        json.dumps({"code": "joint_jump_violation", "index": 0, "jump": 0.73}),
        segments,
        poses,
    )
    assert detail["failed_waypoint"] == 1
    assert "关节跳变" in detail["message"]
    assert "jump=0.73" in detail["message"]
    assert describe_failure(json.dumps({"code": "ik_failed", "index": 999}), segments, poses) is None
    assert describe_failure("backend_unavailable", segments, poses) is None


def test_get_current_pose_carries_last_gripper():
    backend = RosEEFBackend.__new__(RosEEFBackend)
    backend.robot_id = "robot0"
    backend._pose = Pose([0, 0, 0], [0, 0, 0, 1])
    backend._last_gripper = 1.0
    backend._lock = threading.Lock()
    assert backend.get_current_pose("robot0").gripper == 1.0

