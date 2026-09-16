import threading

from nova_executor_raw.ros_backend import RosEEFBackend
from nova_executor_raw.trajectory import Pose, compose, interpolate, validate_waypoints


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
    trajectory = interpolate([Pose([0, 0, 0], [0, 0, 0, 1]), poses[0]])
    assert trajectory[-1].gripper == -1.0


def test_gripper_holds_when_later_waypoint_omits_it():
    # 起点/中间指定夹爪后,后续不带夹爪的路径点应保持该值(修复"提起时松开")
    poses = [Pose([0, 0, 0], [0, 0, 0, 1], 1.0), Pose([0, 0, 0.1], [0, 0, 0, 1])]
    trajectory = interpolate(poses)
    assert all(pose.gripper == 1.0 for pose in trajectory)


def test_relative_compose_inherits_base_gripper():
    base = Pose([0, 0, 0], [0, 0, 0, 1], 1.0)
    delta = Pose([0, 0, 0.1], [0, 0, 0, 1])
    assert compose(base, delta).gripper == 1.0


def test_get_current_pose_carries_last_gripper():
    backend = RosEEFBackend.__new__(RosEEFBackend)
    backend.robot_id = "robot0"
    backend._pose = Pose([0, 0, 0], [0, 0, 0, 1])
    backend._last_gripper = 1.0
    backend._lock = threading.Lock()
    assert backend.get_current_pose("robot0").gripper == 1.0

