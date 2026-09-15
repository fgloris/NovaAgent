from nova_executor_raw.ros_backend import RosEEFBackend
from nova_executor_raw.trajectory import Pose, interpolate, validate_waypoints


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

