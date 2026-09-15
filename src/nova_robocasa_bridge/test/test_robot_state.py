from types import SimpleNamespace

import numpy as np
import pytest

from nova_robocasa_bridge.robocasa_sim_server import RoboCasaSession, wxyz_to_xyzw


class _Model:
    def __init__(self):
        self.addresses = {"base_x": 0, "torso": 1, "joint1": 2, "joint2": 3}

    def get_joint_qpos_addr(self, name):
        return self.addresses[name]

    def get_joint_qvel_addr(self, name):
        return self.addresses[name]


def _session():
    data = SimpleNamespace(qpos=np.array([1.0, 2.0, 3.0, 4.0]), qvel=np.array([.1, .2, .3, .4]))
    sim = SimpleNamespace(model=_Model(), data=data)
    robot = SimpleNamespace(
        base_joints=["base_x"], torso_joints=["torso"], robot_joints=["joint1", "joint2"],
        _ref_joint_pos_indexes=[2, 3], _ref_joint_vel_indexes=[2, 3], gripper=None,
    )
    session = RoboCasaSession()
    session.env = SimpleNamespace(unwrapped=SimpleNamespace(robots=[robot], sim=sim))
    return session


def test_wxyz_is_converted_to_ros_xyzw():
    assert wxyz_to_xyzw([1.0, 0.1, 0.2, 0.3]) == [0.1, 0.2, 0.3, 1.0]


def test_robot_state_uses_canonical_obs_and_real_joint_names():
    session = _session()
    state = session._build_robot_state(
        {
            "state.end_effector_position_relative": np.array([0.2, -0.1, 0.5]),
            "state.end_effector_rotation_relative": np.array([0.0, 0.0, 0.0, 1.0]),
            "state.gripper_qpos": np.array([0.01, 0.02]),
        }
    )
    assert state["eef"]["position"] == [0.2, -0.1, 0.5]
    assert state["joints"]["name"] == ["base_x", "torso", "joint1", "joint2"]
    assert state["joints"]["position"] == [1.0, 2.0, 3.0, 4.0]
    assert state["joints"]["velocity"] == [.1, .2, .3, .4]
    assert state["gripper"]["position"] == [0.01, 0.02]


def test_failed_trajectory_validation_never_steps_environment():
    session = _session()
    calls = {"step": 0}
    session.env.step = lambda _action: calls.__setitem__("step", calls["step"] + 1)
    session._solve_trajectory_ik = lambda *_args: (_ for _ in ()).throw(ValueError("ik_failed"))
    with pytest.raises(ValueError, match="ik_failed"):
        session.validate_eef_trajectory(
            {"waypoints": [{"position": [0.2, 0.0, 0.5], "orientation": [0, 0, 0, 1]}]}
        )
    assert calls["step"] == 0

