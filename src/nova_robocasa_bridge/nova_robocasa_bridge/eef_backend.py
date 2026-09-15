"""RoboCasa EEF backend adapter.

The adapter deliberately keeps simulator-specific details here.  A deployment can
inject a session/object implementing ``get_eef_pose``, ``solve_ik`` and
``apply_eef_target`` without changing the executor or MCP schema.
"""

from __future__ import annotations

from typing import Any, Callable

from nova_executor_raw.trajectory import Pose


class RoboCasaEEFBackend:
    def __init__(self, session: Any | None = None, robot_id: str = "robot0") -> None:
        self.session = session
        self.robot_id = robot_id

    def get_joint_state(self, robot_id: str | None = None) -> dict:
        env = getattr(self.session, "env", None)
        if env is None:
            raise RuntimeError("backend_unavailable")
        robot = env.unwrapped.robots[0]
        ids = list(getattr(robot, "_ref_joint_pos_indexes", []))
        qpos = getattr(env.sim.data, "qpos", [])
        return {"name": list(getattr(robot, "joint_names", [])), "position": [float(qpos[i]) for i in ids]}

    def describe(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "robot_base": f"{self.robot_id}_base",
            "eef_frame": f"{self.robot_id}_ee",
            "supports_gripper": True,
            "supports_absolute_pose": True,
            "supports_relative_pose": True,
            "max_linear_speed": 0.2,
            "max_angular_speed": 1.0,
            "workspace": {"min": [-0.8, -0.8, 0.0], "max": [0.8, 0.8, 1.2]},
            "ik_solver": "robocasa",
        }

    def get_current_pose(self, robot_id: str) -> Pose:
        obs = getattr(self.session, "latest_obs", None) if self.session is not None else None
        if obs:
            p, q = obs.get(f"{self.robot_id}_base_to_eef_pos"), obs.get(f"{self.robot_id}_base_to_eef_quat")
            if p is not None and q is not None:
                from nova_executor_raw.trajectory import normalize
                return Pose(list(map(float, p[:3])), normalize(list(map(float, q[:4]))))
        if self.session is not None and hasattr(self.session, "get_eef_pose"):
            return self.session.get_eef_pose(robot_id)
        raise RuntimeError("eef_observation_unavailable")

    def transform_pose(self, pose: Pose, source_frame: str, target_frame: str) -> Pose:
        if source_frame == target_frame or self.session is None:
            return pose
        if hasattr(self.session, "transform_pose"):
            return self.session.transform_pose(pose, source_frame, target_frame)
        raise ValueError("frame_unavailable")

    def validate_trajectory(
        self, trajectory: list[Pose], constraints: dict
    ) -> str | None:
        if self.session is None or not hasattr(self.session, "solve_ik"):
            return None
        try:
            seed = self.get_joint_state().get("position")
        except Exception:
            seed = None
        for pose in trajectory:
            solution = self.session.solve_ik(self.robot_id, pose, seed=seed)
            if solution is None:
                return "ik_failed"
            seed = solution
        return None

    def execute_trajectory(
        self,
        trajectory: list[Pose],
        feedback_callback: Callable,
        cancel_callback: Callable[[], bool],
    ) -> dict:
        if self.session is None or not hasattr(self.session, "apply_eef_target"):
            return {"success": False, "error": "backend_unavailable"}
        for index, pose in enumerate(trajectory):
            if cancel_callback():
                return {"success": False, "error": "cancelled"}
            self.session.apply_eef_target(self.robot_id, pose)
            feedback_callback(index, len(trajectory), "running", "executing")
        return {"success": True, "executed_points": len(trajectory)}
