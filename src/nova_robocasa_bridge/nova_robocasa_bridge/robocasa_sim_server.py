#!/usr/bin/env python3
"""RoboCasa 仿真 server:独立进程托管环境,经 JSON 帧协议对外提供 reset/step 等接口。

同时计算各相机的内参与投影矩阵,供感知 executor 做多视图 3D 定位。
"""
from __future__ import annotations
import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np


def _ensure_nova_common_importable() -> None:
    """把 nova_common 所在目录加入 sys.path,使直接 python 运行时(未 source)也能 import。"""
    here = Path(__file__).resolve()
    candidates = []
    src_parent = here.parents[2]
    if (src_parent / "nova_common" / "nova_common" / "__init__.py").is_file():
        candidates.append(str(src_parent / "nova_common"))
    for parent in here.parents:
        if parent.name == "site-packages":
            candidates.append(str(parent))
            break
    for candidate in candidates:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_nova_common_importable()

from nova_common.jsonline import serve
from nova_common.obs_codec import build_obs_spec, normalize_obs


# 规范动作向量 -> RoboCasa 动作 dict(缺省补零并截断到 [-1,1])
# 布局约定: [pos3, rot3, gripper, base4, control_mode]
_ROBOCASA_ACTION_MEANING = [
    "pos_x", "pos_y", "pos_z",
    "rot_x", "rot_y", "rot_z",
    "gripper",
    "base_vx", "base_vy", "base_wz", "base_rz",
    "control_mode",
]


def wxyz_to_xyzw(quaternion: Any) -> list[float]:
    """把 MuJoCo/robosuite 格式的 WXYZ 四元数转成 ROS XYZW."""
    q = np.asarray(quaternion, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError("quaternion must contain four values")
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _normalize_quat_xyzw(quaternion: Any) -> np.ndarray:
    """归一化 XYZW 四元数;输入为 3 维时按轴角(罗德里格斯)转四元数。"""
    q = np.asarray(quaternion, dtype=float).reshape(-1)
    if q.size == 3:
        angle = float(np.linalg.norm(q))
        if angle < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        q = np.r_[q / angle * np.sin(angle / 2.0), np.cos(angle / 2.0)]
    if q.size != 4 or not np.all(np.isfinite(q)):
        raise ValueError("invalid quaternion")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("invalid quaternion")
    return q / norm


def _orientation_to_quat_xyzw(orientation: Any) -> np.ndarray:
    """把多种姿态表示统一为 XYZW 四元数。
    兼容:3 维(轴角/旋转向量)、4 维(四元数)、6 维(两个基向量,先正交化)、9 维(旋转矩阵)。
    """
    value = np.asarray(orientation, dtype=float).reshape(-1)
    if value.size in (3, 4):
        return _normalize_quat_xyzw(value)
    if value.size == 6:
        first = value[:3] / np.linalg.norm(value[:3])
        second = value[3:] - np.dot(first, value[3:]) * first
        second /= np.linalg.norm(second)
        return _matrix_to_quat_xyzw(np.column_stack((first, second, np.cross(first, second))))
    if value.size == 9:
        return _matrix_to_quat_xyzw(value.reshape(3, 3))
    raise ValueError("invalid orientation")


def _matrix_to_quat_xyzw(matrix: Any) -> np.ndarray:
    """把 3x3 旋转矩阵转成归一化的 XYZW 四元数(按迹选择数值稳定的分支)。"""
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                          (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                          (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                          0.25 * s, (m[1, 0] - m[0, 1]) / s])
    return _normalize_quat_xyzw(q)


def _quat_multiply_xyzw(a: Any, b: Any) -> np.ndarray:
    """按 XYZW 顺序计算两个四元数的乘积(归一化后,结果表示先 b 后 a 的复合旋转)。"""
    ax, ay, az, aw = _normalize_quat_xyzw(a)
    bx, by, bz, bw = _normalize_quat_xyzw(b)
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _quat_to_matrix_xyzw(q: Any) -> np.ndarray:
    """把 XYZW 四元数转成 3x3 旋转矩阵。"""
    x, y, z, w = _normalize_quat_xyzw(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """计算目标旋转与当前旋转的误差向量(旋转矩阵反对称部分,近似轴角,供 IK 用)。"""
    error = target @ current.T
    return 0.5 * np.array([
        error[2, 1] - error[1, 2],
        error[0, 2] - error[2, 0],
        error[1, 0] - error[0, 1],
    ])


def base_frame_projection(
    projection_world: Any, base_position: Any, base_rotation: Any
) -> list[list[float]]:
    """世界系 3x4 投影矩阵 -> base 系。

    X_world = T_world_base @ X_base,pixel = P_world @ T @ X_base,故 P_base = P_world @ T。
    """
    p = np.asarray(projection_world, dtype=float).reshape(3, 4)
    t = np.eye(4)
    t[:3, :3] = np.asarray(base_rotation, dtype=float).reshape(3, 3)
    t[:3, 3] = np.asarray(base_position, dtype=float).reshape(3)
    return (p @ t).tolist()


def base_rotation_matrix(base_ori: Any) -> np.ndarray:
    """把 robosuite 的 robot.base_ori 统一成 3x3 旋转矩阵。

    robosuite 1.5 起 base_ori 是 3x3 矩阵;兼容旧版 WXYZ 四元数与 9 维展平矩阵。
    """
    if base_ori is None:
        return np.eye(3)
    value = np.asarray(base_ori, dtype=float)
    if value.shape == (3, 3):
        return value
    flat = value.reshape(-1)
    if flat.size == 4:
        return _quat_to_matrix_xyzw(wxyz_to_xyzw(flat))
    if flat.size == 9:
        return flat.reshape(3, 3)
    raise ValueError("invalid base_ori")


def action_vector_to_dict(values: np.ndarray) -> dict[str, list[float]]:
    """把规范动作向量 [pos3,rot3,gripper,base4,control_mode] 转成 RoboCasa 动作 dict。

    长度不足时补零,超出部分截断,并统一 clip 到 [-1, 1]。
    """
    values = np.ravel(values).astype(np.float32)
    if values.size > len(_ROBOCASA_ACTION_MEANING):
        raise ValueError(
            f"expected at most {len(_ROBOCASA_ACTION_MEANING)} action values, got {values.size}"
        )
    if values.size < len(_ROBOCASA_ACTION_MEANING):
        values = np.pad(values, (0, len(_ROBOCASA_ACTION_MEANING) - values.size))
    values = np.clip(values, -1.0, 1.0)

    return {
        "action.end_effector_position": values[0:3],
        "action.end_effector_rotation": values[3:6],
        "action.gripper_close": float(values[6]),
        "action.base_motion": values[7:11],
        "action.control_mode": float(values[11]),
    }


_RENDER_PRESETS: dict[str, dict[str, Any]] = {
    "low": dict(
        shadowsize=1024, offsamples=0,
        ambient=0.4, diffuse=0.6, specular=0.2, shininess=1.0,
    ),
    "medium": dict(
        shadowsize=2048, offsamples=2,
        ambient=0.35, diffuse=0.8, specular=0.4, shininess=1.2,
    ),
    "high": dict(
        shadowsize=4096, offsamples=4,
        ambient=0.3, diffuse=0.95, specular=0.55, shininess=1.4,
    ),
    "ultra": dict(
        shadowsize=8192, offsamples=8,
        ambient=0.25, diffuse=1.0, specular=0.7, shininess=1.6,
    ),
}


def _set_quality_field(model: Any, attr: str, value: int) -> None:
    if hasattr(model.vis.quality, attr):
        setattr(model.vis.quality, attr, value)


def _apply_render_quality(env: Any, quality: str) -> None:
    """MuJoCo 离屏渲染增强:阴影分辨率/抗锯齿/光照对比/材质高光。"""
    preset = _RENDER_PRESETS.get(quality, _RENDER_PRESETS["high"])
    sim = env.env.sim
    model = sim.model
    _set_quality_field(model, "shadowsize", preset["shadowsize"])
    _set_quality_field(model, "offsamples", preset["offsamples"])
    if model.nlight > 0:
        model.light_ambient[:] = preset["ambient"]
        model.light_diffuse[:] = preset["diffuse"]
        if hasattr(model, "light_castshadow"):
            model.light_castshadow[:] = 1
    if hasattr(model, "mat_specular"):
        model.mat_specular[:] = preset["specular"]
    if hasattr(model, "mat_shininess"):
        model.mat_shininess[:] = preset["shininess"]
    print(
        "[render] quality={} shadowsize={} offsamples={} nlight={} "
        "ambient={:.2f} diffuse={:.2f} specular={:.2f} shininess={:.2f}".format(
            quality,
            model.vis.quality.shadowsize,
            model.vis.quality.offsamples,
            model.nlight,
            float(model.light_ambient[0, 0]) if model.nlight else 0.0,
            float(model.light_diffuse[0, 0]) if model.nlight else 0.0,
            float(model.mat_specular[0]) if model.nmat else 0.0,
            float(model.mat_shininess[0]) if model.nmat else 0.0,
        ),
        flush=True,
    )


class RoboCasaSession:
    """托管一个 RoboCasa gym 环境,处理 reset/step/EEF 轨迹等请求(带线程锁)。"""

    def __init__(self, scene_config: dict[str, Any] | None = None) -> None:
        self.scene_config = scene_config or {}
        self.env = None
        self.env_config: dict[str, Any] | None = None
        self.latest_obs: dict[str, Any] | None = None
        self._lock = threading.RLock()

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        """按需初始化环境并 reset,返回 obs、规格、sim 信息与机器人状态。"""
        with self._lock:
            self._ensure_env(request)
            assert self.env is not None
            obs, info = self.env.reset(seed=int(request.get("seed", 0)))
            obs = self._prepare_obs(obs, info)
            self.latest_obs = obs
            robot_state = self._build_robot_state(obs)
        print(
            f"[robocasa] env reset: task description = {obs.get('state.instruction', '')!r}",
            flush=True,
        )
        return {
            "ok": True,
            "obs": obs,
            "info": info,
            "action_spec": self._action_spec(),
            "obs_spec": build_obs_spec(obs),
            "sim_info": self._sim_info(),
            "robot_state": robot_state,
        }

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        """用请求中的规范动作步进一次环境,返回新的 obs/reward/terminated 等。"""
        with self._lock:
            if self.env is None:
                raise RuntimeError("environment is not initialized; call reset first")
            action = action_vector_to_dict(np.asarray(request["action"], dtype=np.float32))
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self._prepare_obs(obs, info)
            self.latest_obs = obs
            response = {
            "ok": True,
            "obs": obs,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
            "action_spec": self._action_spec(),
            "obs_spec": build_obs_spec(obs),
            "sim_info": self._sim_info(),
            "robot_state": self._build_robot_state(obs),
            }
        return response

    def robot_state(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """返回当前机器人的 EEF/关节/夹爪状态(不步进环境)。"""
        del request
        with self._lock:
            if self.env is None:
                raise RuntimeError("environment is not initialized; call reset first")
            return {"ok": True, "robot_state": self._build_robot_state(self.latest_obs or {})}

    def validate_eef_trajectory(self, request: dict[str, Any]) -> dict[str, Any]:
        """校验 EEF 路径点并对每个点求解 IK,返回关节解序列。"""
        with self._lock:
            waypoints = self._validate_waypoint_payload(request.get("waypoints"))
            solutions = self._solve_trajectory_ik(waypoints, request)
            return {"ok": True, "valid": True, "solutions": solutions}

    def step_eef(self, request: dict[str, Any]) -> dict[str, Any]:
        """把单个 EEF 路径点转成相对位姿增量并按 OSC 输出上限归一化后步进一次。"""
        with self._lock:
            if self.env is None:
                raise RuntimeError("environment is not initialized; call reset first")
            waypoint = self._validate_waypoint_payload([request.get("waypoint")])[0]
            current = self._eef_pose_base(self.latest_obs or {})
            position_delta = waypoint["position"] - current[0]
            q_current = current[1]
            q_target = waypoint["orientation"]
            q_error = _quat_multiply_xyzw(q_target, [-q_current[0], -q_current[1], -q_current[2], q_current[3]])
            if q_error[3] < 0:
                q_error = -q_error
            vector_norm = float(np.linalg.norm(q_error[:3]))
            rotation_delta = np.zeros(3)
            if vector_norm > 1e-9:
                rotation_delta = q_error[:3] / vector_norm * (2.0 * np.arctan2(vector_norm, q_error[3]))
            output_max = self._osc_output_max()
            command = np.zeros(len(_ROBOCASA_ACTION_MEANING), dtype=np.float32)
            command[:3] = np.clip(position_delta / output_max[:3], -1.0, 1.0)
            command[3:6] = np.clip(rotation_delta / output_max[3:6], -1.0, 1.0)
            if waypoint["gripper"] is not None:
                command[6] = float(np.clip(waypoint["gripper"], -1.0, 1.0))
            action = action_vector_to_dict(command)
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self._prepare_obs(obs, info)
            self.latest_obs = obs
            return {
                "ok": True,
                "obs": obs,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
                "action_spec": self._action_spec(),
                "obs_spec": build_obs_spec(obs),
                "sim_info": self._sim_info(),
                "robot_state": self._build_robot_state(obs),
            }

    def _validate_waypoint_payload(self, raw: Any) -> list[dict[str, Any]]:
        """校验并规范化路径点:位置需在工作空间内,姿态统一为 XYZW 四元数。"""
        if not isinstance(raw, list) or not raw:
            raise ValueError("empty_waypoints")
        workspace = self.scene_config.get("workspace", {"min": [-0.8, -0.8, 0.0], "max": [0.8, 0.8, 1.2]})
        result = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("invalid_pose")
            position = np.asarray(item.get("position"), dtype=float).reshape(-1)
            if position.size != 3 or not np.all(np.isfinite(position)):
                raise ValueError("invalid_pose")
            if any(position[i] < workspace["min"][i] or position[i] > workspace["max"][i] for i in range(3)):
                raise ValueError("workspace_violation")
            orientation = _orientation_to_quat_xyzw(item.get("orientation"))
            gripper = item.get("gripper")
            if gripper is not None and not np.isfinite(float(gripper)):
                raise ValueError("invalid_gripper")
            result.append({"position": position, "orientation": orientation,
                           "gripper": None if gripper is None else float(gripper)})
        return result

    def _obs_value(self, obs: dict[str, Any], *names: str) -> Any:
        """按候选名依次查找 obs 值,兼容带/不带 "state." 前缀的键。"""
        for name in names:
            for key in (f"state.{name}", name):
                if key in obs:
                    return obs[key]
        return None

    def _eef_pose_base(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """返回 EEF 在机器人基座坐标系下的 (位置, XYZW 四元数)。

        优先取 obs 里现成的相对量;缺失时回退到 sim 的 site 世界位姿,
        再用基座位姿做逆变换(旋转转置)换算到基座系。
        """
        position = self._obs_value(obs, "end_effector_position_relative", "robot0_base_to_eef_pos")
        orientation = self._obs_value(obs, "end_effector_rotation_relative", "robot0_base_to_eef_quat")
        if position is not None and orientation is not None:
            return np.asarray(position, dtype=float).reshape(-1)[:3], _orientation_to_quat_xyzw(orientation)
        robot = self.env.unwrapped.robots[0]
        sim = self.env.unwrapped.sim
        site_ids = getattr(robot, "eef_site_id", {})
        site_id = site_ids.get("right") if isinstance(site_ids, dict) else site_ids
        if site_id is None:
            raise RuntimeError("eef_site_unavailable")
        world_position = np.asarray(sim.data.site_xpos[int(site_id)], dtype=float)
        world_rotation = np.asarray(sim.data.site_xmat[int(site_id)], dtype=float).reshape(3, 3)
        base_position, base_rotation = self._base_pose_world()
        return base_rotation.T @ (world_position - base_position), _matrix_to_quat_xyzw(base_rotation.T @ world_rotation)

    @staticmethod
    def _flatten_indexes(value: Any) -> list[int]:
        """把任意形状的索引容器展平为 int 列表(None 返回空列表)。"""
        if value is None:
            return []
        return [int(v) for v in np.asarray(value).reshape(-1)]

    def _joint_data(self) -> tuple[list[str], list[int], list[int]]:
        """解析机器人关节名及其在 qpos/qvel 中的索引;优先按名字查,回退到机器人自带索引。"""
        robot = self.env.unwrapped.robots[0]
        names: list[str] = []
        for attribute in ("base_joints", "torso_joints", "robot_joints", "joint_names"):
            for name in list(getattr(robot, attribute, None) or []):
                if str(name) not in names:
                    names.append(str(name))
        qpos_ids = self._flatten_indexes(getattr(robot, "_ref_joint_pos_indexes", None))
        qvel_ids = self._flatten_indexes(getattr(robot, "_ref_joint_vel_indexes", None))
        model = self.env.unwrapped.sim.model
        resolved = []
        for name in names:
            try:
                qpos_addr = model.get_joint_qpos_addr(name)
                qvel_addr = model.get_joint_qvel_addr(name)
                if isinstance(qpos_addr, tuple) or isinstance(qvel_addr, tuple):
                    continue
                resolved.append((name, int(qpos_addr), int(qvel_addr)))
            except Exception:
                continue
        if resolved:
            return ([item[0] for item in resolved], [item[1] for item in resolved],
                    [item[2] for item in resolved])
        count = min(len(names), len(qpos_ids))
        return [str(v) for v in names[:count]], qpos_ids[:count], qvel_ids[:count]

    def _build_robot_state(self, obs: dict[str, Any]) -> dict[str, Any]:
        """汇总机器人状态:EEF 位姿、关节名/位置/速度、夹爪位置,供 /robot_state 与心跳使用。"""
        position, orientation = self._eef_pose_base(obs)
        names, qpos_ids, qvel_ids = self._joint_data()
        data = self.env.unwrapped.sim.data
        gripper = self._obs_value(obs, "gripper_qpos", "robot0_gripper_qpos")
        if gripper is None:
            robot = self.env.unwrapped.robots[0]
            gripper_obj = getattr(robot, "gripper", None)
            if isinstance(gripper_obj, dict):
                gripper_obj = gripper_obj.get("right") or next(iter(gripper_obj.values()), None)
            gripper_ids = self._flatten_indexes(getattr(gripper_obj, "_ref_joint_pos_indexes", None))
            gripper = [float(data.qpos[i]) for i in gripper_ids]
        velocities = [float(data.qvel[i]) for i in qvel_ids] if len(qvel_ids) == len(names) else []
        return {
            "robot_id": "robot0",
            "base_frame": "robot0_base",
            "eef": {"position": position.tolist(), "orientation": orientation.tolist()},
            "joints": {
                "name": names,
                "position": [float(data.qpos[i]) for i in qpos_ids],
                "velocity": velocities,
            },
            "gripper": {"position": np.asarray(gripper, dtype=float).reshape(-1).tolist()},
        }

    def _arm_indexes(self) -> tuple[list[int], list[int]]:
        """返回机械臂关节在 qpos/qvel 中的索引;优先用 arm 专属索引,回退取末尾 7 个。"""
        robot = self.env.unwrapped.robots[0]
        qpos = self._flatten_indexes(getattr(robot, "_ref_arm_joint_pos_indexes", None))
        qvel = self._flatten_indexes(getattr(robot, "_ref_arm_joint_vel_indexes", None))
        if not qpos or not qvel:
            qpos = self._flatten_indexes(getattr(robot, "_ref_joint_pos_indexes", None))[-7:]
            qvel = self._flatten_indexes(getattr(robot, "_ref_joint_vel_indexes", None))[-7:]
        if not qpos or len(qpos) != len(qvel):
            raise RuntimeError("arm_joint_indexes_unavailable")
        return qpos, qvel

    def _base_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        """返回机器人在世界系下的 (base_position, base_rotation 3x3)。"""
        robot = self.env.unwrapped.robots[0]
        base_position = np.asarray(getattr(robot, "base_pos", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)[:3]
        base_rotation = base_rotation_matrix(getattr(robot, "base_ori", None))
        return base_position, base_rotation

    def _target_world(self, waypoint: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """把路径点从基座坐标系变换到世界坐标系,返回 (世界位置, 世界旋转矩阵)。"""
        base_position, base_rotation = self._base_pose_world()
        return (base_position + base_rotation @ waypoint["position"],
                base_rotation @ _quat_to_matrix_xyzw(waypoint["orientation"]))

    def _solve_trajectory_ik(self, waypoints: list[dict[str, Any]], constraints: dict[str, Any]) -> list[list[float]]:
        """对每个路径点用阻尼最小二乘(DLS)迭代求解机械臂关节角。

        每步由 site 雅可比反解位姿误差(位置 + 旋转),并对步长/关节限位/关节跳变设约束;
        求解前后保存并恢复 sim 状态,避免污染真实环境。
        """
        if self.env is None:
            raise RuntimeError("environment is not initialized; call reset first")
        import mujoco

        sim = self.env.unwrapped.sim
        model = getattr(sim.model, "_model", sim.model)
        data = getattr(sim.data, "_data", sim.data)
        robot = self.env.unwrapped.robots[0]
        site_ids = getattr(robot, "eef_site_id", {})
        site_id = site_ids.get("right") if isinstance(site_ids, dict) else site_ids
        if site_id is None:
            raise RuntimeError("eef_site_unavailable")
        qpos_ids, dof_ids = self._arm_indexes()
        saved_qpos = np.array(data.qpos, copy=True)
        saved_qvel = np.array(data.qvel, copy=True)
        saved_act = np.array(data.act, copy=True) if getattr(data, "act", None) is not None else None
        saved_state = sim.get_state() if hasattr(sim, "get_state") else None
        seed = np.array(data.qpos[qpos_ids], copy=True)
        jump_limit = float(constraints.get("joint_jump_threshold", self.scene_config.get("joint_jump_threshold", 0.5)))
        damping = float(constraints.get("ik_damping", 1e-3))
        solutions: list[list[float]] = []
        try:
            for waypoint in waypoints:
                data.qpos[qpos_ids] = seed
                mujoco.mj_forward(model, data)
                target_position, target_rotation = self._target_world(waypoint)
                solved = False
                for _ in range(100):
                    current_position = np.asarray(data.site_xpos[int(site_id)])
                    current_rotation = np.asarray(data.site_xmat[int(site_id)]).reshape(3, 3)
                    error = np.r_[target_position - current_position,
                                  _rotation_error(target_rotation, current_rotation)]
                    if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 2e-3:
                        solved = True
                        break
                    jac_pos = np.zeros((3, model.nv))
                    jac_rot = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, int(site_id))
                    jacobian = np.vstack((jac_pos[:, dof_ids], jac_rot[:, dof_ids]))
                    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
                    data.qpos[qpos_ids] += np.clip(delta, -0.1, 0.1)
                    self._enforce_joint_limits(model, data, qpos_ids)
                    mujoco.mj_forward(model, data)
                if not solved:
                    raise ValueError(f"ik_failed:{len(solutions)}")
                solution = np.array(data.qpos[qpos_ids], copy=True)
                if np.max(np.abs(solution - seed)) > jump_limit:
                    raise ValueError(f"joint_jump_violation:{len(solutions)}")
                solutions.append(solution.tolist())
                seed = solution
        finally:
            if saved_state is not None and hasattr(sim, "set_state"):
                sim.set_state(saved_state)
                sim.forward()
            else:
                data.qpos[:] = saved_qpos
                data.qvel[:] = saved_qvel
                if saved_act is not None:
                    data.act[:] = saved_act
                mujoco.mj_forward(model, data)
        return solutions

    @staticmethod
    def _enforce_joint_limits(model: Any, data: Any, qpos_ids: list[int]) -> None:
        """把受限关节的 qpos 夹到其 [low, high] 范围内(只处理给定 qpos 索引)。"""
        for joint_id in range(int(model.njnt)):
            address = int(model.jnt_qposadr[joint_id])
            if address not in qpos_ids or not bool(model.jnt_limited[joint_id]):
                continue
            low, high = model.jnt_range[joint_id]
            data.qpos[address] = np.clip(data.qpos[address], low, high)

    def _osc_output_max(self) -> np.ndarray:
        """读取右臂 OSC 控制器每步输出的最大值(用于把位姿增量归一化到动作空间)。"""
        robot = self.env.unwrapped.robots[0]
        controller = getattr(robot, "composite_controller", None)
        parts = getattr(controller, "part_controllers", {})
        arm = parts.get("right") if isinstance(parts, dict) else None
        values = getattr(arm, "output_max", None)
        if values is None:
            values = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size < 6 or np.any(values[:6] <= 0):
            raise RuntimeError("invalid_osc_output_max")
        return values[:6]

    def close(self) -> None:
        """关闭底层环境并清空缓存配置。"""
        if self.env is not None:
            self.env.close()
            self.env = None
            self.env_config = None

    def _prepare_obs(self, obs: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """归一化键名(修掉相机/state 前缀不匹配)并把指令统一写入 state.instruction。"""
        obs = normalize_obs(obs)
        instr = obs.get("state.annotation.human.task_description", "")
        if not instr:
            instr = info.get("task_description", "")
        obs["state.instruction"] = instr
        return obs

    def _ensure_env(self, request: dict[str, Any]) -> None:
        """按请求与 scene 配置创建 gym 环境;配置未变则复用现有环境。"""
        scene = self.scene_config
        config = {
            "env_id": request["env_id"],
            "split": scene.get("split", "target"),
            "seed": int(request.get("seed", 0)),
            "camera_width": int(request.get("camera_width", 256)),
            "camera_height": int(request.get("camera_height", 256)),
            "robots": scene.get("robots", "PandaOmron"),
            "layout_ids": scene.get("layout_ids"),
            "style_ids": scene.get("style_ids"),
            "layout_and_style_ids": scene.get("layout_and_style_ids"),
            "use_novel_instructions": bool(scene.get("use_novel_instructions", False)),
            "render_quality": scene.get("render_quality", "high"),
        }
        if self.env is not None and self.env_config == config:
            return

        self.close()

        import gymnasium as gym
        import robocasa  # noqa: F401
        import robocasa.wrappers.gym_wrapper  # noqa: F401

        self.env = gym.make(
            config["env_id"],
            split=config["split"],
            seed=config["seed"],
            camera_widths=config["camera_width"],
            camera_heights=config["camera_height"],
            robots=config["robots"],
            layout_ids=config["layout_ids"],
            style_ids=config["style_ids"],
            layout_and_style_ids=config["layout_and_style_ids"],
            use_novel_instructions=config["use_novel_instructions"],
            enable_render=True,
        )
        _apply_render_quality(self.env, config["render_quality"])
        self.env_config = config

    def _action_spec(self) -> dict[str, Any]:
        """返回动作维度与各维含义;超出已知含义的维度用 dim{i} 占位。"""
        dim = None
        if self.env is not None:
            try:
                dim = int(self.env.action_space.shape[0])
            except Exception:
                dim = None
        dim = dim if dim is not None else len(_ROBOCASA_ACTION_MEANING)
        if dim <= len(_ROBOCASA_ACTION_MEANING):
            meaning = _ROBOCASA_ACTION_MEANING[:dim]
        else:
            meaning = _ROBOCASA_ACTION_MEANING + [
                f"dim{i}" for i in range(len(_ROBOCASA_ACTION_MEANING), dim)
            ]
        return {"dim": dim, "meaning": meaning}

    def _sim_info(self) -> dict[str, Any]:
        """返回仿真元信息:sim 名、机器人、控制器、env_id 与各相机投影。"""
        config = self.env_config or {}
        controller = ""
        cameras: dict[str, Any] = {}
        if self.env is not None:
            try:
                robots = self.env.unwrapped.robots
                controller = robots[0].composite_controller.__class__.__name__
            except Exception:
                controller = ""
            cameras = self._camera_projections()
        return {
            "sim": "robocasa",
            "robots": config.get("robots", "PandaOmron"),
            "controller": controller,
            "env_id": config.get("env_id", ""),
            "base_frame": "robot0_base",
            "cameras": cameras,
        }

    @staticmethod
    def _camera_names(model: Any) -> list[str]:
        """取 mujoco 模型里的全部相机名。
        robosuite 的 sim.model 是 binding_utils 包装类,底层 mujoco.MjModel 在 ._model,
        mj_id2name 只接受它(枚举类型需先转 int)。
        """
        base = getattr(model, "_model", None) or model
        if hasattr(base, "cam_names"):
            raw_names = base.cam_names
            return [(c.decode() if isinstance(c, bytes) else str(c)) for c in raw_names]
        import mujoco

        cam_type = getattr(mujoco.mjtObj, "mjOBJ_CAMERA", None)
        cam_type = int(cam_type) if cam_type is not None else 6
        return [mujoco.mj_id2name(base, cam_type, i) or "" for i in range(int(base.ncam))]

    def _camera_projections(self) -> dict[str, Any]:
        """按 robosuite camera_utils 约定计算各相机的内参与投影矩阵,供感知 executor 三角化定位。

        intrinsics: 3x3 内参(fx=fy=0.5*H/tan(fovy/2),主点在图像中心);
        projection: 3x4 base->像素投影矩阵,project(X)=P@[X;1],归一化后 col=x/z, row=y/z。
        投影矩阵已折算到机器人 base 系,三角化结果直接是 base 系坐标(与 raw executor 一致)。
        键为 mujoco 模型相机名,与 obs 的 video.* 键一致(robocasa 相机名即 model cam 名)。
        """
        config = self.env_config or {}
        height = int(config.get("camera_height", 256))
        width = int(config.get("camera_width", 256))
        sim = self.env.unwrapped.sim
        model = sim.model
        base_position, base_rotation = self._base_pose_world()
        out: dict[str, Any] = {}
        for cam_id, name in enumerate(self._camera_names(model)):
            if not name:
                continue
            try:
                fovy = float(model.cam_fovy[cam_id])
                f = 0.5 * height / np.tan(fovy * np.pi / 360)
                k = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
                cam_pos = sim.data.cam_xpos[cam_id]
                cam_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
                r_ext = np.eye(4)
                r_ext[:3, :3] = cam_rot
                r_ext[:3, 3] = cam_pos
                axis_correction = np.diag([1.0, -1.0, -1.0, 1.0])
                r_ext = r_ext @ axis_correction
                p = k @ np.linalg.inv(r_ext)[:3, :4]
                out[name] = {
                    "intrinsics": k.tolist(),
                    "projection": base_frame_projection(p, base_position, base_rotation),
                }
            except Exception as exc:
                print(f"[robocasa] camera {name} projection failed: {exc}", flush=True)
        return out


def _load_scene_config(path: str | None) -> dict[str, Any]:
    """读取 scene.yaml,兼容 robocasa_bridge.ros__parameters 段或顶层 dict。"""
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        document = yaml.safe_load(f) or {}
    params = document.get("robocasa_bridge", {}).get("ros__parameters", {})
    return params or (document if isinstance(document, dict) else {})


def _default_scene_config() -> str | None:
    """按相对位置定位 project 里的 config/scene.yaml,无 --scene-config 时兜底。"""
    here = Path(__file__).resolve()
    candidates = [here.parent.parent / "config" / "scene.yaml"]
    for parent in here.parents:
        if parent.name == "site-packages":
            candidates.append(
                parent.parent.parent / "share" / "nova_robocasa_bridge" / "config" / "scene.yaml"
            )
            break
    env_path = os.environ.get("NOVA_SCENE_CONFIG")
    if env_path:
        candidates.insert(0, Path(env_path))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def main() -> int:
    """命令行入口:加载 scene 配置、预热环境并启动 JSON 帧服务。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--scene-config",
        default=None,
        help="path to scene.yaml; defaults to the project's config/scene.yaml",
    )
    args = parser.parse_args()

    scene_path = args.scene_config or _default_scene_config()
    scene_config = _load_scene_config(scene_path)
    print(f"scene config: {scene_path or '(none)'}", flush=True)
    session = RoboCasaSession(scene_config)

    # 预热:提前加载场景/资产到缓存,让第一个 reset 秒回
    import time
    t0 = time.time()
    print("warming up environment (may take a while)...", flush=True)
    try:
        session.reset(
            {
                "env_id": scene_config.get("env_id", "robocasa/PickPlaceCounterToCabinet"),
                "seed": int(scene_config.get("seed", 0)),
                "camera_width": int(scene_config.get("camera_width", 512)),
                "camera_height": int(scene_config.get("camera_height", 512)),
            }
        )
    except Exception as exc:
        print(f"warmup failed (will lazily init on first reset): {exc}", flush=True)

    print(f"warmup done in {time.time() - t0:.1f}s", flush=True)
    return serve(args.host, args.port, session, "RoboCasa")


if __name__ == "__main__":
    raise SystemExit(main())
