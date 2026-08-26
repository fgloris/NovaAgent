#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np


# 直接 python 运行时(未 source install/setup.bash)也能找到 nova_common 包
def _ensure_nova_common_importable() -> None:
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


def action_vector_to_dict(values: np.ndarray) -> dict[str, list[float]]:
    values = np.ravel(values).astype(np.float32)
    if values.size > len(_ROBOCASA_ACTION_MEANING):
        raise ValueError(
            f"expected at most {len(_ROBOCASA_ACTION_MEANING)} action values, got {values.size}"
        )
    if values.size < len(_ROBOCASA_ACTION_MEANING):
        values = np.pad(values, (0, len(_ROBOCASA_ACTION_MEANING) - values.size))
    values = np.clip(values, -1.0, 1.0)

    return {
        "action.end_effector_position": values[0:3].tolist(),
        "action.end_effector_rotation": values[3:6].tolist(),
        "action.gripper_close": [1.0 if values[6] >= 0.5 else 0.0],
        "action.base_motion": values[7:11].tolist(),
        "action.control_mode": [1.0 if values[11] >= 0.5 else 0.0],
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
    def __init__(self, scene_config: dict[str, Any] | None = None) -> None:
        self.scene_config = scene_config or {}
        self.env = None
        self.env_config: dict[str, Any] | None = None

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        self._ensure_env(request)
        assert self.env is not None
        obs, info = self.env.reset(seed=int(request.get("seed", 0)))
        obs = self._prepare_obs(obs, info)
        return {
            "ok": True,
            "obs": obs,
            "info": info,
            "action_spec": self._action_spec(),
            "obs_spec": build_obs_spec(obs),
            "sim_info": self._sim_info(),
        }

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("environment is not initialized; call reset first")
        action = action_vector_to_dict(np.asarray(request["action"], dtype=np.float32))
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._prepare_obs(obs, info)
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
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self.env_config = None

    # 归一化键名(修掉相机/state 前缀不匹配)+ 指令统一写入 state.instruction
    def _prepare_obs(self, obs: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        obs = normalize_obs(obs)
        instr = obs.get("state.annotation.human.task_description", "")
        if not instr:
            instr = info.get("task_description", "")
        obs["state.instruction"] = instr
        return obs

    def _ensure_env(self, request: dict[str, Any]) -> None:
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
        config = self.env_config or {}
        return {
            "sim": "robocasa",
            "robots": config.get("robots", "PandaOmron"),
            "controller": "",
            "env_id": config.get("env_id", ""),
        }


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
    return serve(args.host, args.port, RoboCasaSession(scene_config), "RoboCasa")


if __name__ == "__main__":
    raise SystemExit(main())
