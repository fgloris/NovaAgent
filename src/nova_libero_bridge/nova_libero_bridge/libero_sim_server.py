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
from nova_common.obs_codec import build_obs_spec, encode_value, normalize_obs


# LIBERO 源码以 namespace 包形式安装(libero/libero),需要把 LIBERO 根目录加入 sys.path。
# 优先用 LIBERO_ROOT 环境变量,否则从 ~/.libero/config.yaml 的 benchmark_root 向上推断。
def _find_libero_root() -> str | None:
    env_root = os.environ.get("LIBERO_ROOT")
    if env_root:
        return env_root
    config = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")) / "config.yaml"
    if config.is_file():
        import yaml

        with open(config, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        root = Path(doc.get("benchmark_root", ""))
        # benchmark_root = <LIBERO>/libero/libero,namespace 根在 <LIBERO>
        if root.is_dir() and (root.parent.parent / "libero").is_dir():
            return str(root.parent.parent)
    return None


def _ensure_libero_importable() -> None:
    root = _find_libero_root()
    if root:
        sys.path.insert(0, root)


_ensure_libero_importable()


# 基于 controller 生成动作布局约定(维度优先从 env.action_space 自省)
_CONTROLLER_ACTION_MEANING = {
    "OSC_POSE": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
    "IK_POSE": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
    "OSC_POSITION": ["dx", "dy", "dz", "gripper"],
}


def _build_action_spec(controller: str, dim_override: int | None) -> dict[str, Any]:
    meaning = list(_CONTROLLER_ACTION_MEANING.get(controller, []))
    dim = dim_override if dim_override is not None else (len(meaning) or 7)
    if not meaning:
        meaning = [f"dim{i}" for i in range(dim)]
    elif len(meaning) != dim:
        if len(meaning) > dim:
            meaning = meaning[:dim]
        else:
            meaning = meaning + [f"dim{i}" for i in range(len(meaning), dim)]
    return {"dim": dim, "meaning": meaning}


class LiberoSession:
    def __init__(self, scene_config: dict[str, Any] | None = None) -> None:
        self.scene_config = scene_config or {}
        self.env = None
        self.benchmark = None
        self.task = None
        self.env_config: dict[str, Any] | None = None
        self.language_instruction = ""
        self.init_mode = str(self.scene_config.get("init_mode", "random"))
        self.init_state_index = int(self.scene_config.get("init_state_index", 0))
        self.task_id = int(self.scene_config.get("task_id", 0))
        if self.init_mode not in ("random", "fixed"):
            raise ValueError(f"init_mode must be 'random' or 'fixed', got {self.init_mode!r}")

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        self._ensure_env(request)
        assert self.env is not None

        seed = int(request.get("seed", 0))
        self.env.seed(seed)

        # init_mode: random 用 env.reset() 随机采样;fixed 加载固定 pruned init states
        if self.init_mode == "fixed":
            assert self.benchmark is not None
            init_states = self.benchmark.get_task_init_states(self.task_id)
            obs = self.env.set_init_state(init_states[self.init_state_index])
            info: dict[str, Any] = {}
        else:
            ret = self.env.reset()
            # LIBERO 的旧 gym API reset 可能返回 obs 或 (obs, info),统一为 obs
            if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict):
                obs, info = ret
            else:
                obs, info = ret, {}

        success = bool(self.env.check_success())
        info["success"] = success
        obs = normalize_obs(obs)
        obs["state.instruction"] = self.language_instruction
        return {
            "ok": True,
            "obs": encode_value(obs),
            "info": encode_value(info),
            "action_spec": self._action_spec(),
            "obs_spec": build_obs_spec(obs),
            "sim_info": self._sim_info(),
        }

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("environment is not initialized; call reset first")
        action = np.asarray(request["action"], dtype=np.float32)
        # LIBERO 使用旧 gym API: obs, reward, done, info
        obs, reward, done, info = self.env.step(action)
        success = bool(self.env.check_success())
        info = dict(info)
        info["success"] = success
        obs = normalize_obs(obs)
        obs["state.instruction"] = self.language_instruction
        return {
            "ok": True,
            "obs": encode_value(obs),
            "reward": float(reward),
            "terminated": bool(done),
            "truncated": False,
            "info": encode_value(info),
            "action_spec": self._action_spec(),
            "obs_spec": build_obs_spec(obs),
            "sim_info": self._sim_info(),
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self.benchmark = None
            self.task = None
            self.env_config = None

    def _ensure_env(self, request: dict[str, Any]) -> None:
        scene = self.scene_config
        config = {
            "benchmark": scene.get("benchmark", request.get("benchmark", "libero_spatial")),
            "task_id": int(scene.get("task_id", request.get("task_id", 0))),
            "seed": int(request.get("seed", 0)),
            "camera_heights": int(request.get("camera_height", 256)),
            "camera_widths": int(request.get("camera_width", 256)),
            "camera_names": scene.get("camera_names", ["agentview", "robot0_eye_in_hand"]),
            "robots": scene.get("robots", ["Panda"]),
            "controller": scene.get("controller", "OSC_POSE"),
            "action_dim": int(scene.get("action_dim", 7)),
            "renderer": scene.get("renderer", "mujoco"),
        }
        if self.env is not None and self.env_config == config:
            return

        self.close()

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bench = benchmark.get_benchmark_dict()[config["benchmark"]]()
        task = bench.get_task(config["task_id"])
        bddl_file = bench.get_task_bddl_file_path(config["task_id"])
        assert Path(bddl_file).exists(), f"bddl file not found: {bddl_file}"

        env_args = {
            "bddl_file_name": bddl_file,
            "robots": config["robots"],
            "controller": config["controller"],
            "camera_heights": config["camera_heights"],
            "camera_widths": config["camera_widths"],
            "camera_names": config["camera_names"],
        }
        if config["renderer"] != "mujoco":
            env_args["renderer"] = config["renderer"]

        self.env = OffScreenRenderEnv(**env_args)
        self.benchmark = bench
        self.task = task
        self.task_id = config["task_id"]
        self.env_config = config
        self.language_instruction = str(task.language)
        print(
            f"[libero] benchmark={config['benchmark']} task_id={config['task_id']} "
            f"task={task.name} instruction={task.language}",
            flush=True,
        )
        print(
            f"[libero] init_mode={self.init_mode} init_state_index={self.init_state_index}",
            flush=True,
        )

    def _action_spec(self) -> dict[str, Any]:
        dim = None
        if self.env is not None:
            try:
                dim = int(self.env.action_space.shape[0])
            except Exception:
                dim = None
        controller = (self.env_config or {}).get("controller", "OSC_POSE")
        return _build_action_spec(controller, dim)

    def _sim_info(self) -> dict[str, Any]:
        config = self.env_config or {}
        return {
            "sim": "libero",
            "robots": config.get("robots", ["Panda"]),
            "controller": config.get("controller", "OSC_POSE"),
            "benchmark": config.get("benchmark", ""),
            "task_id": config.get("task_id", 0),
        }


def _load_scene_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        document = yaml.safe_load(f) or {}
    params = document.get("libero_bridge", {}).get("ros__parameters", {})
    return params or (document if isinstance(document, dict) else {})


def _default_scene_config() -> str | None:
    here = Path(__file__).resolve()
    candidates = [here.parent.parent / "config" / "scene.yaml"]
    for parent in here.parents:
        if parent.name == "site-packages":
            candidates.append(
                parent.parent.parent / "share" / "nova_libero_bridge" / "config" / "scene.yaml"
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
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--scene-config",
        default=None,
        help="path to scene.yaml; defaults to the project's config/scene.yaml",
    )
    args = parser.parse_args()

    scene_path = args.scene_config or _default_scene_config()
    scene_config = _load_scene_config(scene_path)
    print(f"scene config: {scene_path or '(none)'}", flush=True)
    return serve(args.host, args.port, LiberoSession(scene_config), "LIBERO")


if __name__ == "__main__":
    raise SystemExit(main())
