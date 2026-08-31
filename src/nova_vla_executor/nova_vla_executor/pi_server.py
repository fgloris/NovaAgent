#!/usr/bin/env python3
"""pi0 推理 HTTP server —— 部署在 GPU 服务器(如 A6000,不需要安装 ROS2)。

通信:WebSocket /predict,二进制帧(免 base64)。
客户端 -> 服务端(二进制):
    首行 JSON header + 图像原始字节拼接。header:
    {
      "instruction": "open the drawer",
      "state": [0.1, 0.2, 0.3],            # 可选
      "images": {"agentview": {"shape": [256,256,3], "dtype": "uint8"}, ...}
    }
服务端 -> 客户端(JSON):
    {"action": [0.1, ...], "action_dim": 7}

观测键映射:客户端发的相机名 -> 模型 config 的 observation 键。
默认按 robocasa(pi0_robocasa_pretrain_human300, observation/image 第三视角 + observation/wrist_image 腕部)配置;
换模型可用 --obs-key-map 覆盖,例如:
    python pi_server.py --model pi0_droid \
      --obs-key-map agentview=observation/exterior_image_1_left \
                     robot0_eye_in_hand=observation/wrist_image_left
"""
import argparse
import asyncio
import json
import os
import re
import time

import numpy as np

APP = None  # FastAPI app(延迟 import,避免 --help 也要 fastapi)

# 相机名 -> 模型 config 的 observation 键。
# 客户端相机名 = env 实际相机名(robocasa groot gym wrapper),模型键 = 训练时
# groot_openpi_dataset 的键:observation/image(=robot0_agentview_left)、
# observation/wrist_image(=robot0_eye_in_hand)、observation/right_image(=robot0_agentview_right,PI0 忽略)。
DEFAULT_OBS_KEY_MAP = {
    "robot0_agentview_left": "observation/image",
    "robot0_agentview_right": "observation/right_image",
    "robot0_eye_in_hand": "observation/wrist_image",
}

# vla.yaml 里的 vla_model 取值 -> openpi training config 名
VLA_MODEL_IDS = {
    "pi0": "pi0_robocasa_pretrain_human300",
    "pi05": "pi05_pretrain_human300",
}


def _expand_env(s: str) -> str:
    """展开 ${ENV_VAR} 占位符;未设置的环境变量展开为空串。"""
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), s)


def _default_config_path() -> str | None:
    """自动定位 vla.yaml:优先同包 config,其次当前工作目录下的 src/nova_vla_executor/config/vla.yaml。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "config", "vla.yaml"),
        os.path.join(os.getcwd(), "src", "nova_vla_executor", "config", "vla.yaml"),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(c):
            return c
    return None


def _resolve_model(checkpoint: str | None, model: str, config_path: str | None) -> tuple[str | None, str]:
    """优先读 vla.yaml 的 vla_model 选择 checkpoint+模型;失败则回退命令行/环境变量。"""
    if config_path and os.path.isfile(config_path):
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
            params = ((doc.get("nova_vla_executor") or {}).get("ros__parameters") or {})
            vla_model = str(params.get("vla_model", "") or "").strip().lower()
            if vla_model in VLA_MODEL_IDS:
                ckpt = _expand_env(str(params.get(f"{vla_model}_checkpoint", "") or "")).strip()
                if ckpt:
                    checkpoint = os.path.expanduser(ckpt)
                    model = VLA_MODEL_IDS[vla_model]
                    print(
                        f"[pi0] config {config_path}: vla_model={vla_model} -> model={model}, checkpoint={checkpoint}",
                        flush=True,
                    )
                    return checkpoint, model
                print(f"[pi0] warn: vla_model={vla_model} 但 {vla_model}_checkpoint 为空({config_path})", flush=True)
        except Exception as exc:  # noqa: WPS462
            print(f"[pi0] warn: 解析 config {config_path} 失败,回退命令行/环境变量: {exc}", flush=True)
    return checkpoint, model


def _ensure_robocasa_stub():
    """robocasa-benchmark fork 的 openpi 在 import 阶段无条件 import robocasa(仅为注册 robocasa 训练配置),
    而真 robocasa 硬锁 numpy==2.2.5 / mujoco==3.3.1 / robosuite>=1.5.2,与 openpi(numpy<2)冲突,无法同环境共存。
    推理不碰 robocasa 数据(norm stats 从 checkpoint 的 assets/norm_stats.json 加载),
    这里注入最小 stub 让 import 链通过;若环境里真装了 robocasa 则跳过。
    """
    import sys
    import types

    try:
        import robocasa  # noqa: F401

        return
    except ImportError:
        pass

    def _module(name):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
        return m

    _module("robocasa")
    macros = _module("robocasa.macros")
    macros.DATASET_BASE_PATH = ""
    _module("robocasa.utils")
    registry = _module("robocasa.utils.dataset_registry")
    registry.DATASET_SOUP_REGISTRY = {
        "target50": [],
        "target_atomic_seen": [],
        "target_composite_seen": [],
        "target_composite_unseen": [],
        "pretrain_human300": [],
        "pretrain_human300_mg60": [],
    }
    registry.get_ds_meta = lambda **kwargs: None
    _module("robocasa.utils.groot_utils")
    tags = _module("robocasa.utils.groot_utils.embodiment_tags")
    tags.EmbodimentTag = type("EmbodimentTag", (), {"NEW_EMBODIMENT": "new_embodiment"})
    ds = _module("robocasa.utils.groot_utils.groot_dataset")
    ds.LeRobotSingleDataset = type("LeRobotSingleDataset", (), {})
    ds.LeRobotMixtureDataset = type("LeRobotMixtureDataset", (), {})
    ds.ModalityConfig = type("ModalityConfig", (), {})
    ds.LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
    ds.LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"


def _load_engine(checkpoint_dir: str, model_id: str):
    _ensure_robocasa_stub()
    from openpi.policies import policy_config
    from openpi.training import config as _config

    config = _config.get_config(model_id)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    print(f"[pi0] loaded checkpoint: {checkpoint_dir} (model={model_id})", flush=True)
    return policy


# 按 header["images"] 顺序切分 body,还原为 numpy 数组
def _split_body(header: dict, body: bytes) -> dict[str, np.ndarray]:
    images = {}
    offset = 0
    for name, meta in (header.get("images") or {}).items():
        size = int(np.dtype(meta["dtype"]).itemsize) * int(np.prod(meta["shape"]))
        arr = np.frombuffer(body[offset:offset + size], dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
        offset += size
        images[name] = arr
    return images


def _action_chunk_from_output(out: dict) -> np.ndarray:
    """返回完整动作 chunk (horizon, action_dim);单行时补成 2D。"""
    action = np.asarray(out["actions"], dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    return action


def build_app(checkpoint_dir: str, model_id: str, obs_key_map: dict[str, str]):
    import fastapi
    from fastapi import WebSocket, WebSocketDisconnect

    policy = _load_engine(checkpoint_dir, model_id)

    app = fastapi.FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.websocket("/predict")
    async def predict(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_event_loop()
        while True:
            try:
                message = await ws.receive_bytes()
            except WebSocketDisconnect:
                return
            header_b, _, body = message.partition(b"\n")
            header = json.loads(header_b.decode("utf-8"))
            images = _split_body(header, body)
            instruction = header.get("instruction", "")
            print(f"[pi0] instruction: {instruction!r}", flush=True)
            obs = {"prompt": instruction}
            # 客户端相机名 -> 模型 observation 匹配。
            for client_key, model_key in obs_key_map.items():
                if client_key not in images:
                    raise KeyError(
                        f"missing image key {client_key!r}; available image keys={list(images.keys())}"
                    )
                obs[model_key] = images[client_key]
                print(f"[pi0] obs map: {client_key} -> {model_key}", flush=True)
            state = header.get("state")
            if state is not None:
                obs["observation/state"] = np.asarray(state, dtype=np.float32)
            print(f"[pi0] obs keys: {list(obs.keys())}, images keys: {list(images.keys())}", flush=True)
            t0 = time.perf_counter()
            out = await loop.run_in_executor(None, lambda: policy.infer(obs))
            chunk = _action_chunk_from_output(out)
            infer_ms = (out.get("policy_timing") or {}).get("infer_ms", float("nan"))
            total_ms = (time.perf_counter() - t0) * 1000
            print(
                f"[pi0] infer model={infer_ms:.0f}ms total={total_ms:.0f}ms chunk={chunk.shape}",
                flush=True,
            )
            await ws.send_json(
                {
                    "action": chunk[0].astype(float).tolist(),
                    "action_chunk": chunk.astype(float).tolist(),
                    "action_dim": int(chunk.shape[1]),
                }
            )

    return app


def _parse_obs_key_map(items: list[str]) -> dict[str, str]:
    mapping = dict(DEFAULT_OBS_KEY_MAP)
    for item in items:
        if "=" in item:
            key, value = item.split("=", 1)
            mapping[key] = value
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="pi0/pi0.5 推理 server(robocasa/openpi环境,无 ROS2,WebSocket 二进制)")
    parser.add_argument(
        "--config",
        default=None,
        help="vla.yaml 路径,内含 vla_model(pi0|pi05)与 pi0_checkpoint/pi05_checkpoint(${ENV} 自动展开);"
        "默认自动定位 src/nova_vla_executor/config/vla.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="openpi checkpoint 目录(新版,含 params/ 或 model.safetensors);"
        "默认由 vla.yaml 的 vla_model 决定取 ROBOCASA_PI0_CHECKPOINT / ROBOCASA_PI05_CHECKPOINT,"
        "兜底取 ROBOCASA_CHECKPOINT_PATH",
    )
    parser.add_argument(
        "--model",
        default="pi0_robocasa_pretrain_human300",
        help="openpi training config 名;有 vla.yaml 的 vla_model 时自动覆盖为 pi0_robocasa_pretrain_human300 / pi05_pretrain_human300",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--obs-key-map",
        nargs="*",
        default=[],
        help='相机名=模型observation键,如 agentview=observation/image;可多个',
    )
    args = parser.parse_args()
    config_path = args.config or _default_config_path()
    args.checkpoint, args.model = _resolve_model(args.checkpoint, args.model, config_path)
    if args.checkpoint:
        args.checkpoint = os.path.expanduser(args.checkpoint)
    else:
        # 没有 vla.yaml / 未指定时,按模型类型选环境变量
        env_key = f"ROBOCASA_{args.model.upper()}_CHECKPOINT"
        args.checkpoint = os.environ.get(env_key) or os.environ.get("ROBOCASA_CHECKPOINT_PATH")
    if not args.checkpoint:
        parser.error(
            "--checkpoint 未提供,且环境变量 ROBOCASA_PI0_CHECKPOINT / ROBOCASA_PI05_CHECKPOINT "
            "(或旧 ROBOCASA_CHECKPOINT_PATH)未设置,且 vla.yaml 未配置"
        )

    global APP
    obs_key_map = _parse_obs_key_map(args.obs_key_map)
    print(f"[pi0] obs key map: {obs_key_map}", flush=True)
    APP = build_app(args.checkpoint, args.model, obs_key_map)

    import uvicorn

    print(f"[pi0] serving on ws://{args.host}:{args.port}/predict", flush=True)
    uvicorn.run(APP, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
