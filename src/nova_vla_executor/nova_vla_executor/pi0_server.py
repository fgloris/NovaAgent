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
默认按 robocasa(pi0_robocasa_pretrain_human300,observation/image 第三视角 + observation/wrist_image 腕部)配置;
换模型可用 --obs-key-map 覆盖,例如:
    python pi0_server.py --model pi0_droid \
      --obs-key-map agentview=observation/exterior_image_1_left \
                     robot0_eye_in_hand=observation/wrist_image_left
"""
import argparse
import asyncio
import json
import os
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
            obs = {"prompt": header.get("instruction", "")}
            # 客户端相机名 -> 模型 observation 键:精确匹配优先,其次子串匹配
            # (实际相机名可能是 video_robot0_agentview_left,而配置键是 agentview)
            for client_key, model_key in obs_key_map.items():
                matched = [k for k in images if k == client_key] or [k for k in images if client_key in k]
                if not matched:
                    continue
                obs[model_key] = images[matched[0]]
                print(f"[pi0] obs map: {matched[0]} -> {model_key}", flush=True)
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
    parser = argparse.ArgumentParser(description="pi0 推理 server(新版 openpi,无 ROS2,WebSocket 二进制)")
    parser.add_argument("--checkpoint", default=os.environ.get("ROBOCASA_CHECKPOINT_PATH"), help="openpi checkpoint 目录(新版,含 params/ 或 model.safetensors);默认取环境变量 ROBOCASA_CHECKPOINT_PATH")
    parser.add_argument("--model", default="pi0_robocasa_pretrain_human300", help="openpi training config 名,默认 pi0_robocasa_pretrain_human300")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--obs-key-map",
        nargs="*",
        default=[],
        help='相机名=模型observation键,如 agentview=observation/image;可多个',
    )
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint 未提供,且环境变量 ROBOCASA_CHECKPOINT_PATH 未设置")
    args.checkpoint = os.path.expanduser(args.checkpoint)

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
