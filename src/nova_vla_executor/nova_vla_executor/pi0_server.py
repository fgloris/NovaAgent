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
默认按 pi05_libero(observation/image 第三视角 + observation/wrist_image 腕部)配置;
换模型可用 --obs-key-map 覆盖,例如:
    python pi0_server.py --model pi0_droid \
      --obs-key-map agentview=observation/exterior_image_1_left \
                     robot0_eye_in_hand=observation/wrist_image_left
"""
import argparse
import asyncio
import json

import numpy as np

APP = None  # FastAPI app(延迟 import,避免 --help 也要 fastapi)

# 相机名 -> 模型 config 的 observation 键(pi05_libero / pi0_libero 默认布局)
DEFAULT_OBS_KEY_MAP = {
    "agentview": "observation/image",
    "robot0_eye_in_hand": "observation/wrist_image",
}


def _load_engine(checkpoint_dir: str, model_id: str):
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


def _action_from_output(out: dict) -> list[float]:
    action = np.asarray(out["actions"])
    if action.ndim == 2:
        action = action[0]
    return action.astype(float).tolist()


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
            for client_key, model_key in obs_key_map.items():
                if client_key in images:
                    obs[model_key] = images[client_key]
            state = header.get("state")
            if state is not None:
                obs["observation/state"] = np.asarray(state, dtype=np.float32)
            out = await loop.run_in_executor(None, lambda: policy.infer(obs))
            action = _action_from_output(out)
            await ws.send_json({"action": action, "action_dim": len(action)})

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
    parser.add_argument("--checkpoint", required=True, help="openpi checkpoint 目录(新版,含 params/ 或 model.safetensors)")
    parser.add_argument("--model", default="pi05_libero", help="openpi training config 名,默认 pi05_libero")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--obs-key-map",
        nargs="*",
        default=[],
        help='相机名=模型observation键,如 agentview=observation/image;可多个',
    )
    args = parser.parse_args()

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
