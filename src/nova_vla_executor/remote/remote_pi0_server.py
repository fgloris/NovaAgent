#!/usr/bin/env python3
"""pi0 推理 HTTP server —— 部署在 4090 服务器(不需要安装 ROS2)。

用法:
    pip install fastapi uvicorn requests numpy openpi
    python remote_pi0_server.py --checkpoint /path/to/pi0_checkpoint --host 0.0.0.0 --port 8001

请求(POST /predict):
    {
      "images": {
        "agentview": {"data": "<base64 原始 RGB>", "shape": [256, 256, 3]},
        "robot0_eye_in_hand": {"data": "...", "shape": [256, 256, 3]}
      },
      "instruction": "open the drawer",
      "state": [0.1, 0.2, 0.3]          # 可选,维度以 checkpoint config 为准
    }
响应:
    {"action": [0.1, ...], "action_dim": 7}

说明:
- openpi 的观测键为 image_primary / image_wrist / state / prompt,
  agentview -> image_primary,robot0_eye_in_hand -> image_wrist。
- 若你的 checkpoint 是 safetensors(HF 格式),需先转成 openpi 格式,
  见 openpi 仓库 scripts/convert_checkpoint.py 或 README。
- 如果 openpi 版本 API 有变化,只改 Pi0Engine 里的加载/推理两处即可。
"""
import argparse
import base64
import io

import numpy as np

APP = None  # FastAPI app(延迟 import,避免 --help 也要 fastapi)


def _load_engine(checkpoint_path: str, model_id: str):
    from openpi.policies import policy_config

    config = policy_config.load_policy_config(model_id)
    policy = policy_config.create_inference_fn(config, checkpoint_path=checkpoint_path)
    print(f"[pi0] loaded checkpoint: {checkpoint_path}", flush=True)
    return policy


def _decode_image(payload: dict) -> np.ndarray:
    data = base64.b64decode(payload["data"])
    shape = [int(x) for x in payload["shape"]]
    return np.frombuffer(data, dtype=np.uint8).reshape(shape)


def build_app(checkpoint_path: str, model_id: str):
    import json

    import fastapi
    from pydantic import BaseModel

    policy = _load_engine(checkpoint_path, model_id)

    class PredictReq(BaseModel):
        images: dict[str, str] = {}
        instruction: str = ""
        state: list[float] | None = None

    app = fastapi.FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/predict")
    def predict(req: PredictReq):
        obs = {"prompt": req.instruction}
        if "agentview" in req.images:
            obs["image_primary"] = _decode_image(req.images["agentview"])
        if "robot0_eye_in_hand" in req.images:
            obs["image_wrist"] = _decode_image(req.images["robot0_eye_in_hand"])
        if req.state is not None:
            obs["state"] = np.asarray(req.state, dtype=np.float32)
        out = policy.infer(obs, action_horizon=1)
        action = np.asarray(out["action"])
        if action.ndim == 2:
            action = action[0]
        action = action.astype(float).tolist()
        return {"action": action, "action_dim": len(action)}

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="pi0 推理 HTTP server(无 ROS2)")
    parser.add_argument("--checkpoint", required=True, help="openpi 格式的 pi0 checkpoint 路径")
    parser.add_argument("--model", default="pi0", help="openpi 模型 id,默认 pi0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    global APP
    APP = build_app(args.checkpoint, args.model)

    import uvicorn

    print(f"[pi0] serving on http://{args.host}:{args.port}/predict", flush=True)
    uvicorn.run(APP, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
