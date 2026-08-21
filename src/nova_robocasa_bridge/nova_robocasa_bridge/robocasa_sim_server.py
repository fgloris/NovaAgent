#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import socketserver
import traceback
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np


def encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(v) for v in value]
    return value


def action_to_numpy(action: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value, dtype=np.float32)
        for key, value in action.items()
    }


class RoboCasaSession:
    def __init__(self) -> None:
        self.env = None
        self.env_config: dict[str, Any] | None = None

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        self._ensure_env(request)
        assert self.env is not None
        obs, info = self.env.reset(seed=int(request.get("seed", 0)))
        return {"ok": True, "obs": encode_value(obs), "info": encode_value(info)}

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("environment is not initialized; call reset first")
        action = action_to_numpy(request["action"])
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {
            "ok": True,
            "obs": encode_value(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": encode_value(info),
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self.env_config = None

    def _ensure_env(self, request: dict[str, Any]) -> None:
        config = {
            "env_id": request["env_id"],
            "split": request.get("split", "target"),
            "seed": int(request.get("seed", 0)),
            "camera_width": int(request.get("camera_width", 256)),
            "camera_height": int(request.get("camera_height", 256)),
            "robots": request.get("robots", "PandaOmron"),
            "layout_ids": request.get("layout_ids"),
            "style_ids": request.get("style_ids"),
            "layout_and_style_ids": request.get("layout_and_style_ids"),
            "use_novel_instructions": bool(request.get("use_novel_instructions", False)),
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
        self.env_config = config


class RoboCasaRequestHandler(socketserver.StreamRequestHandler):
    session = RoboCasaSession()

    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                request = json.loads(line.decode("utf-8"))
                response = self.dispatch(request)
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            self.wfile.write(json.dumps(response, separators=(",", ":")).encode("utf-8"))
            self.wfile.write(b"\n")
            self.wfile.flush()

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type == "reset":
            return self.session.reset(request)
        if request_type == "step":
            return self.session.step(request)
        if request_type == "close":
            self.session.close()
            return {"ok": True}
        raise ValueError(f"unknown request type: {request_type!r}")


class ThreadingTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    with ThreadingTcpServer((args.host, args.port), RoboCasaRequestHandler) as server:
        print(f"RoboCasa sim server listening on {args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            RoboCasaRequestHandler.session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
