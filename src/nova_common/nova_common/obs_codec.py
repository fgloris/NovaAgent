# 观测/动作的编码解码与归一化(无 ROS 依赖)。
# 由 sim server 与 bridge 共用,保证两端格式一致。
# 传输走帧式二进制协议(jsonline.py):numpy 数组转成 __blob__ 占位符,字节收集到帧 body,免 base64。
# future annotations:兼容 python3.8 的 libero conda 环境
from __future__ import annotations
from typing import Any

import numpy as np


# 帧编码:值里的 numpy 数组替换为 __blob__ 占位符,原始字节收集进 blobs
def encode_frame(value: Any, blobs: list[bytes]) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        blobs.append(array.tobytes())
        return {"__blob__": len(blobs) - 1, "dtype": str(array.dtype), "shape": list(array.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode_frame(v, blobs) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_frame(v, blobs) for v in value]
    return value


# 帧解码:按 header 里的 __blob__ 顺序切分 body,还原为 numpy 数组
def decode_frame(value: Any, blobs: list[bytes]) -> Any:
    if isinstance(value, dict) and "__blob__" in value:
        raw = blobs[int(value["__blob__"])]
        return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    if isinstance(value, dict):
        return {str(k): decode_frame(v, blobs) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_frame(v, blobs) for v in value]
    return value


# 帧 body 总字节数(所有 __blob__ 大小之和,用于读取)
def blob_size(value: Any) -> int:
    if isinstance(value, dict) and "__blob__" in value:
        return int(np.dtype(value["dtype"]).itemsize) * int(np.prod(value["shape"]))
    if isinstance(value, dict):
        return sum(blob_size(v) for v in value.values())
    if isinstance(value, list):
        return sum(blob_size(v) for v in value)
    return 0


# 压缩观测值用于 topic 发布,大数组只保留 shape/dtype/min/max
def summarize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size > 32:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(np.nanmin(value)) if value.size else 0.0,
                "max": float(np.nanmax(value)) if value.size else 0.0,
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): summarize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize_value(v) for v in value]
    return value


# 键名统一规则:
#   3D 且 shape[-1]==3 的数组 -> video.{key};若 key 以 "_image" 结尾则先剥掉
#   其余(含字符串/标量/低维数组) -> state.{key}
# LIBERO 键 agentview      -> video.agentview
# RoboCasa 键 agentview_image -> video.agentview(剥 _image,相机名统一)
def normalize_obs(obs: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
            name = key
            if name.endswith("_image"):
                name = name[: -len("_image")]
            normalized[f"video.{name}"] = value
        else:
            normalized[f"state.{key}"] = value
    return normalized


def spec_of(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__}


# 从归一化 obs 自省 obs_spec:video 键 -> cameras;state 键 -> state
def build_obs_spec(obs: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    cameras: dict[str, Any] = {}
    for key, value in obs.items():
        if key.startswith("video."):
            cameras[key[len("video."):]] = spec_of(value)
        elif key.startswith("state."):
            state[key[len("state."):]] = spec_of(value)
    return {"state": state, "cameras": cameras}
