"""观测/动作的编码解码与归一化(无 ROS 依赖)。

由 sim server 与 bridge 共用,保证两端格式一致。
传输走帧式二进制协议(jsonline.py):numpy 数组转成 __blob__ 占位符,字节收集到帧 body,免 base64。
future annotations:兼容 python3.8 的 libero conda 环境。
"""
from __future__ import annotations
import re
from typing import Any

import numpy as np


def encode_frame(value: Any, blobs: list[bytes]) -> Any:
    """帧编码:把值里的 numpy 数组替换为 __blob__ 占位符,原始字节按序收集进 blobs。"""
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


def decode_frame(value: Any, blobs: list[bytes]) -> Any:
    """帧解码:按 header 里的 __blob__ 索引从 blobs 还原 numpy 数组,递归重建结构。"""
    if isinstance(value, dict) and "__blob__" in value:
        raw = blobs[int(value["__blob__"])]
        return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    if isinstance(value, dict):
        return {str(k): decode_frame(v, blobs) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_frame(v, blobs) for v in value]
    return value


def blob_size(value: Any) -> int:
    """计算帧 body 总字节数(所有 __blob__ 大小之和),用于读取定长 body。"""
    if isinstance(value, dict) and "__blob__" in value:
        return int(np.dtype(value["dtype"]).itemsize) * int(np.prod(value["shape"]))
    if isinstance(value, dict):
        return sum(blob_size(v) for v in value.values())
    if isinstance(value, list):
        return sum(blob_size(v) for v in value)
    return 0


def summarize_value(value: Any) -> Any:
    """压缩观测值用于 topic 发布:大数组只保留 shape/dtype/min/max,小数组转 list。"""
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


def _safe_camera_name(name: str) -> str:
    """把相机名里的非法字符(. 等)替换为 _,否则 ROS2 话题名会非法。"""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)

def normalize_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """把原始 obs 的键统一为 video./state. 前缀(幂等,兼容已带前缀的键)。

    规则:
      3D 且 shape[-1]==3 的数组 -> video.{key};若 key 以 "_image" 结尾则先剥掉;
      其余(含字符串/标量/低维数组) -> state.{key}。
    例如 LIBERO 的 agentview -> video.agentview;
    RoboCasa 的 agentview_image -> video.agentview(剥 _image,统一相机名)。
    """
    normalized: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
            if key.startswith("video."):
                name = key[len("video."):]
            else:
                name = key
                if name.endswith("_image"):
                    name = name[: -len("_image")]
            normalized[f"video.{_safe_camera_name(name)}"] = value
        elif key.startswith("state."):
            normalized[key] = value
        else:
            normalized[f"state.{key}"] = value
    return normalized


def spec_of(value: Any) -> dict[str, Any]:
    """返回单个值的规格描述:数组给出 shape/dtype,其它给出类型名。"""
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__}


def build_obs_spec(obs: dict[str, Any]) -> dict[str, Any]:
    """从归一化 obs 自省出 obs_spec:video.* 键归入 cameras,state.* 键归入 state。"""
    state: dict[str, Any] = {}
    cameras: dict[str, Any] = {}
    for key, value in obs.items():
        if key.startswith("video."):
            cameras[key[len("video."):]] = spec_of(value)
        elif key.startswith("state."):
            state[key[len("state."):]] = spec_of(value)
    return {"state": state, "cameras": cameras}
