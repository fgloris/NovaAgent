"""VLM 图像缩放、编解码与像素坐标换算的共享实现。

约定:显示分辨率 = 等比缩小(只缩不放)到 max_size;编码不裁剪,
所以 display 与 native 之间的像素坐标是纯线性缩放。
"""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

DEFAULT_MAX_IMAGE_SIZE = 768


def display_size(width: int, height: int, max_size: int = DEFAULT_MAX_IMAGE_SIZE) -> tuple[int, int]:
    """返回图像发送给 VLM 的显示分辨率 (w, h):等比缩小,只缩不放。"""
    width, height = int(width), int(height)
    scale = min(1.0, float(max_size) / max(width, height, 1))
    if scale < 1.0:
        return int(round(width * scale)), int(round(height * scale))
    return width, height


def scale_factors(src_size: tuple[int, int], dst_size: tuple[int, int]) -> tuple[float, float]:
    """从 src 分辨率到 dst 分辨率的 (sx, sy)。"""
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    return (dst_w / src_w if src_w else 1.0, dst_h / src_h if src_h else 1.0)


def convert_points(value: Any, src_size: tuple[int, int], dst_size: tuple[int, int]) -> Any:
    """把像素点从 src 分辨率换算到 dst 分辨率。

    支持三种形状:``[u, v]``、``[[u, v], ...]``、``{key: [u, v]}``;其它原样返回。
    """
    sx, sy = scale_factors(src_size, dst_size)

    def one(point):
        return [float(point[0]) * sx, float(point[1]) * sy]

    if isinstance(value, dict):
        return {key: one(point) for key, point in value.items()}
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return [one(point) for point in value]
        if len(value) == 2:
            return one(value)
    return value


def encode_data_url(image: np.ndarray, max_size: int = DEFAULT_MAX_IMAGE_SIZE, quality: int = 80) -> str:
    """numpy RGB 图等比缩小后编码为 ``data:image/jpeg;base64,...``。"""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    cw, ch = display_size(width, height, max_size)
    if (cw, ch) != (width, height):
        from PIL import Image as PILImage

        image = np.asarray(PILImage.fromarray(image, mode="RGB").resize((cw, ch)))
    buf = io.BytesIO()
    from PIL import Image as PILImage

    PILImage.fromarray(image, mode="RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
