"""VLM 图像缩放、编解码与像素坐标换算的共享实现。

约定:显示分辨率 = 等比缩小(只缩不放)到 max_size;编码不裁剪,
所以 display 与 native 之间的像素坐标是纯线性缩放。
"""
from __future__ import annotations

import base64
import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MAX_IMAGE_SIZE = 768

# 图像记忆的统一引用形式:file://<kind>/<file>.jpg,executor 用隐藏的根目录解析
IMAGE_URL_PREFIX = "file://"
_JPEG_COM = b"\xff\xfe"


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


# ---------- 图像记忆:file:// 引用、JPEG 元信息读写 ----------


def image_url(kind: str, filename: str) -> str:
    """构造模型可见的图像引用 ``file://<kind>/<filename>``。"""
    return f"{IMAGE_URL_PREFIX}{kind}/{filename}"


def resolve_image_path(url: str, root: str | Path | None = None) -> Path:
    """把 ``file://<kind>/<file>`` 或普通路径解析为文件系统路径。

    指定 root 时只允许解析到 root 之内,防止越界读取。
    """
    text = str(url).strip()
    if text.startswith("data:"):
        raise ValueError("不支持 data URL,请使用 file:// 引用")
    if text.startswith(IMAGE_URL_PREFIX):
        text = text[len(IMAGE_URL_PREFIX):]
    path = Path(text)
    if root is None:
        return path
    base = Path(root).expanduser().resolve()
    resolved = (base / text.lstrip("/")).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"图像路径越界: {url!r}")
    return resolved


def load_image(url: str, root: str | Path | None = None) -> np.ndarray:
    """读取 ``file://`` 引用的图像为 RGB HxWx3 uint8 数组。"""
    from PIL import Image as PILImage

    path = resolve_image_path(url, root)
    with PILImage.open(path) as image:
        return np.asarray(image.convert("RGB"))


def data_url_to_bytes(url: str) -> bytes:
    """把 ``data:...;base64,...`` 拆成原始字节。"""
    if ";base64," not in url:
        raise ValueError("不是 base64 data URL")
    return base64.b64decode(url.split(";base64,", 1)[1])


def file_to_data_url(path: str | Path) -> str:
    """把磁盘上的 JPEG 直接编码为 data URL(不重新压缩)。"""
    data = Path(path).read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def downscale(image: np.ndarray, max_size: int) -> np.ndarray:
    """等比缩小到最长边不超过 max_size(只缩不放),返回 uint8 RGB。"""
    height, width = image.shape[:2]
    cw, ch = display_size(width, height, max_size)
    if (cw, ch) == (width, height):
        return np.asarray(image, dtype=np.uint8)
    from PIL import Image as PILImage

    resized = PILImage.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").resize(
        (cw, ch), PILImage.LANCZOS
    )
    return np.asarray(resized)


def _strip_jpeg_comments(data: bytes) -> bytes:
    """删除 JPEG 中已有的 COM 段,避免重复写入。"""
    if data[:2] != b"\xff\xd8":
        return data
    out = bytearray(data[:2])
    i = 2
    n = len(data)
    while i + 1 < n:
        if data[i] != 0xFF:
            out += data[i:]
            break
        marker = data[i + 1]
        if marker == 0xD8:  # 嵌套 SOI
            out += data[i:i + 2]
            i += 2
            continue
        if marker == 0xDA:  # SOS:之后是压缩数据,原样保留
            out += data[i:]
            break
        if i + 3 >= n:
            out += data[i:]
            break
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        segment = data[i:i + 2 + seg_len]
        if marker != 0xFE:  # 保留非 COM 段
            out += segment
        i += 2 + seg_len
    return bytes(out)


def inject_jpeg_metadata(data: bytes, metadata: dict) -> bytes:
    """把 metadata 以 JSON 形式写入 JPEG 的 COM 段(替换已有 COM)。"""
    payload = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    if len(payload) + 2 > 0xFFFF:
        payload = payload[:0xFFFF - 2]
    segment = _JPEG_COM + struct.pack(">H", len(payload) + 2) + payload
    stripped = _strip_jpeg_comments(data)
    if stripped[:2] != b"\xff\xd8":
        return stripped
    return stripped[:2] + segment + stripped[2:]


def read_jpeg_metadata(path: str | Path) -> dict:
    """读取 JPEG COM 段里的 JSON 元信息;缺失或损坏时返回空 dict。"""
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as image:
            raw = image.info.get("comment")
    except Exception:
        return {}
    if raw is None:
        return {}
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_image_with_metadata(
    path: str | Path, image: np.ndarray, metadata: dict, quality: int = 80
) -> None:
    """把 numpy RGB 图编码为 JPEG,写入元信息后落盘。"""
    from PIL import Image as PILImage

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    PILImage.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        buf, format="JPEG", quality=quality
    )
    target.write_bytes(inject_jpeg_metadata(buf.getvalue(), metadata))


def save_jpeg_bytes_with_metadata(path: str | Path, jpeg_bytes: bytes, metadata: dict) -> None:
    """把已有的 JPEG 字节写入元信息后落盘(用于工具返回图)。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(inject_jpeg_metadata(jpeg_bytes, metadata))
