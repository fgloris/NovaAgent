"""AgentOS 图像记忆:session 级图像落盘、统一命名、JPEG 元信息与保留策略。

落盘结构::

    <root>/current/<camera>.jpg
    <root>/history/<epoch_ms>-<camera>.jpg
    <root>/processed/<epoch_ms>-<camera>-<seq>.jpg

所有元信息(相机/时间/来源/底图)写进 JPEG COM 段,文件名保持简短;
模型只通过 ``file://<kind>/<file>`` 引用,executor 用隐藏的 root 解析。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nova_common import image_codec

KIND_CURRENT = "current"
KIND_HISTORY = "history"
KIND_PROCESSED = "processed"
ORIGIN_RAW = "raw_camera"
ORIGIN_TOOL = "tool"

_DIFF_SIZE = 32  # 去重比较用的缩略图边长


def _iso(ts: float) -> str:
    """epoch 秒 -> UTC ISO 字符串。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int((ts % 1) * 1000):03d}Z"


@dataclass
class ImageRecord:
    """一张已落盘图像的元数据。"""

    kind: str
    filename: str
    camera: str
    time: float
    width: int
    height: int
    origin: str
    iso: str = ""
    tool: str = ""
    params: dict = field(default_factory=dict)
    base: str = ""

    @property
    def url(self) -> str:
        return image_codec.image_url(self.kind, self.filename)

    def to_meta(self) -> dict:
        """写入 JPEG COM 段的元信息(不含内部 root/filename)。"""
        meta = {
            "kind": self.kind,
            "camera": self.camera,
            "time": round(self.time, 6),
            "iso": self.iso,
            "width": self.width,
            "height": self.height,
            "origin": self.origin,
        }
        if self.origin == ORIGIN_TOOL:
            meta.update({"tool": self.tool, "params": self.params, "base": self.base})
        return meta

    def describe(self) -> dict:
        """给模型看的精简描述(url/time/camera/size/provenance)。"""
        out = {
            "url": self.url,
            "kind": self.kind,
            "camera": self.camera,
            "time": round(self.time, 3),
            "iso": self.iso,
            "size": [self.width, self.height],
            "origin": self.origin,
        }
        if self.origin == ORIGIN_TOOL:
            out["tool"] = self.tool
            out["params"] = self.params
            out["base"] = self.base
        return out


class ImageMemory:
    """session 级图像记忆:负责落盘、去重、检索与保留;线程安全。"""

    def __init__(
        self,
        root: str | Path,
        link_max: int = 60,
        processed_max: int = 16,
        jpeg_quality: int = 80,
        max_size: int = 768,
        diff_mse_threshold: float = 0.0005,
    ) -> None:
        self.root = Path(root).expanduser()
        self.link_max = max(1, int(link_max))
        self.processed_max = max(1, int(processed_max))
        self.jpeg_quality = min(95, max(20, int(jpeg_quality)))
        self.max_size = max(64, int(max_size))
        self.diff_mse_threshold = max(0.0, float(diff_mse_threshold))
        self._lock = threading.RLock()
        self._current: dict[str, ImageRecord] = {}
        self._history: dict[str, list[ImageRecord]] = {}
        self._processed: list[ImageRecord] = []
        self._last_frames: dict[str, np.ndarray] = {}
        self._seq = 0
        self._load()

    # ---------- 写入 ----------

    def refresh_current(self, frames: dict[str, tuple[np.ndarray, float]]) -> dict[str, ImageRecord]:
        """用各链路最新帧刷新 ``current/<camera>.jpg``(覆盖写),返回记录。"""
        out: dict[str, ImageRecord] = {}
        for camera, (frame, ts) in frames.items():
            record = self._write(KIND_CURRENT, f"{camera}.jpg", camera, frame, ts, ORIGIN_RAW)
            with self._lock:
                self._current[camera] = record
            out[camera] = record
        return out

    def save_history(self, camera: str, frame: np.ndarray, ts: float) -> ImageRecord | None:
        """按去重阈值保存一张历史图;与上一张差异过小时返回 None。"""
        with self._lock:
            previous = self._last_frames.get(camera)
        if previous is not None and self._diff(previous, frame) < self.diff_mse_threshold:
            return None
        filename = f"{int(ts * 1000)}-{camera}.jpg"
        record = self._write(KIND_HISTORY, filename, camera, frame, ts, ORIGIN_RAW)
        with self._lock:
            self._last_frames[camera] = frame
            items = self._history.setdefault(camera, [])
            items.append(record)
            removed = items[: max(0, len(items) - self.link_max)]
            del items[: len(removed)]
        for old in removed:
            self._unlink(old)
        return record

    def save_processed(
        self,
        jpeg_bytes: bytes,
        camera: str,
        tool: str,
        params: dict,
        base_url: str,
        ts: float,
    ) -> ImageRecord:
        """保存工具返回图(JPEG 字节),并记录工具来源与底图。"""
        with self._lock:
            self._seq += 1
            seq = self._seq
        filename = f"{int(ts * 1000)}-{camera}-{seq}.jpg"
        path = self.root / KIND_PROCESSED / filename
        record = ImageRecord(
            kind=KIND_PROCESSED,
            filename=filename,
            camera=camera,
            time=ts,
            width=0,
            height=0,
            origin=ORIGIN_TOOL,
            iso=_iso(ts),
            tool=tool,
            params=params,
            base=base_url,
        )
        image_codec.save_jpeg_bytes_with_metadata(path, jpeg_bytes, record.to_meta())
        with self._lock:
            self._processed.append(record)
            removed = self._processed[: max(0, len(self._processed) - self.processed_max)]
            del self._processed[: len(removed)]
        for old in removed:
            self._unlink(old)
        return record

    # ---------- 查询 ----------

    def current_records(self) -> list[ImageRecord]:
        with self._lock:
            return [self._current[cam] for cam in sorted(self._current)]

    def history_records(self, camera: str, depth: int) -> list[ImageRecord]:
        with self._lock:
            items = list(self._history.get(camera, []))
        return items[-depth:] if depth > 0 else []

    def processed_records(self) -> list[ImageRecord]:
        with self._lock:
            return list(self._processed)

    def all_history_cameras(self) -> list[str]:
        with self._lock:
            return sorted(self._history)

    def nearest_history(self, camera: str, ts: float) -> ImageRecord | None:
        """返回该链路中时间最接近 ts 的历史图。"""
        with self._lock:
            items = list(self._history.get(camera, []))
        if not items:
            return None
        return min(items, key=lambda r: abs(r.time - ts))

    def find(self, url: str) -> ImageRecord | None:
        """按 ``file://<kind>/<file>`` 查找记录(用于校验/补全元信息)。"""
        text = str(url)
        if text.startswith(image_codec.IMAGE_URL_PREFIX):
            text = text[len(image_codec.IMAGE_URL_PREFIX):]
        kind, _, filename = text.partition("/")
        if not filename:
            return None
        with self._lock:
            if kind == KIND_CURRENT:
                for record in self._current.values():
                    if record.filename == filename:
                        return record
            elif kind == KIND_HISTORY:
                for items in self._history.values():
                    for record in items:
                        if record.filename == filename:
                            return record
            elif kind == KIND_PROCESSED:
                for record in self._processed:
                    if record.filename == filename:
                        return record
        return None

    # ---------- 内部 ----------

    def path_of(self, record: ImageRecord) -> Path:
        """返回记录的落盘路径。"""
        return self.root / record.kind / record.filename

    def data_url(self, record: ImageRecord) -> str:
        """读取记录对应的 JPEG 并编码为 data URL。"""
        return image_codec.file_to_data_url(self.path_of(record))

    def _write(
        self, kind: str, filename: str, camera: str, frame: np.ndarray, ts: float, origin: str
    ) -> ImageRecord:
        frame = image_codec.downscale(frame, self.max_size)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        path = self.root / kind / filename
        record = ImageRecord(
            kind=kind,
            filename=filename,
            camera=camera,
            time=ts,
            width=width,
            height=height,
            origin=origin,
            iso=_iso(ts),
        )
        image_codec.save_image_with_metadata(path, frame, record.to_meta(), quality=self.jpeg_quality)
        return record

    def _unlink(self, record: ImageRecord) -> None:
        try:
            self.path_of(record).unlink()
        except OSError:
            pass

    @staticmethod
    def _diff(a: np.ndarray, b: np.ndarray) -> float:
        """两张图的归一化 MSE(缩到 32x32 灰度,除以 255^2)。"""
        from PIL import Image as PILImage

        def thumb(image: np.ndarray) -> np.ndarray:
            pil = PILImage.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").convert("L")
            return np.asarray(pil.resize((_DIFF_SIZE, _DIFF_SIZE)), dtype=np.float64)

        delta = thumb(a) - thumb(b)
        return float((delta * delta).mean()) / (255.0 * 255.0)

    def _load(self) -> None:
        """启动时扫描目录、读取 JPEG 元信息重建索引。"""
        for kind, camera_glob in ((KIND_HISTORY, "*"), (KIND_PROCESSED, "*")):
            directory = self.root / kind
            if not directory.exists():
                continue
            for path in sorted(directory.glob(f"{camera_glob}.jpg")):
                record = self._record_from_file(kind, path)
                if record is None:
                    continue
                with self._lock:
                    if kind == KIND_HISTORY:
                        self._history.setdefault(record.camera, []).append(record)
                    else:
                        self._processed.append(record)
        for path in sorted((self.root / KIND_CURRENT).glob("*.jpg")):
            record = self._record_from_file(KIND_CURRENT, path)
            if record is not None:
                with self._lock:
                    self._current[record.camera] = record
        with self._lock:
            for items in self._history.values():
                items.sort(key=lambda r: r.time)
            self._processed.sort(key=lambda r: r.time)
            for items in self._history.values():
                del items[: max(0, len(items) - self.link_max)]
            del self._processed[: max(0, len(self._processed) - self.processed_max)]

    @staticmethod
    def _record_from_file(kind: str, path: Path) -> ImageRecord | None:
        meta = image_codec.read_jpeg_metadata(path)
        if not meta:
            return None
        try:
            return ImageRecord(
                kind=kind,
                filename=path.name,
                camera=str(meta.get("camera", "")),
                time=float(meta.get("time", 0.0)),
                width=int(meta.get("width", 0)),
                height=int(meta.get("height", 0)),
                origin=str(meta.get("origin", ORIGIN_RAW)),
                iso=str(meta.get("iso", "")),
                tool=str(meta.get("tool", "")),
                params=dict(meta.get("params") or {}),
                base=str(meta.get("base", "")),
            )
        except (TypeError, ValueError):
            return None
