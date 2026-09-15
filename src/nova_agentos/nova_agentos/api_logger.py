"""把 provider 的完整请求与响应记录到文件。"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
from io import BytesIO
import time
from pathlib import Path
from typing import Any


class ApiLogger:
    """按请求 id 把 API 事件写成 JSONL,可选把图像单独落盘并脱敏密钥。"""

    def __init__(self, enabled: bool = True, images: bool = True, directory: str | Path | None = None, retention_days: int = 30):
        default = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        self.enabled = bool(enabled)
        self.images = bool(images)
        self.directory = Path(directory or Path(default) / "nova_agentos" / "api_logs").expanduser()
        self.retention_days = max(0, int(retention_days))
        self._paths: dict[str, Path] = {}
        self._lock = threading.Lock()

    def _path(self, request_id: str) -> Path:
        """按日期分目录、时间戳+request_id 命名,返回该请求的日志文件路径。"""
        day = time.strftime("%Y-%m-%d")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return self.directory / day / f"{stamp}-{request_id}.jsonl"

    def log(self, event: dict, request_id: str, messages: list | None = None) -> None:
        """把一条事件追加写入对应请求的 JSONL 文件,并在末尾触发过期清理。"""
        if not self.enabled:
            return
        with self._lock:
            path = self._paths.setdefault(request_id, self._path(request_id))
        path.parent.mkdir(parents=True, exist_ok=True)
        safe = dict(event)
        if messages is not None:
            safe["messages"] = self._sanitize_messages(messages, request_id)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(safe, ensure_ascii=False) + "\n")
        self._cleanup()

    def _sanitize_messages(self, messages: list, request_id: str) -> list:
        """逐条脱敏消息:文本里的相机元信息提取为字段,base64 图像转成引用。"""
        output = []
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), list):
                camera = ""
                timestamp = ""
                shape = ""
                content = []
                for value in message["content"]:
                    if isinstance(value, dict) and value.get("type") == "text":
                        text = str(value.get("text", ""))
                        match = re.search(
                            r"camera:\s*([^;]+)(?:;\s*timestamp:\s*([^;]+))?(?:;\s*shape:\s*(.+))?",
                            text,
                        )
                        if match:
                            camera, timestamp, shape = (
                                match.group(1).strip(),
                                (match.group(2) or "").strip(),
                                (match.group(3) or "").strip(),
                            )
                    if isinstance(value, dict) and value.get("type") == "image_url":
                        content.append(
                            self._sanitize_image(value, request_id, camera, timestamp, shape)
                        )
                    else:
                        content.append(self._sanitize(value, request_id))
                output.append({**message, "content": content})
            else:
                output.append(self._sanitize(message, request_id))
        return output

    def _sanitize_image(
        self, value: dict, request_id: str, camera: str, timestamp: str, shape: str
    ) -> dict:
        """把 data URL 图像解码后落盘,返回含尺寸/哈希/路径的图像引用。"""
        url = (value.get("image_url") or {}).get("url", "")
        if not (url.startswith("data:") and ";base64," in url):
            return self._sanitize(value, request_id)
        header, encoded = url.split(";base64,", 1)
        raw = base64.b64decode(encoded)
        digest = hashlib.sha256(raw).hexdigest()
        width = height = 0
        try:
            from PIL import Image

            with Image.open(BytesIO(raw)) as image:
                width, height = image.size
        except Exception:
            pass
        image_path = ""
        if self.images:
            folder = self.directory / "images" / request_id
            folder.mkdir(parents=True, exist_ok=True)
            image_path = str(folder / f"{len(list(folder.iterdir())):03d}.jpg")
            Path(image_path).write_bytes(raw)
        return {
            "type": "image_ref",
            "camera": camera,
            "timestamp": timestamp,
            "shape": shape,
            "media_type": header[5:],
            "path": image_path,
            "width": width,
            "height": height,
            "bytes": len(raw),
            "sha256": digest,
        }

    def _sanitize(self, value: Any, request_id: str) -> Any:
        """递归脱敏:移除鉴权字段、遮蔽 Bearer token、把内联图像替换为引用。"""
        if isinstance(value, list):
            return [self._sanitize(v, request_id) for v in value]
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                url = (value.get("image_url") or {}).get("url", "")
                if url.startswith("data:") and ";base64," in url:
                    header, encoded = url.split(";base64,", 1)
                    raw = base64.b64decode(encoded)
                    digest = hashlib.sha256(raw).hexdigest()
                    image_path = ""
                    if self.images:
                        folder = self.directory / "images" / request_id
                        folder.mkdir(parents=True, exist_ok=True)
                        image_path = str(folder / f"{len(list(folder.iterdir())):03d}.jpg")
                        Path(image_path).write_bytes(raw)
                    return {"type": "image_ref", "media_type": header[5:], "path": image_path, "bytes": len(raw), "sha256": digest}
            return {k: self._sanitize(v, request_id) for k, v in value.items() if k.lower() not in {"authorization", "api_key", "apikey", "x-api-key"}}
        if isinstance(value, str):
            return re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", value)
        return value

    def request(self, request_id: str, metadata: dict, messages: list, tools: list | None, body: dict) -> None:
        """记录请求事件:保存 provider 原始载荷,并从原始消息中提取图像引用。"""
        event = {
            "type": "request",
            **metadata,
            "tools": tools or [],
            "body": self._sanitize({k: v for k, v in body.items() if k != "messages"}, request_id),
        }
        self.log(event, request_id, messages)

    def response(self, request_id: str, metadata: dict, response: Any = None, error: str = "", http_status: int | None = None) -> None:
        """记录响应事件:保存脱敏后的响应体、tool_calls 与错误信息。"""
        event = {"type": "response", **metadata, "http_status": http_status, "error": error}
        if response is not None:
            event["response"] = self._sanitize(response, request_id)
            if isinstance(response, dict):
                event["tool_calls"] = (response.get("choices") or [{}])[0].get("message", {}).get("tool_calls", [])
        self.log(event, request_id)

    def _cleanup(self) -> None:
        """删除超过 retention_days 的旧 JSONL 日志文件。"""
        if not self.enabled or not self.retention_days:
            return
        cutoff = time.time() - self.retention_days * 86400
        for path in self.directory.glob("*/*.jsonl"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass
