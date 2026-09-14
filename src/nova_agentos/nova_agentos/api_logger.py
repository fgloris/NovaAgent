"""File logger for complete provider requests and responses."""
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
    def __init__(self, enabled: bool = True, images: bool = True, directory: str | Path | None = None, retention_days: int = 30):
        default = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        self.enabled = bool(enabled)
        self.images = bool(images)
        self.directory = Path(directory or Path(default) / "nova_agentos" / "api_logs").expanduser()
        self.retention_days = max(0, int(retention_days))
        self._paths: dict[str, Path] = {}
        self._lock = threading.Lock()

    def _path(self, request_id: str) -> Path:
        day = time.strftime("%Y-%m-%d")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return self.directory / day / f"{stamp}-{request_id}.jsonl"

    def log(self, event: dict, request_id: str, messages: list | None = None) -> None:
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
        # Store the exact provider payload as text, while extracting image
        # references once from the original message list.
        event = {
            "type": "request",
            **metadata,
            "tools": tools or [],
            "body": self._sanitize({k: v for k, v in body.items() if k != "messages"}, request_id),
        }
        self.log(event, request_id, messages)

    def response(self, request_id: str, metadata: dict, response: Any = None, error: str = "", http_status: int | None = None) -> None:
        event = {"type": "response", **metadata, "http_status": http_status, "error": error}
        if response is not None:
            event["response"] = self._sanitize(response, request_id)
            if isinstance(response, dict):
                event["tool_calls"] = (response.get("choices") or [{}])[0].get("message", {}).get("tool_calls", [])
        self.log(event, request_id)

    def _cleanup(self) -> None:
        if not self.enabled or not self.retention_days:
            return
        cutoff = time.time() - self.retention_days * 86400
        for path in self.directory.glob("*/*.jsonl"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass
