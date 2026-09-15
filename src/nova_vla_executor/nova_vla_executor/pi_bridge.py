"""远程 pi0 推理服务 WebSocket 客户端。

发送二进制帧:首行 JSON header(instruction/state/图像元数据)+ 图像原始字节拼接,免 base64。
长连接复用,每次 predict 一帧推理,线程安全。
"""
import json
import threading

import numpy as np
import websocket


class RemotePi0Client:
    """pi0 推理服务的 WebSocket 客户端,负责打包图像/状态并解析动作块。"""

    def __init__(self, server_url: str, timeout_sec: float = 60.0) -> None:
        self.ws_url = (
            server_url.rstrip("/")
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            + "/predict"
        )
        self.timeout_sec = timeout_sec
        self._ws: websocket.WebSocket | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        """关闭底层 WebSocket 连接(线程安全,幂等)。"""
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                finally:
                    self._ws = None

    def predict(
        self,
        images: dict[str, np.ndarray],
        instruction: str,
        state: np.ndarray | None = None,
    ) -> np.ndarray:
        """发送一帧推理请求并返回动作块;优先取 action_chunk,否则包装单步 action。

        图像按顺序拼接进二进制 body,header 记录各自的 shape/dtype 供服务端切分。
        """
        meta: dict[str, dict] = {}
        blob = bytearray()
        for name, img in images.items():
            arr = np.ascontiguousarray(img)
            meta[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            blob += arr.tobytes()
        header = {
            "instruction": instruction,
            "state": None if state is None else state.astype(float).tolist(),
            "images": meta,
        }
        payload = json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n" + bytes(blob)

        with self._lock:
            self._connect()
            try:
                self._ws.send_binary(payload)
                data = json.loads(self._ws.recv())
            except Exception:
                self._ws = None
                raise
        if "action_chunk" in data:
            return np.asarray(data["action_chunk"], dtype=np.float32)
        if "action" in data:
            return np.asarray(data["action"], dtype=np.float32)[None, :]
        raise RuntimeError(f"远程服务响应缺少 action/action_chunk: {data}")

    def _connect(self) -> None:
        """懒建立 WebSocket 长连接,已连接则直接返回。"""
        if self._ws is not None:
            return
        self._ws = websocket.create_connection(self.ws_url, timeout=self.timeout_sec)
