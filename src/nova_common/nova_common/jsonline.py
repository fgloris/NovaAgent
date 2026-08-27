# 帧式二进制协议客户端/服务端(无 ROS 依赖)。
# 帧格式:4 字节大端 header 长度 + header JSON + 二进制 body(所有 numpy 数组按序拼接)。
# 数组在 header 里以 __blob__ 占位符描述(shape/dtype),原始字节放 body,免 base64。
# JsonLineClient 由 bridge 使用;JsonLineServer/JsonLineRequestHandler 由 sim server 使用。
from __future__ import annotations
import json
import socket
import socketserver
import struct
import traceback
from typing import Any

from nova_common.obs_codec import blob_size, decode_frame, encode_frame


def _read_exact(reader, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = reader.read(remaining)
        if not chunk:
            raise ConnectionError("connection closed while reading frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# 空闲 idle_sec 后开始 keepalive 探测,间隔 interval_sec;连续探测次数用系统默认
def _set_keepalive(sock: socket.socket, idle_sec: int = 60, interval_sec: int = 30) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle_sec)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_sec)


def _to_frame(payload: dict[str, Any]) -> bytes:
    blobs: list[bytes] = []
    header = encode_frame(payload, blobs)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(header_bytes)) + header_bytes + b"".join(blobs)


def _from_frame(reader) -> dict[str, Any]:
    header_bytes = _read_exact(reader, struct.unpack(">I", _read_exact(reader, 4))[0])
    header = json.loads(header_bytes.decode("utf-8"))
    body = _read_exact(reader, blob_size(header))
    blobs = _split_blobs(header, body)
    return decode_frame(header, blobs)


# 按 header 里 __blob__ 的出现顺序切分 body
def _split_blobs(header: Any, body: bytes) -> list[bytes]:
    blobs = []
    offset = 0
    for size in _iter_blob_sizes(header):
        blobs.append(body[offset:offset + size])
        offset += size
    return blobs


def _iter_blob_sizes(value: Any):
    if isinstance(value, dict) and "__blob__" in value:
        yield blob_size(value)
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from _iter_blob_sizes(v)
    elif isinstance(value, list):
        for v in value:
            yield from _iter_blob_sizes(v)


class JsonLineClient:
    def __init__(self, host: str, port: int, timeout_sec: float) -> None:
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.sock: socket.socket | None = None
        self.file = None

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    # 发送一帧请求并读取一帧响应,失败时关闭连接
    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._connect()
        assert self.sock is not None
        assert self.file is not None
        try:
            self.sock.sendall(_to_frame(payload))
            response = _from_frame(self.file)
        except Exception:
            self.close()
            raise

        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "sim server request failed"))
        return response

    def _connect(self) -> None:
        if self.sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
        sock.settimeout(self.timeout_sec)
        _set_keepalive(sock)
        self.sock = sock
        self.file = sock.makefile("rb")


class JsonLineRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            try:
                request = _from_frame(self.rfile)
            except ConnectionError:
                return
            try:
                response = self.dispatch(request)
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            try:
                self.wfile.write(_to_frame(response))
                self.wfile.flush()
            except (ConnectionError, BrokenPipeError, OSError):
                return

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type == "reset":
            return self.server.session.reset(request)
        if request_type == "step":
            return self.server.session.step(request)
        if request_type == "close":
            self.server.session.close()
            return {"ok": True}
        raise ValueError(f"unknown request type: {request_type!r}")


class JsonLineServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, session: Any) -> None:
        self.session = session
        super().__init__(address, JsonLineRequestHandler)

    def get_request(self):
        sock, addr = super().get_request()
        _set_keepalive(sock)
        return sock, addr


def serve(host: str, port: int, session: Any, name: str) -> int:
    with JsonLineServer((host, port), session) as server:
        print(f"{name} sim server listening on {host}:{port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            session.close()
    return 0
