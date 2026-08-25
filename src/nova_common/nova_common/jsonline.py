# 行式 JSON 协议客户端/服务端(无 ROS 依赖)。
# JsonLineClient 由 bridge 使用;JsonLineServer/JsonLineRequestHandler 由 sim server 使用。
import json
import socket
import socketserver
import traceback
from typing import Any


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

    # 发送一行 JSON 请求并读取一行响应,失败时关闭连接
    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._connect()
        assert self.sock is not None
        assert self.file is not None
        try:
            line = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            self.sock.sendall(line)
            response_line = self.file.readline()
            if not response_line:
                raise ConnectionError("sim server closed the connection")
            response = json.loads(response_line.decode("utf-8"))
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
        self.sock = sock
        self.file = sock.makefile("rb")


class JsonLineRequestHandler(socketserver.StreamRequestHandler):
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
