# 远程 pi0 推理服务 HTTP 客户端。
# 仅做 base64 图像 + 指令 + state 的打包与动作向量的解析,不依赖 ROS。
import base64

import numpy as np
import requests


class RemotePi0Client:
    def __init__(self, server_url: str, timeout_sec: float = 60.0) -> None:
        self.url = server_url.rstrip("/") + "/predict"
        self.timeout_sec = timeout_sec

    def predict(
        self,
        images: dict[str, np.ndarray],
        instruction: str,
        state: np.ndarray | None = None,
    ) -> np.ndarray:
        payload = {
            "instruction": instruction,
            "images": {
                name: {
                    "data": base64.b64encode(np.ascontiguousarray(img)).decode("ascii"),
                    "shape": list(img.shape),
                }
                for name, img in images.items()
            },
            "state": None if state is None else state.astype(float).tolist(),
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        data = response.json()
        if "action" not in data:
            raise RuntimeError(f"远程服务响应缺少 action: {data}")
        return np.asarray(data["action"], dtype=np.float32)
