"""工具返回值里内联图像的约定与解析。

约定:任何工具的 ``result_json`` 可带保留键 ``images``:
  - dict: ``{image_id: data_url}``
  - list: ``[data_url, ...]``(自动编号 image_0, image_1, ...)

AgentOS 用 :func:`split_images` 把图像从给模型的文本里剥离,再作为多模态消息注入。
"""
from __future__ import annotations

from typing import Any

IMAGES_KEY = "images"


def split_images(result: Any) -> tuple[Any, dict[str, str]]:
    """剥离 ``result`` 里的 ``images`` 键,返回 (去掉图像的 result, {id: data_url})。

    - result 不是 dict 时原样返回,图像为空。
    - 只接受 ``data:`` 开头的字符串,其它项忽略。
    """
    if not isinstance(result, dict):
        return result, {}
    raw = result.get(IMAGES_KEY)
    stripped = {key: value for key, value in result.items() if key != IMAGES_KEY}
    if not raw:
        return stripped, {}
    images: dict[str, str] = {}
    if isinstance(raw, dict):
        for key, url in raw.items():
            if isinstance(url, str) and url.startswith("data:"):
                images[str(key)] = url
    elif isinstance(raw, list):
        for index, url in enumerate(raw):
            if isinstance(url, str) and url.startswith("data:"):
                images[f"image_{index}"] = url
    return stripped, images
