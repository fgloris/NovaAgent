#!/usr/bin/env python3
"""查看 JPEG COM 段里的 JSON 元数据(NovaAgent 图像记忆注入的那种)。

独立脚本,不依赖本项目。用法::

    python3 scripts/inspect_jpeg_metadata.py a.jpg b.jpg
    python3 scripts/inspect_jpeg_metadata.py ~/.cache/nova_agentos/memory/*/processed/*.jpg
"""
import json
import struct
import sys
from pathlib import Path

COM_MARKER = 0xFE


def read_com_metadata(path):
    """解析 JPEG 的 COM 段并返回其中的 JSON 字典;无 COM 段返回 None。"""
    data = Path(path).read_bytes()
    if data[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(data)
    while i + 1 < n:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xD8:
            i += 2
            continue
        if marker == 0xDA:  # SOS:之后是压缩数据
            break
        if i + 3 >= n:
            break
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker == COM_MARKER:
            payload = data[i + 4:i + 2 + seg_len]
            text = payload.decode("utf-8", "replace")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"_raw_comment": text}
        i += 2 + seg_len
    return None


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    for path in argv:
        meta = read_com_metadata(path)
        print(f"== {path} ==")
        if meta is None:
            print("(无 COM 元数据)")
        else:
            print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
