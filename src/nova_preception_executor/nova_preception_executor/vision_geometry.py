# 感知几何:网格↔像素换算、DLT 三角化、重投影误差、图像标注。纯 numpy,无 ROS 依赖。
import base64
import io
import re

import numpy as np


# ---------- 投影 ----------

def project_point(P, X):
    # P: 3x4 世界->像素矩阵;X: 3D 世界坐标。返回 (u, v) 像素坐标。
    x = P @ np.append(np.asarray(X, dtype=np.float64), 1.0)
    return x[0] / x[2], x[1] / x[2]


# DLT 三角化:由多视图像素点 (u_i, v_i) 与投影矩阵 P_i 求 3D 点(最小化代数误差)。
def triangulate(pts2d, projs):
    assert len(pts2d) == len(projs) and len(pts2d) >= 2
    rows = []
    for (u, v), P in zip(pts2d, projs):
        P = np.asarray(P, dtype=np.float64)
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    X = vt[-1]
    return (X[:3] / X[3]).tolist()


def reprojection_errors(pts2d, projs, X):
    errors = []
    for (u, v), P in zip(pts2d, projs):
        uu, vv = project_point(P, X)
        errors.append(np.hypot(uu - u, vv - v))
    return errors


# ---------- 网格 ----------

def grid_cell_to_pixel(row, col, grid_size, height, width):
    # 网格行自上而下 1..grid_size,列自左而右 1..grid_size;取该格的像素中心。
    cell_h = height / grid_size
    cell_w = width / grid_size
    u = (col - 0.5) * cell_w
    v = (row - 0.5) * cell_h
    return u, v


# 解析 VLM 输出的网格单元坐标,兼容 "3-5" / "C4" / "4C" / "row3-col5" / {"row":..,"col":..}。
def parse_grid_cell(text):
    if isinstance(text, dict):
        if "row" in text and "col" in text:
            return int(text["row"]), int(text["col"])
        text = str(text.get("grid", ""))
    s = str(text).strip().lower()
    m = re.search(r"(\d+)\s*[-,:]\s*(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d+)\s*[xX\s]*([a-h])$", s)
    if m:
        return int(m.group(1)), ord(m.group(2)) - ord("a") + 1
    m = re.match(r"^([a-h])\s*[xX\s]*(\d+)$", s)
    if m:
        return int(m.group(2)), ord(m.group(1)) - ord("a") + 1
    return None


# 解析 VLM 输出的像素坐标,兼容 "[x, y]" / "(x,y)" / {"x":..,"y":..}。
def parse_pixel(text):
    if isinstance(text, dict):
        if "x" in text and "y" in text:
            return float(text["x"]), float(text["y"])
        text = str(text.get("pixel", ""))
    s = str(text).strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


# ---------- 图像标注 ----------

def draw_grid(img, grid_size, line_color=(200, 200, 200), label_color=(0, 200, 0)):
    out = img.copy()
    h, w = out.shape[:2]
    for i in range(1, grid_size):
        y = int(round(h * i / grid_size))
        x = int(round(w * i / grid_size))
        out[y, :] = line_color
        out[:, x] = line_color
    cell_h, cell_w = h / grid_size, w / grid_size
    for r in range(grid_size):
        for c in range(grid_size):
            # 标签顺序与提示词/parse_grid_cell 一致:行-列(r+1 行, c+1 列)
            label = f"{r + 1}-{c + 1}"
            x = int((c + 0.5) * cell_w)
            y = int((r + 0.5) * cell_h)
            _put_label(out, label, x, y, label_color)
    return out


def draw_marker(img, pixel, color, radius=8, label=None):
    # 画空心圆圈(仅外圈着色),不遮挡圈内物体
    out = img.copy()
    u, v = float(pixel[0]), float(pixel[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return out
    h, w = out.shape[:2]
    cu = int(round(u))  # u = 横/列
    cv = int(round(v))  # v = 纵/行
    cu = int(np.clip(cu, 0, w - 1))
    cv = int(np.clip(cv, 0, h - 1))
    rr, cc = np.ogrid[:h, :w]
    d2 = (rr - cv) ** 2 + (cc - cu) ** 2
    thick = max(1, int(round(radius * 0.25)))
    ring = (d2 <= radius * radius) & (d2 >= (radius - thick) ** 2)
    out[ring] = color
    if label:
        _put_label(out, label, cu, max(cv - radius - 8, 0), color)
    return out


def _put_label(img, text, x, y, color):
    h, w = img.shape[:2]
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(img)
        ImageDraw.Draw(pil).text((x, y), text, fill=tuple(int(c) for c in color))
        img[:] = np.asarray(pil)
        return img
    except Exception:
        pass
    # 无 PIL 时用像素粗体画 label(仅调试用)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            yy, xx = int(y) + dy, int(x) + dx
            if 0 <= yy < h and 0 <= xx < w:
                img[yy, xx] = color
    return img


_ENCODE_MAX_SIZE = 768


def sent_image_size(height, width, max_size=_ENCODE_MAX_SIZE):
    # 等比缩小规则与 encode_image 一致:只缩不放;返回发送给 VLM 的实际 (宽, 高)
    scale = min(1.0, max_size / max(height, width))
    if scale < 1.0:
        return int(round(width * scale)), int(round(height * scale))
    return int(width), int(height)


def encode_image(img, max_size=_ENCODE_MAX_SIZE, quality=80):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    cw, ch = sent_image_size(h, w, max_size)
    if (cw, ch) != (w, h):
        img = _resize(img, cw, ch)
    buf = io.BytesIO()
    from PIL import Image as PILImage
    PILImage.fromarray(img, mode="RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _resize(img, w, h):
    from PIL import Image as PILImage
    return np.asarray(PILImage.fromarray(img, mode="RGB").resize((w, h)))
