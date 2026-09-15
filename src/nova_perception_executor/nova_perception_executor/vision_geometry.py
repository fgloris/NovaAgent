"""感知几何:网格↔像素换算、DLT 三角化、重投影误差、图像标注。纯 numpy,无 ROS 依赖。"""
import base64
import io
import re

import numpy as np


# ---------- 投影 ----------

def project_point(P, X):
    """把 3D 世界坐标 X 用 3x4 投影矩阵 P 投影为像素坐标 (u, v)。"""
    x = P @ np.append(np.asarray(X, dtype=np.float64), 1.0)
    return x[0] / x[2], x[1] / x[2]


def triangulate(pts2d, projs):
    """DLT 三角化:由多视图像素点 (u_i, v_i) 与投影矩阵 P_i 求 3D 点(最小化代数误差)。

    每帧构造两个线性约束 (u*P2-P0, v*P2-P1),拼成 A,对 A 做 SVD,
    取最小奇异值对应的右奇异向量即为齐次解,再除以 w 得到欧氏坐标。
    """
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
    """计算 3D 点 X 投影回各视图后与原始像素点的欧氏误差,用于评估三角化精度。"""
    errors = []
    for (u, v), P in zip(pts2d, projs):
        uu, vv = project_point(P, X)
        errors.append(np.hypot(uu - u, vv - v))
    return errors


# ---------- 网格 ----------

def grid_cell_to_pixel(row, col, grid_size, height, width):
    """把网格行列(1-based,行自上而下、列自左而右)换算为该格中心的像素坐标 (u, v)。"""
    cell_h = height / grid_size
    cell_w = width / grid_size
    u = (col - 0.5) * cell_w
    v = (row - 0.5) * cell_h
    return u, v


def parse_grid_cell(text):
    """解析 VLM 输出的网格单元坐标,兼容 "3-5" / "C4" / "4C" / "row3-col5" / {"row":..,"col":..}。"""
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


def parse_pixel(text):
    """解析 VLM 输出的像素坐标,兼容 "[x, y]" / "(x,y)" / {"x":..,"y":..}。"""
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

# 网格编号渲染:字号随格子大小缩放;低于该下限(≈此前默认小字)就放弃渲染编号
_LABEL_MIN_PX = 11
_LABEL_FONT_SCALE = 0.5  # 字号约为格子短边的比例


def _load_grid_font(px):
    """按像素高度加载 TrueType 字体(只有它才能缩放字号);找不到可用字体时返回 None(退化为默认小字)。"""
    try:
        from PIL import ImageFont
    except Exception:
        return None
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, px)
        except Exception:
            continue
    return None


def draw_grid(img, grid_size, line_color=(200, 200, 200), label_color=(0, 200, 0)):
    """在图像上叠加 grid_size×grid_size 网格线,并在每格中心渲染 "行-列" 编号。"""
    out = img.copy()
    h, w = out.shape[:2]
    for i in range(1, grid_size):
        y = int(round(h * i / grid_size))
        x = int(round(w * i / grid_size))
        out[y, :] = line_color
        out[:, x] = line_color
    cell_h, cell_w = h / grid_size, w / grid_size
    px = int(round(min(cell_h, cell_w) * _LABEL_FONT_SCALE))
    if px < _LABEL_MIN_PX:
        # 字号太小(低于当前可用最小字号),放弃渲染编号,只留网格线
        return out
    font = _load_grid_font(px)
    for r in range(grid_size):
        for c in range(grid_size):
            # 标签顺序与提示词/parse_grid_cell 一致:行-列(r+1 行, c+1 列)
            label = f"{r + 1}-{c + 1}"
            x = int((c + 0.5) * cell_w)
            y = int((r + 0.5) * cell_h)
            _put_label(out, label, x, y, label_color, font=font)
    return out


def draw_marker(img, pixel, color, radius=8, label=None):
    """在像素位置画空心圆圈标记(仅外圈着色),不遮挡圈内物体;可选在圈上方加标签。"""
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


def _put_label(img, text, x, y, color, font=None):
    """在图像 (x, y) 处居中绘制文本;优先用 PIL,无 PIL 时退化为像素点。"""
    h, w = img.shape[:2]
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        if font is not None:
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            x = int(x - (r - l) / 2)
            y = int(y - (b - t) / 2)
            draw.text((x, y), text, fill=tuple(int(c) for c in color), font=font)
        else:
            draw.text((x, y), text, fill=tuple(int(c) for c in color))
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
    """按 encode_image 的等比缩小规则(只缩不放)返回实际发送给 VLM 的 (宽, 高)。"""
    scale = min(1.0, max_size / max(height, width))
    if scale < 1.0:
        return int(round(width * scale)), int(round(height * scale))
    return int(width), int(height)


def encode_image(img, max_size=_ENCODE_MAX_SIZE, quality=80):
    """把图像等比缩小后编码为 data:image/jpeg;base64, 字符串,供 VLM 接口使用。"""
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
    """用 PIL 把图像缩放到 (w, h)。"""
    from PIL import Image as PILImage
    return np.asarray(PILImage.fromarray(img, mode="RGB").resize((w, h)))


# ---------- 3D 网格构建与投影渲染(base 系) ----------

def normalize_quat_xyzw(q):
    """归一化 XYZW 四元数;非法输入抛 ValueError。"""
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError("orientation 必须是 4 元 xyzw 四元数")
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("orientation 四元数范数为 0")
    return q / n


def quat_to_matrix_xyzw(q):
    """XYZW 四元数 -> 3x3 旋转矩阵。"""
    x, y, z, w = normalize_quat_xyzw(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def decompose_projection(intrinsics, projection):
    """由内参 K(3x3) 与投影矩阵 P(3x4, base->像素) 反解 base->camera 的 [R|t](3x4)。"""
    K = np.asarray(intrinsics, dtype=float).reshape(3, 3)
    P = np.asarray(projection, dtype=float).reshape(3, 4)
    return np.linalg.inv(K) @ P


def project_point_safe(K, Rt, point):
    """投影 base 系点;相机后方(depth<=0)返回 None,否则返回 (u, v, depth)。"""
    K = np.asarray(K, dtype=float).reshape(3, 3)
    Rt = np.asarray(Rt, dtype=float).reshape(3, 4)
    cam = Rt[:, :3] @ np.asarray(point, dtype=float).reshape(3) + Rt[:, 3]
    if cam[2] <= 1e-6:
        return None
    p = K @ cam
    return float(p[0] / p[2]), float(p[1] / p[2]), float(cam[2])


def _perp_basis(axis):
    """给定单位轴,返回两个与之正交的单位向量 u, v。"""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(a, u)
    return u, v


def _ring(center, u, v, radius, segments):
    """在 center 处、以 u/v 为平面基、给定半径生成一圈顶点。"""
    ang = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)
    return np.asarray(center, dtype=float) + radius * (
        np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    )


class _Mesh:
    """三角网格累加器:顶点/面/面颜色。"""

    def __init__(self):
        self.vertices: list[list[float]] = []
        self.faces: list[list[int]] = []
        self.colors: list[list[float]] = []

    def add_ring(self, ring):
        start = len(self.vertices)
        self.vertices.extend(np.asarray(ring, dtype=float).tolist())
        return list(range(start, start + len(ring)))

    def add_vertex(self, point):
        self.vertices.append([float(v) for v in np.asarray(point, dtype=float).reshape(3)])
        return len(self.vertices) - 1

    def add_face(self, i, j, k, color):
        self.faces.append([int(i), int(j), int(k)])
        self.colors.append([float(c) for c in color])

    def as_arrays(self):
        return (
            np.asarray(self.vertices, dtype=float),
            np.asarray(self.faces, dtype=int),
            np.asarray(self.colors, dtype=float),
        )


def _merge_meshes(meshes):
    """把多个 (vertices, faces, colors) 合并成一个,自动偏移面索引。"""
    vertices, faces, colors = [], [], []
    offset = 0
    for verts, tris, cols in meshes:
        vertices.extend(np.asarray(verts, dtype=float).tolist())
        for tri, col in zip(tris, cols):
            faces.append([int(tri[0]) + offset, int(tri[1]) + offset, int(tri[2]) + offset])
            colors.append([float(c) for c in col])
        offset += len(verts)
    return (
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=int),
        np.asarray(colors, dtype=float),
    )


def build_arrow(origin, orientation, length, radius, head_length=None, head_radius=None,
                segments=16, color=(255, 80, 80)):
    """沿 orientation 的局部 +z 构建「圆柱杆 + 圆锥头」箭头网格(base 系)。"""
    R = quat_to_matrix_xyzw(orientation)
    axis = R @ np.array([0.0, 0.0, 1.0])
    origin = np.asarray(origin, dtype=float).reshape(3)
    length = float(length)
    radius = float(radius)
    head_length = float(head_length) if head_length else max(radius * 2.5, length * 0.3)
    head_length = min(head_length, length * 0.9)
    head_radius = float(head_radius) if head_radius else max(radius * 2.0, radius + 1e-4)
    u, v = _perp_basis(axis)
    p0 = origin
    p1 = origin + axis * (length - head_length)
    apex = origin + axis * length
    mesh = _Mesh()
    r0 = mesh.add_ring(_ring(p0, u, v, radius, segments))
    r1 = mesh.add_ring(_ring(p1, u, v, radius, segments))
    r2 = mesh.add_ring(_ring(p1, u, v, head_radius, segments))
    c0 = mesh.add_vertex(p0)
    apex_i = mesh.add_vertex(apex)
    n = int(segments)
    for i in range(n):
        j = (i + 1) % n
        mesh.add_face(c0, r0[j], r0[i], color)
        mesh.add_face(r0[i], r0[j], r1[j], color)
        mesh.add_face(r0[i], r1[j], r1[i], color)
        mesh.add_face(r1[i], r1[j], r2[j], color)
        mesh.add_face(r1[i], r2[j], r2[i], color)
        mesh.add_face(r2[i], r2[j], apex_i, color)
    return mesh.as_arrays()


def build_cylinder(p0, p1, radius, segments=16, color=(80, 180, 255)):
    """两点之间的圆柱网格(base 系);两点重合时退化为小球。"""
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return build_sphere(p0, radius, color, segments, max(4, segments // 2))
    u, v = _perp_basis(axis / length)
    mesh = _Mesh()
    r0 = mesh.add_ring(_ring(p0, u, v, radius, segments))
    r1 = mesh.add_ring(_ring(p1, u, v, radius, segments))
    c0 = mesh.add_vertex(p0)
    c1 = mesh.add_vertex(p1)
    n = int(segments)
    for i in range(n):
        j = (i + 1) % n
        mesh.add_face(c0, r0[j], r0[i], color)
        mesh.add_face(r0[i], r0[j], r1[j], color)
        mesh.add_face(r0[i], r1[j], r1[i], color)
        mesh.add_face(c1, r1[i], r1[j], color)
    return mesh.as_arrays()


def build_sphere(center, radius, color=(255, 220, 0), segments=16, rings=8):
    """以 center 为心的小球网格(base 系),用于标记 3D 点。"""
    center = np.asarray(center, dtype=float).reshape(3)
    mesh = _Mesh()
    top = mesh.add_vertex(center + [0.0, 0.0, radius])
    bottom = mesh.add_vertex(center - [0.0, 0.0, radius])
    ring_ids = []
    for r in range(1, int(rings)):
        phi = np.pi * r / int(rings)
        z = np.cos(phi) * radius
        rr = np.sin(phi) * radius
        ring = _ring(center + [0.0, 0.0, z], np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), rr, segments)
        ring_ids.append(mesh.add_ring(ring))
    n = int(segments)
    for i in range(n):
        j = (i + 1) % n
        mesh.add_face(top, ring_ids[0][j], ring_ids[0][i], color)
        mesh.add_face(bottom, ring_ids[-1][i], ring_ids[-1][j], color)
    for r in range(len(ring_ids) - 1):
        a, b = ring_ids[r], ring_ids[r + 1]
        for i in range(n):
            j = (i + 1) % n
            mesh.add_face(a[i], a[j], b[j], color)
            mesh.add_face(a[i], b[j], b[i], color)
    return mesh.as_arrays()


def build_frame(origin, orientation, axis_length=0.1, radius=0.006, segments=16):
    """在 origin/orientation 处构建 x/y/z 三色箭头(base 系),颜色红/绿/蓝。"""
    colors = ((255, 70, 70), (70, 200, 70), (70, 120, 255))
    axes = (
        quat_to_matrix_xyzw(orientation) @ np.array([1.0, 0.0, 0.0]),
        quat_to_matrix_xyzw(orientation) @ np.array([0.0, 1.0, 0.0]),
        quat_to_matrix_xyzw(orientation) @ np.array([0.0, 0.0, 1.0]),
    )
    meshes = []
    for axis, color in zip(axes, colors):
        q = _axis_to_quat(axis)
        meshes.append(build_arrow(origin, q, axis_length, radius, segments=segments, color=color))
    return _merge_meshes(meshes)


def _axis_to_quat(axis):
    """把单位方向向量转成「局部 +z 指向它」的 XYZW 四元数。"""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z, a), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -1.0 + 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    axis_v = np.cross(z, a)
    s = np.sqrt((1.0 + dot) * 2.0)
    return [float(axis_v[0] / s), float(axis_v[1] / s), float(axis_v[2] / s), float(s / 2.0)]


def rasterize_mesh(img, vertices, faces, colors, intrinsics, projection,
                   alpha=0.45, supersample=2):
    """画家算法光栅化:背面剔除 + 面心深度从远到近 + 光照 + 超采样 alpha 合成。"""
    from PIL import Image, ImageDraw

    K = np.asarray(intrinsics, dtype=float).reshape(3, 3)
    Rt = decompose_projection(K, projection)
    verts = np.asarray(vertices, dtype=float)
    if verts.size == 0 or len(faces) == 0:
        return img
    cam = verts @ Rt[:, :3].T + Rt[:, 3]
    depth = cam[:, 2]
    px = cam @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        u = px[:, 0] / px[:, 2]
        v = px[:, 1] / px[:, 2]
    h, w = img.shape[:2]
    ss = max(1, int(supersample))
    overlay = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    light = np.array([0.3, -0.5, 1.0])
    light = light / np.linalg.norm(light)
    order = []
    for fi, tri in enumerate(faces):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if depth[i0] <= 0 or depth[i1] <= 0 or depth[i2] <= 0:
            continue
        normal = np.cross(cam[i1] - cam[i0], cam[i2] - cam[i0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            continue
        normal /= norm
        centroid = cam[[i0, i1, i2]].mean(axis=0)
        if float(np.dot(normal, centroid)) > 0:  # 法向统一朝向相机一侧
            normal = -normal
        order.append((float(centroid[2]), fi, normal))
    order.sort(key=lambda item: -item[0])  # 远的先画(画家算法,半透明双面叠加)
    for _, fi, normal in order:
        i0, i1, i2 = (int(x) for x in faces[fi])
        shade = 0.5 + 0.5 * max(0.0, float(np.dot(normal, light)))
        r, g, b = (np.clip(np.asarray(colors[fi], dtype=float) * shade, 0, 255)).tolist()
        pts = [(float(u[i]) * ss, float(v[i]) * ss) for i in (i0, i1, i2)]
        draw.polygon(pts, fill=(int(r), int(g), int(b), int(round(255 * alpha))))
    if ss != 1:
        overlay = overlay.resize((w, h), Image.LANCZOS)
    base = Image.fromarray(img).convert("RGBA")
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))


def draw_label(img, pixel, text, color=(255, 255, 255), font_px=18):
    """在像素位置绘制带描边的文字标签(居中)。"""
    if pixel is None:
        return img
    from PIL import Image, ImageDraw
    font = _load_grid_font(int(font_px))
    out = Image.fromarray(img)
    draw = ImageDraw.Draw(out)
    x, y = float(pixel[0]), float(pixel[1])
    if font is not None:
        box = draw.textbbox((0, 0), text, font=font)
        x -= (box[2] - box[0]) / 2
        y -= (box[3] - box[1]) / 2
        draw.text((x, y), text, fill=tuple(int(c) for c in color), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    else:
        draw.text((x, y), text, fill=tuple(int(c) for c in color))
    return np.asarray(out)
