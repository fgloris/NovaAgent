# VLM 多视图定位循环状态机。
#   第1轮:图上叠网格,让 VLM 给出各图网格单元(默认取中心像素)-> 三角化 -> 重投影误差检查。
#   第2轮起:网格仍画在图上,但直接让 VLM 给像素坐标;上一轮点(灰)与重投影点(黑)带坐标画在图上。
#   误差过大(多视图不一致)-> 丢本轮历史重来(回到网格轮);直到 VLM 满意(DONE)或达到上限。
import json
import time

from nova_preception_executor.vision_geometry import (
    draw_grid,
    draw_marker,
    encode_image,
    grid_cell_to_pixel,
    parse_grid_cell,
    parse_pixel,
    project_point,
    reprojection_errors,
    triangulate,
)

_GRAY = (150, 150, 150)
_BLACK = (0, 0, 0)


def _extract_text(result) -> str:
    try:
        content = result.raw["choices"][0]["message"].get("content") or ""
    except Exception:
        content = result.content or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


def _extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    return None


class VlmLocator:
    def __init__(
        self,
        llm,
        grid_size: int = 8,
        max_rounds: int = 5,
        max_restarts: int = 2,
        max_reproj_error_px: float = 25.0,
    ) -> None:
        self.llm = llm
        self.grid_size = max(2, int(grid_size))
        self.max_rounds = max(2, int(max_rounds))
        self.max_restarts = max(0, int(max_restarts))
        self.max_reproj_error_px = float(max_reproj_error_px)

    def locate(
        self,
        object_desc: str,
        images: dict[str, object],
        projections: dict[str, object],
        agent_context: str = "",
        task_id: str = "",
        on_round=None,
    ) -> dict:
        cams = [c for c in images if c in projections]
        if len(cams) < 2:
            return {"ok": False, "error": f"可用相机不足(需要>=2 且都有投影矩阵,现有 {cams})"}
        self.images = images
        self.projections = projections
        self.cams = cams
        self.object = object_desc
        self.agent_context = (agent_context or "").strip()
        self.task_id = task_id or ""
        self.on_round = on_round
        self.history: list[dict] = []
        self.iterations = 0
        self.restarts = 0
        self.error_px = None
        self.position = None
        self.last_points: dict[str, tuple[float, float]] = {}
        self.proj_px: dict[str, tuple[float, float]] = {}

        while self.position is None:
            if self.restarts > 0:
                self._push("system", f"第 {self.restarts} 次重来:重新用网格单元定位。")
            self._grid_round()
            if self.position is None:
                if self.restarts >= self.max_restarts:
                    break
                self.restarts += 1

        if self.position is None:
            return {
                "ok": False,
                "error": "定位失败(网格/像素解析失败或重投影误差始终超限)",
                "iterations": self.iterations,
                "restarts": self.restarts,
                "history": self.history,
            }
        return {
            "ok": True,
            "object": self.object,
            "position": self.position,
            "reprojection_error_px": round(self.error_px, 2),
            "iterations": self.iterations,
            "restarts": self.restarts,
            "camera_names": self.cams,
            "final_points": {c: [round(p[0], 1), round(p[1], 1)] for c, p in self.last_points.items()},
            "history": self.history,
        }

    # ---------- 第1轮:网格定位 ----------
    def _grid_round(self) -> None:
        grid_imgs = {c: draw_grid(self.images[c], self.grid_size) for c in self.cams}
        prompt = (
            f"你要帮助把物体定位到 3D 世界坐标。目标物体是\"{self.object}\"。\n"
            f"以下 {len(self.cams)} 张图来自不同视角,每张都叠加了 {self.grid_size}x{self.grid_size} 的网格:"
            f"行自上而下编号 1..{self.grid_size},列自左而右编号 1..{self.grid_size},"
            f"网格单元写作 \"行-列\"(例如 \"3-5\" 表示第3行第5列),选择某格即默认取其中心像素。\n"
            f"请对每张图给出目标物体所在的网格单元(若该图看不到物体给 null),"
            f"严格返回 JSON: {{\"<相机名>\": \"行-列\", ...}} 相机名为: {self.cams}。不要输出其它内容。"
        )
        reply = self._ask_vlm(prompt, grid_imgs, "grid")
        data = _extract_json(reply)
        points: dict[str, tuple[float, float]] = {}
        for c in self.cams:
            cell = parse_grid_cell(data.get(c)) if isinstance(data, dict) else None
            if cell is None:
                self._push("vlm", reply)
                self._push("system", "网格解析失败(未得到有效 \"行-列\" 单元)。")
                return
            h, w = self.images[c].shape[:2]
            points[c] = grid_cell_to_pixel(*cell, self.grid_size, h, w)
            self._push("vlm", f"{c}: 网格 {cell[0]}-{cell[1]} -> 像素 ({points[c][0]:.1f}, {points[c][1]:.1f})")
        self._settle(points)

    # ---------- 第2轮起:像素微调 ----------
    def _refine_round(self) -> None:
        marked = {
            c: self._annotate(self.images[c], self.last_points.get(c), self.proj_px.get(c))
            for c in self.cams
        }
        lines = []
        for c in self.cams:
            if c in self.last_points and c in self.proj_px:
                px = self.last_points[c]
                rp = self.proj_px[c]
                lines.append(
                    f"相机 {c}: 上一轮你给出的点是灰色,坐标 ({px[0]:.1f}, {px[1]:.1f});"
                    f"系统重投影点是黑色,坐标 ({rp[0]:.1f}, {rp[1]:.1f})"
                )
        prompt = (
            f"继续定位物体\"{self.object}\"。\n"
            + "\n".join(lines)
            + f"\n网格仍画在图上作为参考。请给出调整后的像素坐标(图像左上角为原点,向右为x,向下为y),"
            f"每张图一个,严格返回 JSON: {{\"<相机名>\": [x, y], ...}};"
            f"若你认为当前定位已足够准确,严格返回 {{\"done\": true}}。不要输出其它内容。"
        )
        reply = self._ask_vlm(prompt, marked, f"refine{self.iterations + 1}")
        data = _extract_json(reply)
        if isinstance(data, dict) and data.get("done"):
            self._push("vlm", "DONE")
            return
        points: dict[str, tuple[float, float]] = {}
        for c in self.cams:
            px = parse_pixel(data.get(c)) if isinstance(data, dict) else None
            if px is None:
                self._push("vlm", reply)
                self._push("system", "像素解析失败(未得到有效 [x, y])。")
                return
            points[c] = px
        self._push("vlm", " | ".join(f"{c}: ({p[0]:.1f}, {p[1]:.1f})" for c, p in points.items()))
        self._settle(points)

    # 三角化 + 重投影误差检查 + 记录反投影点
    def _settle(self, points: dict[str, tuple[float, float]]) -> None:
        pts = [points[c] for c in self.cams]
        projs = [self.projections[c] for c in self.cams]
        X = triangulate(pts, projs)
        errors = reprojection_errors(pts, projs, X)
        self.error_px = sum(errors) / len(errors)
        self.position = X
        self.last_points = dict(points)
        self.proj_px = {
            c: project_point(self.projections[c], X)
            for c in self.cams
        }
        self.iterations += 1
        self._push("system", f"三角化 X={[round(v, 3) for v in X]}, 重投影误差={self.error_px:.2f}px")

        if self.error_px > self.max_reproj_error_px:
            self._push(
                "system",
                f"重投影误差 {self.error_px:.2f}px 超过阈值 {self.max_reproj_error_px}px,"
                "视图间不一致,丢弃本轮结果重来。",
            )
            self.position = None
            return
        if self.iterations >= self.max_rounds:
            self._push("system", "达到最大调整轮数,采纳当前结果。")
            return
        self._refine_round()

    # 标注图像:灰点 = 上一轮 VLM 点,黑点 = 重投影点
    def _annotate(self, img, prev_pixel, proj_pixel):
        out = img.copy()
        if prev_pixel is not None:
            out = draw_marker(out, prev_pixel, _GRAY, radius=8)
        if proj_pixel is not None:
            out = draw_marker(out, proj_pixel, _BLACK, radius=8)
        return out

    # 发一轮 VLM 请求:组装多图内容,记录历史并通过 on_round 发布本轮输入/输出
    def _ask_vlm(self, prompt: str, imgs: dict[str, object], round_tag: str) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        urls: dict[str, str] = {}
        for c in self.cams:
            url = encode_image(imgs[c])
            urls[c] = url
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages: list[dict] = [
            {"role": "system", "content": "你是严谨的物体定位辅助,只输出要求的 JSON,不要解释。"},
            {"role": "user", "content": content},
        ]
        if self.agent_context:
            messages.insert(
                1,
                {
                    "role": "user",
                    "content": f"以下是 agent 的对话上下文(供你理解任务意图,不要重复):\n{self.agent_context}",
                },
            )
        t0 = time.time()
        result = self.llm.chat(messages, temperature=0.0, max_tokens=1024)
        elapsed = time.time() - t0
        reply = _extract_text(result)
        self._push("vlm", f"[{round_tag}][{elapsed:.1f}s] {reply}")
        if self.on_round is not None:
            try:
                self.on_round(
                    {
                        "round": round_tag,
                        "task_id": self.task_id,
                        "duration_sec": round(elapsed, 2),
                        "prompt": prompt,
                        "reply": reply,
                        "images": urls,
                    }
                )
            except Exception:
                pass
        return reply

    def _push(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
