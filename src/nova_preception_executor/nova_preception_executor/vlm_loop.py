# VLM 多视图定位循环状态机。
#   第1轮:图上叠网格,让 VLM 给出各图网格单元(默认取中心像素)-> DLT 三角化 -> 重投影误差检查。
#   每轮分别维护两套像素位置:VLM 原始标注点(蓝圈)与系统 3D 定位在各图的重投影点(红圈)。
#   第2轮起:让 VLM 给出把蓝色圈移向物体所需的像素偏移量(dx, dy),由系统累加到蓝点后重新三角化。
#   结束条件 = 重投影误差低于阈值(蓝/红圈一致)且至少完成一轮像素微调;满足后才允许 VLM 返回 done,
#   否则提示词不给 done,只让其继续优化。
#   无效轮(VLM 提前 done/偏移量解析失败/三角化非有限)不再退回网格:保留上一轮正常定位状态,
#   直接重发一轮 refine 重试;重试耗尽或达到最大轮数时,若仍不满足结束条件则整次定位判失败。
import json
import time

import numpy as np

from nova_preception_executor.vision_geometry import (
    draw_grid,
    draw_marker,
    encode_image,
    grid_cell_to_pixel,
    parse_grid_cell,
    parse_pixel,
    project_point,
    reprojection_errors,
    sent_image_size,
    triangulate,
)

_RED = (255, 100, 100)
_BLUE = (100, 100, 255)


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
        max_reproj_error_px: float = 40.0,
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
        self.converged = False
        self.position = None
        self._failed_reason = ""
        self.observed: dict[str, tuple[float, float]] = {}
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
                "error": self._failed_reason or "定位失败(网格解析多次失败,或未满足结束条件)",
                "iterations": self.iterations,
                "restarts": self.restarts,
            }
        if self._failed_reason:
            return {
                "ok": False,
                "error": self._failed_reason,
                "iterations": self.iterations,
                "restarts": self.restarts,
            }
        return {
            "ok": True,
            "object": self.object,
            "position": self.position,
            "reprojection_error_px": round(self.error_px, 2),
            "converged": bool(self.converged),
            "iterations": self.iterations,
            "restarts": self.restarts,
            "camera_names": self.cams,
        }

    # 结束条件:多视图一致(误差低于阈值)且至少经历过一轮像素微调,防止 grid 粗定位结果直接判 done
    def _ready(self) -> bool:
        return bool(self.converged and self.iterations >= 2)

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

    # ---------- 第2轮起:以蓝色圈为基准的偏移量微调 ----------
    def _refine_round(self) -> None:
        marked = {
            c: self._annotate(self.images[c], self.observed.get(c), self.proj_px.get(c))
            for c in self.cams
        }
        lines = []
        for c in self.cams:
            bl = self.observed.get(c)
            rp = self.proj_px.get(c)
            if bl is None or rp is None:
                continue
            lines.append(
                f"相机 {c}: 蓝色圈(你的像素点)=({bl[0]:.1f}, {bl[1]:.1f}),"
                f"红色圈(系统重投影)=({rp[0]:.1f}, {rp[1]:.1f})"
            )
        if self._ready():
            tail = (
                "若你认为蓝色圈都已对准目标物体、"
                "无需再移动,可直接返回 {\"done\": true} 结束;否则继续给出偏移量。"
            )
        elif self.converged:
            tail = (
                "当前各视图已一致,但还需一轮像素微调把蓝色圈对准物体以确认,"
                "请给出偏移量,不要返回 done。"
            )
        else:
            tail = (
                "当前各视图重投影误差仍高于阈值(蓝圈与红圈不一致),尚未达到结束条件,"
                "请继续给出偏移量,直到多视图一致。"
            )
        prompt = (
            f"继续优化对物体\"{self.object}\"的定位。\n"
            + "\n".join(lines)
            + f"\n蓝色空心圆是你在对应图上标注的像素位置,红色空心圆是系统由这些点三角化后"
            f"的 3D 点在每张图的重投影;两圆重合表示多视图判断一致。\n"
            f"请把蓝色圆圆心移动到目标物体中心,给出每张图所需的像素偏移 [dx, dy]"
            f"(图像左上角为原点,向右为 +x,向下为 +y,无需移动给 [0, 0]),"
            f"严格返回 JSON: {{\"<相机名>\": [dx, dy], ...}}。{tail}不要输出其它内容。"
        )
        reply = self._ask_vlm(prompt, marked, f"refine{self.iterations + 1}")
        data = _extract_json(reply)
        if isinstance(data, dict) and data.get("done"):
            if self._ready():
                self._push("vlm", "DONE")
                return
            self._push("vlm", reply)
            if not self._retry_refine_or_adopt("VLM 提前返回 done 但尚未满足结束条件,不采纳。"):
                return
            self._refine_round()
            return
        new_points: dict[str, tuple[float, float]] = {}
        deltas: dict[str, tuple[float, float]] = {}
        for c in self.cams:
            delta = parse_pixel(data.get(c)) if isinstance(data, dict) else None
            if delta is None:
                self._push("vlm", reply)
                if not self._retry_refine_or_adopt("偏移量解析失败,未得到有效 [dx, dy]。"):
                    return
                self._refine_round()
                return
            base = self.observed.get(c)
            if base is None:
                if not self._retry_refine_or_adopt(f"相机 {c} 缺少基准蓝色点。"):
                    return
                self._refine_round()
                return
            deltas[c] = delta
            # VLM 的偏移量在"发送图"坐标系(可能被等比缩小),按缩放比回乘到原始像素后再三角化
            h, w = self.images[c].shape[:2]
            sw, sh = sent_image_size(h, w)
            new_points[c] = (base[0] + delta[0] * (w / sw), base[1] + delta[1] * (h / sh))
        self._push(
            "vlm",
            " | ".join(f"{c}: Δ({deltas[c][0]:+.1f}, {deltas[c][1]:+.1f})" for c in self.cams),
        )
        self._settle(new_points)

    # 无效轮不再退回网格:保留上一轮正常定位状态(蓝点/红圈/position 均未动)重发一轮 refine。
    # 返回 False 表示重试预算耗尽;已达结束条件则采纳上一轮结果,否则整次定位判失败。
    def _retry_refine_or_adopt(self, reason: str) -> bool:
        self.restarts += 1
        if self.restarts > self.max_restarts:
            if self._ready():
                self._push(
                    "system",
                    f"{reason}已重试 {self.max_restarts} 次仍无效,采纳上一轮正常定位结果结束。",
                )
            else:
                self._failed_reason = f"{reason}重试耗尽且未满足结束条件(多视图一致 + 至少一轮像素微调)。"
                self._push("system", self._failed_reason)
            return False
        self._push(
            "system",
            f"{reason}丢弃本轮,仍从上一轮正常状态继续微调(第 {self.restarts} 次重试)。",
        )
        return True

    # DLT 三角化 + 重投影误差检查 + 记录原始点(蓝)与反投影点(红)
    def _settle(self, points: dict[str, tuple[float, float]]) -> None:
        pts = [points[c] for c in self.cams]
        projs = [self.projections[c] for c in self.cams]
        X = triangulate(pts, projs)
        errors = reprojection_errors(pts, projs, X)
        if not (np.isfinite(X).all() and all(np.isfinite(e) for e in errors)):
            self._push("system", "三角化/重投影出现非有限值(点可能在相机后方)。")
            if not self.observed:
                # 尚无上一轮正常定位(冷启动网格轮失败),只能重新网格定位
                self._push("system", "尚无上一轮正常定位,重新用网格单元粗定位。")
                self.position = None
                return
            if not self._retry_refine_or_adopt("应用本轮偏移后三角化失败。"):
                return
            self._refine_round()
            return
        self.error_px = sum(errors) / len(errors)
        self.converged = self.error_px <= self.max_reproj_error_px
        self.position = X
        self.observed = dict(points)
        self.proj_px = {
            c: project_point(self.projections[c], X)
            for c in self.cams
        }
        self.iterations += 1
        state = "一致" if self.converged else f"不一致(阈值 {self.max_reproj_error_px:.0f}px)"
        self._push(
            "system",
            f"三角化 X={[round(v, 3) for v in X]}, 平均重投影误差={self.error_px:.2f}px [{state}]",
        )

        if self.iterations >= self.max_rounds:
            if self._ready():
                self._push("system", "达到最大调整轮数,已多视图一致,采纳当前结果。")
            else:
                self._failed_reason = (
                    f"达到最大轮数 {self.max_rounds} 仍未多视图一致"
                    f"(误差 {self.error_px:.2f}px > 阈值 {self.max_reproj_error_px:.0f}px)。"
                )
                self._push("system", self._failed_reason)
            return
        self._refine_round()

    # 标注图像:蓝色空心圆 = VLM 标注的原始像素点;红色空心圆 = 系统重投影点
    def _annotate(self, img, blue_pixel, red_pixel):
        out = img.copy()
        if blue_pixel is not None:
            out = draw_marker(out, blue_pixel, _BLUE, radius=8)
        if red_pixel is not None:
            out = draw_marker(out, red_pixel, _RED, radius=8)
        return out

    # 发一轮 VLM 请求:组装多图内容,记录历史(prompt+reply)并通过 on_round 发布本轮输入/输出
    def _ask_vlm(self, prompt: str, imgs: dict[str, object], round_tag: str) -> str:
        # 提示词末尾追加各图实际发送尺寸(encode 可能已等比缩小),坐标以发送图为准
        sizes = {c: sent_image_size(*imgs[c].shape[:2]) for c in self.cams}
        size_text = "、".join(f"{c}={w}x{h}" for c, (w, h) in sizes.items())
        prompt = f"{prompt}\n\n图像像素尺寸:{size_text}"
        self._push("prompt", f"[{round_tag}] {prompt}")
        self._dump_history(round_tag)
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
                    "content": f"以下是 agent 的对话上下文(供你理解任务意图):\n{self.agent_context}",
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

    # debug:每次发请求前把完整历史(prompt/vlm/system)打到进程日志
    def _dump_history(self, round_tag: str) -> None:
        print(f"\n[vlm_loop] ==== task={self.task_id} round={round_tag} 请求历史(共 {len(self.history)} 条) ====", flush=True)
        for idx, entry in enumerate(self.history):
            text = str(entry.get("content", "")).replace("\n", "\\n")
            if len(text) > 2000:
                text = text[:2000] + f"...(截断,共{len(str(entry.get('content', '')))}字)"
            print(f"[vlm_loop] [{idx}][{entry.get('role')}] {text}", flush=True)
