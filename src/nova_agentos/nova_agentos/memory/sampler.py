"""图像历史采样:定时把各链路最新帧按去重阈值落盘为离散历史图。"""
from __future__ import annotations

from typing import Callable

from rclpy.node import Node

from nova_agentos.memory.images import ImageMemory


class ImageSampler:
    """按固定周期从 frame_provider 取各链路最新帧,去重后写入当前 session 的 ImageMemory。"""

    def __init__(
        self,
        node: Node,
        frame_provider: Callable[[], dict],
        memory_provider: Callable[[], ImageMemory | None],
        state_provider: Callable[[], list[float] | None] | None = None,
        period_sec: float = 1.0,
    ) -> None:
        self._node = node
        self._provider = frame_provider
        self._memory_provider = memory_provider
        self._state_provider = state_provider
        node.create_timer(max(0.1, float(period_sec)), self.tick)

    def tick(self) -> None:
        """采样一次:所有链路画面与状态都几乎不变才跳过,否则整组落盘。"""
        memory = self._memory_provider()
        if memory is None:
            return
        try:
            frames = self._provider()
        except Exception as exc:
            self._node.get_logger().warn(f"图像采样失败: {exc}")
            return
        if not frames:
            return
        state = None
        if self._state_provider is not None:
            try:
                state = self._state_provider()
            except Exception:
                state = None
        try:
            memory.save_snapshot(frames, state)
        except Exception as exc:
            self._node.get_logger().warn(f"图像采样落盘失败: {exc}")
