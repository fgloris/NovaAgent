import io

import numpy as np
from PIL import Image

from nova_agentos.memory import ImageMemory


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)


def _jpeg_bytes(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG")
    return buf.getvalue()


def test_history_dedup_and_retention(tmp_path):
    memory = ImageMemory(tmp_path, link_max=2, diff_mse_threshold=0.0005)
    frame = _frame(1)
    assert memory.save_history("camA", frame, 1.0) is not None
    assert memory.save_history("camA", frame.copy(), 2.0) is None  # 完全相同 -> 跳过
    changed = frame.copy()
    changed[:] = 255
    assert memory.save_history("camA", changed, 3.0) is not None
    assert memory.save_history("camA", _frame(2), 4.0) is not None
    records = memory.history_records("camA", 10)
    assert len(records) == 2  # link_max=2
    assert records[-1].time == 4.0
    assert all(r.url.startswith("file://history/") for r in records)


def test_processed_provenance_and_reload(tmp_path):
    memory = ImageMemory(tmp_path)
    base = memory.save_history("camA", _frame(3), 10.0)
    record = memory.save_processed(
        _jpeg_bytes(_frame(3)), "camA", "visualize_pixels", {"radius_px": 12}, base.url, 11.0
    )
    described = record.describe()
    assert described["origin"] == "tool"
    assert described["tool"] == "visualize_pixels"
    assert described["base"] == base.url
    assert described["size"] == [60, 40]  # 从 JPEG 解码出的真实尺寸

    reloaded = ImageMemory(tmp_path)
    assert len(reloaded.processed_records()) == 1
    assert reloaded.processed_records()[0].tool == "visualize_pixels"
    assert reloaded.find(base.url) is not None
    assert reloaded.nearest_history("camA", 10.4).time == 10.0


def test_clear_processed_removes_files_and_refs_but_keeps_history(tmp_path):
    memory = ImageMemory(tmp_path)
    base = memory.save_history("camA", _frame(7), 1.0)
    first = memory.save_processed(_jpeg_bytes(_frame(7)), "camA", "visualize_grid", {}, base.url, 2.0)
    second = memory.save_processed(_jpeg_bytes(_frame(8)), "camA", "visualize_grid", {}, base.url, 3.0)
    path = memory.path_of(first)
    assert path.exists()

    assert memory.clear_processed() == 2
    assert memory.processed_records() == []
    assert not path.exists()
    assert not memory.path_of(second).exists()
    assert memory.find(base.url) is not None  # history 不受影响


def test_refresh_current_overwrites_and_tracks_camera(tmp_path):
    memory = ImageMemory(tmp_path)
    memory.refresh_current({"camA": (_frame(5), 100.0)})
    memory.refresh_current({"camA": (_frame(6), 101.0)})
    records = memory.current_records()
    assert len(records) == 1
    assert records[0].camera == "camA"
    assert records[0].time == 101.0
    assert records[0].url == "file://current/camA.jpg"


def test_snapshot_gate_requires_all_links_and_state_unchanged(tmp_path):
    memory = ImageMemory(tmp_path, diff_mse_threshold=0.0005, state_diff_threshold=0.005)
    a, b = _frame(10), _frame(11)
    frames = {"camA": (a, 1.0), "camB": (b, 1.0)}

    assert memory.save_snapshot(frames, [0.04]) is not None  # 首帧保存
    # 画面与夹爪都不变 -> 跳过
    same = {"camA": (a.copy(), 2.0), "camB": (b.copy(), 2.0)}
    assert memory.save_snapshot(same, [0.04]) is None
    # 画面不变但夹爪变化 -> 保存(夹爪动作幅度小,MSE 反映不出)
    assert memory.save_snapshot(same, [0.0]) is not None
    # 夹爪不变但某一路画面变化 -> 保存整组
    changed = {"camA": (a.copy(), 4.0), "camB": (b.copy(), 4.0)}
    changed["camA"][0][:] = 255
    assert memory.save_snapshot(changed, [0.0]) is not None
    assert len(memory.history_records("camA", 10)) == 3
    assert len(memory.history_records("camB", 10)) == 3


def test_snapshot_same_stamp_not_recorded_twice(tmp_path):
    memory = ImageMemory(tmp_path, diff_mse_threshold=0.0005, state_diff_threshold=0.005)
    a = _frame(20)
    assert memory.save_snapshot({"camA": (a, 1.0)}, [0.04]) is not None
    # 同一帧 stamp 因夹爪变化再次触发 -> 不重复记录
    assert memory.save_snapshot({"camA": (a.copy(), 1.0)}, [0.0]) is None
    assert len(memory.history_records("camA", 10)) == 1
    # 帧更新(stamp 变化)且画面变化 -> 正常记录
    changed = a.copy()
    changed[:] = 255
    assert memory.save_snapshot({"camA": (changed, 2.0)}, [0.0]) is not None
    assert len(memory.history_records("camA", 10)) == 2
