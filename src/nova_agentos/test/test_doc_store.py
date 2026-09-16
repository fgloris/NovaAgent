"""doc 存储与 load_doc 多参数支持。"""
from pathlib import Path

from nova_agentos.agent_loop import TaskRunner
from nova_agentos.doc_store import DocStore

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
DOC_NAMES = [
    "reproject_pixels",
    "visualize_pixels",
    "visualize_grid",
    "visualize_frame",
    "visualize_point",
    "visualize_segment",
    "visualize_ray",
]


def test_doc_store_names_and_load():
    store = DocStore(DOCS_DIR)
    assert set(DOC_NAMES).issubset(set(store.names()))
    contents = store.load(["visualize_frame", "visualize_ray"])
    assert "3D 坐标系" in contents["visualize_frame"]
    assert "射线" in contents["visualize_ray"]


def test_load_doc_accepts_comma_string_and_list():
    runner = object.__new__(TaskRunner)
    runner.docs = DocStore(DOCS_DIR)
    text, images = runner._run_tool("load_doc", {"tool": "visualize_frame, visualize_ray"})
    assert images == {}
    assert "# doc: visualize_frame" in text
    assert "# doc: visualize_ray" in text
    listed, _ = runner._run_tool("load_doc", {"tool": ["visualize_point"]})
    assert "# doc: visualize_point" in listed


def test_load_doc_reports_missing():
    runner = object.__new__(TaskRunner)
    runner.docs = DocStore(DOCS_DIR)
    text, _ = runner._run_tool("load_doc", {"tool": "visualize_frame, not_a_doc"})
    assert "# doc: visualize_frame" in text
    assert "未找到 doc: not_a_doc" in text
