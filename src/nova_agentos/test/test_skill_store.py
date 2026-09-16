"""skill 索引/加载与 load_skill 多参数支持。"""
from pathlib import Path

from nova_agentos.agent_loop import TaskRunner
from nova_agentos.skill_store import SkillStore

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


def test_skills_indexed_and_loaded():
    store = SkillStore(SKILLS_DIR)
    index = store.index_text()
    for name in ("locate_object", "tidy_table", "make_coffee"):
        assert name in index
    contents = store.load(["locate_object"])
    assert "三角化" in contents["locate_object"]


def test_load_skill_accepts_comma_string_and_list():
    runner = object.__new__(TaskRunner)
    runner.skills = SkillStore(SKILLS_DIR)
    text, images = runner._run_tool("load_skill", {"skill": "locate_object, tidy_table"})
    assert images == {}
    assert "# skill: locate_object" in text
    assert "# skill: tidy_table" in text
    listed, _ = runner._run_tool("load_skill", {"skill": ["make_coffee"]})
    assert "# skill: make_coffee" in listed


def test_load_skill_reports_missing():
    runner = object.__new__(TaskRunner)
    runner.skills = SkillStore(SKILLS_DIR)
    text, _ = runner._run_tool("load_skill", {"skill": "locate_object, not_a_skill"})
    assert "# skill: locate_object" in text
    assert "未找到 skill: not_a_skill" in text
