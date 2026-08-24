# 统一 LLM 配置加载。
# 查找顺序:环境变量 NOVA_LLM_CONFIG -> 包内 share/nova_common/config/llm.yaml。
import os
from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
except Exception:  # 未 source ROS 环境时退化为相对路径
    get_package_share_directory = None  # type: ignore
    PackageNotFoundError = FileNotFoundError


def find_config_path() -> Path | None:
    env = os.environ.get("NOVA_LLM_CONFIG")
    if env:
        p = Path(env)
        if p.exists():
            return p
    if get_package_share_directory is not None:
        try:
            p = Path(get_package_share_directory("nova_common")) / "config" / "llm.yaml"
            if p.exists():
                return p
        except PackageNotFoundError:
            pass
    p = Path(__file__).resolve().parent / "config" / "llm.yaml"
    return p if p.exists() else None


def load() -> dict:
    path = find_config_path()
    if path is None:
        raise RuntimeError("找不到 llm.yaml(可设置环境变量 NOVA_LLM_CONFIG 或先构建 nova_common)")
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
