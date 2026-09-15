"""从 MJCF 片段与 YAML 配置装配机器人描述对象。"""
from pathlib import Path
import hashlib, json, xml.etree.ElementTree as ET
import yaml
from .model import RobotDescription

def _root():
    """返回包根目录(用于定位 descriptions/ 与 config/)。"""
    return Path(__file__).resolve().parents[1]

def load_robot_description(name="panda_omron", config_path=None):
    """加载指定机器人的描述:解析 MJCF 关节/连杆、合并 YAML 配置并计算文件哈希。

    依次读取 manifest.yaml 与各 XML 片段,汇总关节(含类型/轴/限位)与连杆,
    最后把所有描述文件的 sha256 汇总成一个版本指纹写入 source。
    """
    base = _root() / "descriptions" / name
    if not base.exists():
        base = Path(__file__).resolve().parent.parent / "share" / "descriptions" / name
    manifest = yaml.safe_load((base / "manifest.yaml").read_text())
    cfg = yaml.safe_load(Path(config_path).read_text()) if config_path else yaml.safe_load((_root()/"config"/f"{name}.yaml").read_text())
    joints, links = [], set()
    for fn in ("omron_mobile_base.xml", "panda_arm.xml", "panda_gripper.xml"):
        root = ET.parse(base / fn).getroot()
        for body in root.iter("body"):
            if body.get("name"): links.add(body.get("name"))
        for j in root.iter("joint"):
            if j.get("name"):
                item={"name":j.get("name"),"type":j.get("type","hinge"),"axis":j.get("axis","0 0 1")}
                if j.get("range"): item["limit"]=[float(x) for x in j.get("range").split()]
                joints.append(item)
    hashes=[]
    for p in sorted(base.iterdir()):
        if p.is_file(): hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
    return RobotDescription(manifest["robot_type"], cfg.get("robot_id", "robot0"), cfg["base_frame"], cfg["eef_frame"], joints, sorted(links), manifest.get("joint_groups",{}), cfg.get("controller",{}), cfg.get("safety",{}), cfg.get("observation",{}), cfg.get("action",{}), {"files": manifest.get("components",{}), "sha256": hashlib.sha256("".join(hashes).encode()).hexdigest()})
