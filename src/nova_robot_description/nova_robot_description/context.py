"""把机器人描述渲染成 JSON 或 Markdown,供 CLI 输出。"""
import argparse, json
from dataclasses import asdict
from .loaders import load_robot_description

def to_json(d):
    """把描述对象递归转成可 JSON 序列化的 dict。"""
    return asdict(d)

def to_markdown(d):
    """把描述对象渲染成便于阅读的 Markdown 文本。"""
    groups = "\n".join(f"- {k}: {', '.join(v)}" for k,v in d.joint_groups.items())
    return f"Robot: {d.robot_type}\nBase frame: {d.base_frame}\nEEF frame: {d.eef_frame}\n\nJoint groups:\n{groups}\n\nControllers:\n{json.dumps(d.controllers, ensure_ascii=False)}\n\nSafety limits:\n{json.dumps(d.safety, ensure_ascii=False)}\n\nState mapping:\n{json.dumps(d.state_mapping, ensure_ascii=False)}"

def main():
    """命令行入口:按 --json 选择输出格式并打印机器人描述。"""
    p=argparse.ArgumentParser(); p.add_argument("--name",default="panda_omron"); p.add_argument("--json",action="store_true"); a=p.parse_args(); d=load_robot_description(a.name); print(json.dumps(to_json(d),ensure_ascii=False) if a.json else to_markdown(d))
