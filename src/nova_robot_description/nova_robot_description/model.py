from dataclasses import dataclass, field
from typing import Any

@dataclass
class RobotDescription:
    """机器人的结构化描述:连杆/关节、控制器、安全限位与观测/动作映射。"""

    robot_type: str
    robot_id: str
    base_frame: str
    eef_frame: str
    joints: list[dict[str, Any]] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    joint_groups: dict[str, list[str]] = field(default_factory=dict)
    controllers: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    state_mapping: dict[str, str] = field(default_factory=dict)
    action_mapping: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
