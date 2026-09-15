from dataclasses import dataclass, field
from typing import Any

@dataclass
class RobotDescription:
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
