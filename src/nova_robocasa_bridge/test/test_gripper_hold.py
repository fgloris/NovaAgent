"""回归:夹爪命令在缺省路径点与空闲零动作中都要保持,避免被自动张开。"""
from nova_robocasa_bridge.robocasa_bridge_node import RoboCasaBridgeNode


class _WP:
    def __init__(self, has_gripper=False, gripper=0.0):
        self.position = [1.0, 2.0, 3.0]
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        self.has_gripper = has_gripper
        self.gripper = gripper


def _bare():
    node = RoboCasaBridgeNode.__new__(RoboCasaBridgeNode)
    node.action_spec = {"dim": 12, "meaning": []}
    node._last_gripper = -1.0
    return node


def test_waypoint_dict_holds_last_gripper_when_omitted():
    node = _bare()
    node._waypoint_dict(_WP(has_gripper=True, gripper=1.0))
    item = node._waypoint_dict(_WP(has_gripper=False))
    assert item["gripper"] == 1.0


def test_zero_action_preserves_last_gripper():
    node = _bare()
    node._last_gripper = 1.0
    action = node._zero_action()
    assert action[6] == 1.0
    assert all(action[i] == 0.0 for i in range(len(action)) if i != 6)
