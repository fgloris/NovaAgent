import json
import time
from types import SimpleNamespace

from nova_agentos.robot_state_observer import RobotStateObserver


class _FakeClient:
    def service_is_ready(self):
        return False

    def call_async(self, request):
        raise AssertionError("refresh should be skipped when service is not ready")


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass


class _FakeNode:
    def __init__(self):
        self.subscriptions = []

    def create_client(self, *_args, **_kwargs):
        return _FakeClient()

    def create_timer(self, *_args, **_kwargs):
        return None

    def create_subscription(self, _msg_type, topic, _cb, _qos):
        self.subscriptions.append(topic)

    def get_logger(self):
        return _Logger()


class _FakeFuture:
    def __init__(self, response):
        self._response = response

    def result(self):
        return self._response


def test_snapshot_returns_none_without_state():
    observer = RobotStateObserver(_FakeNode())
    assert observer.snapshot_message() is None


def test_snapshot_includes_latest_state():
    observer = RobotStateObserver(_FakeNode())
    now = time.time()
    observer._eef = ([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0], now)
    observer._joints = (["j1", "j2"], [0.1, 0.2], now)
    observer._gripper = ([0.01, 0.02], now)

    message = observer.snapshot_message()
    assert message["role"] == "user"
    payload = json.loads(message["content"].split("\n", 1)[1])
    assert payload["eef"]["position"] == [0.1, 0.2, 0.3]
    assert payload["joints"]["name"] == ["j1", "j2"]
    assert payload["gripper"]["position"] == [0.01, 0.02]


def test_info_done_discovers_state_topics_with_fallback():
    node = _FakeNode()
    observer = RobotStateObserver(node)
    response = SimpleNamespace(
        success=True,
        spec_json=json.dumps(
            {
                "robot": {
                    "robot_id": "robot0",
                    "state_topics": {"eef_pose": "/r/eef", "joint_states": "/r/joints"},
                }
            }
        ),
    )
    observer._info_done(_FakeFuture(response))
    assert "/r/eef" in node.subscriptions
    assert "/r/joints" in node.subscriptions
    assert "/robot0/gripper_state" in node.subscriptions
