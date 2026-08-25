#!/usr/bin/env python3
# 示例 executor:演示如何接入 NovaAgent。
# 每个工具一个 MCPExecute action server,周期发布心跳供 executor_manager 发现。
# pi0_policy 演示 VLA 语义:由 AgentOS 注入 topic_namespace,订阅 session 相机/state 并把动作回灌仿真。
import json
import random
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from nova_interfaces.action import MCPExecute
from nova_interfaces.msg import ExecutorHeartbeat, ToolDescriptor

HEARTBEAT_TOPIC = "/nova/executors/heartbeat"

# 工具名 -> (执行函数, 参数 JSON Schema, obs_bindings JSON)
TOOL_SPECS = {
    "wait": (
        "阻塞指定秒数,用于在任务中插入时间间隔",
        {"type": "object", "properties": {"duration_sec": {"type": "number", "description": "等待秒数"}}, "required": ["duration_sec"]},
        "",
    ),
    "echo": (
        "返回传入的文本,用于验证工具调用链路",
        {"type": "object", "properties": {"text": {"type": "string", "description": "要回显的文本"}}, "required": ["text"]},
        "",
    ),
    "grasp": (
        "模拟抓取一个物体(仿真用,真实环境替换为 VLA/机械臂执行器)",
        {"type": "object", "properties": {"object": {"type": "string", "description": "要抓取的物体名"}}, "required": ["object"]},
        "",
    ),
    "place": (
        "模拟放置物体到指定位置(仿真用,真实环境替换为 VLA/机械臂执行器)",
        {"type": "object", "properties": {"object": {"type": "string"}, "surface": {"type": "string", "description": "放置目标,如桌子/台面"}}, "required": ["object", "surface"]},
        "",
    ),
    "pi0_policy": (
        "VLA 演示工具:订阅 AgentOS 注入的 session 命名空间下相机与 state,持续把随机动作回灌到仿真,运行 duration_sec 秒",
        {"type": "object", "properties": {
            "topic_namespace": {"type": "string", "description": "AgentOS 注入的 session 命名空间,无需用户提供"},
            "duration_sec": {"type": "number", "description": "策略运行秒数"},
        }, "required": ["topic_namespace"]},
        {"cameras": ["agentview"], "state": ["robot0_eef_pos"]},
    ),
}


def _run_wait(params: dict) -> dict:
    seconds = float(params.get("duration_sec", 0.0))
    time.sleep(seconds)
    return {"waited_sec": seconds}


def _run_echo(params: dict) -> dict:
    return {"echo": params.get("text", "")}


def _run_grasp(params: dict) -> dict:
    return {"grasped": params.get("object", "")}


def _run_place(params: dict) -> dict:
    return {"placed": params.get("object", ""), "surface": params.get("surface", "")}


class ExecutorDemoNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_executor_demo")
        self.declare_parameter("heartbeat_rate_hz", 1.0)
        rate = float(self.get_parameter("heartbeat_rate_hz").value)

        self._handlers = {
            "wait": _run_wait,
            "echo": _run_echo,
            "grasp": _run_grasp,
            "place": _run_place,
            "pi0_policy": self._run_pi0_policy,
        }

        for name, (desc, schema, _) in TOOL_SPECS.items():
            action_server = f"/{self.get_name()}/{name}/execute"
            self.create_action_server(MCPExecute, action_server, self._make_execute_cb(name))
            self.get_logger().info(f"tool {name} @ {action_server}")

        self._heartbeat_pub = self.create_publisher(ExecutorHeartbeat, HEARTBEAT_TOPIC, 1)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_heartbeat)
        self._publish_heartbeat()

    def _publish_heartbeat(self) -> None:
        hb = ExecutorHeartbeat()
        hb.executor_name = self.get_name()
        for name, (desc, schema, bindings) in TOOL_SPECS.items():
            tool = ToolDescriptor()
            tool.name = name
            tool.description = desc
            tool.params_schema_json = json.dumps(schema)
            tool.action_server_name = f"/{self.get_name()}/{name}/execute"
            tool.obs_bindings = bindings if isinstance(bindings, str) else json.dumps(bindings)
            hb.tools.append(tool)
        self._heartbeat_pub.publish(hb)

    def _make_execute_cb(self, name):
        def execute(goal_handle):
            goal = goal_handle.request
            try:
                params = json.loads(goal.params_json) if goal.params_json else {}
                feedback = MCPExecute.Feedback()
                feedback.status = "running"
                feedback.message = f"{name} 开始执行"
                goal_handle.publish_feedback(feedback)
                result_json = self._handlers[name](params)
                result = MCPExecute.Result()
                result.success = True
                result.result_json = json.dumps(result_json, ensure_ascii=False)
                result.error = ""
                goal_handle.succeed()
                self.get_logger().info(f"tool {name} 完成: {result_json}")
                return result
            except Exception as exc:
                result = MCPExecute.Result()
                result.success = False
                result.result_json = ""
                result.error = str(exc)
                goal_handle.abort()
                self.get_logger().error(f"tool {name} 失败: {exc}")
                return result

        return execute

    # VLA 演示:订阅 <ns>/camera/agentview + <ns>/state,收到一帧后周期发随机动作到 <ns>/action_cmd
    def _run_pi0_policy(self, params: dict) -> dict:
        ns = str(params.get("topic_namespace", "")).rstrip("/")
        duration_sec = float(params.get("duration_sec", 3.0))
        if not ns:
            return {"ok": False, "executed": False, "error": "缺少 topic_namespace(需 AgentOS 注入)"}

        state = {"camera_frame": None, "action_dim": 0}
        cam_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        sub_cam = self.create_subscription(Image, f"{ns}/camera/agentview", self._make_cam_cb(state), cam_qos)
        sub_state = self.create_subscription(String, f"{ns}/state", self._make_state_cb(state), 10)
        pub_action = self.create_publisher(Float32MultiArray, f"{ns}/action_cmd", 10)
        try:
            # 等相机帧与 state(含 action_spec)都就绪,容忍动态 topic discovery 延迟
            deadline = time.time() + 5.0
            while time.time() < deadline and (
                state["camera_frame"] is None or state["action_dim"] <= 0
            ):
                rclpy.spin_once(self, timeout_sec=0.2)
            if state["camera_frame"] is None:
                self.get_logger().warn(f"pi0_policy 未收到相机帧({ns}/camera/agentview),跳过")
                return {"ok": False, "executed": False, "error": "未收到相机帧"}

            dim = state["action_dim"] if state["action_dim"] > 0 else 7
            start = time.time()
            count = 0
            while time.time() - start < duration_sec:
                rclpy.spin_once(self, timeout_sec=0.1)
                values = [random.uniform(-0.05, 0.05) for _ in range(dim)]
                if dim > 6:
                    values[6] = random.choice([0.0, 1.0])
                msg = Float32MultiArray()
                msg.data = values
                pub_action.publish(msg)
                count += 1
                time.sleep(0.2)
            return {"ok": True, "executed": True, "steps": count, "action_dim": dim}
        finally:
            self.destroy_subscription(sub_cam)
            self.destroy_subscription(sub_state)
            self.destroy_publisher(pub_action)

    @staticmethod
    def _make_cam_cb(state: dict):
        def cb(msg):
            state["camera_frame"] = msg

        return cb

    @staticmethod
    def _make_state_cb(state: dict):
        def cb(msg):
            try:
                doc = json.loads(msg.data)
                spec = doc.get("action_spec", {})
                dim = int(spec.get("dim", 0))
                if dim > 0:
                    state["action_dim"] = dim
            except Exception:
                pass

        return cb


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ExecutorDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
