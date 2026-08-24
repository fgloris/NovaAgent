#!/usr/bin/env python3
# 示例 executor:演示如何接入 NovaAgent。
# 每个工具一个 MCPExecute action server,周期发布心跳供 executor_manager 发现。
import json
import time

import rclpy
from rclpy.node import Node

from nova_interfaces.action import MCPExecute
from nova_interfaces.msg import ExecutorHeartbeat, ToolDescriptor

HEARTBEAT_TOPIC = "/nova/executors/heartbeat"

# 工具名 -> (执行函数, 参数 JSON Schema)
TOOL_SPECS = {
    "wait": (
        "阻塞指定秒数,用于在任务中插入时间间隔",
        {"type": "object", "properties": {"duration_sec": {"type": "number", "description": "等待秒数"}}, "required": ["duration_sec"]},
    ),
    "echo": (
        "返回传入的文本,用于验证工具调用链路",
        {"type": "object", "properties": {"text": {"type": "string", "description": "要回显的文本"}}, "required": ["text"]},
    ),
    "grasp": (
        "模拟抓取一个物体(仿真用,真实环境替换为 VLA/机械臂执行器)",
        {"type": "object", "properties": {"object": {"type": "string", "description": "要抓取的物体名"}}, "required": ["object"]},
    ),
    "place": (
        "模拟放置物体到指定位置(仿真用,真实环境替换为 VLA/机械臂执行器)",
        {"type": "object", "properties": {"object": {"type": "string"}, "surface": {"type": "string", "description": "放置目标,如桌子/台面"}}, "required": ["object", "surface"]},
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


_HANDLERS = {"wait": _run_wait, "echo": _run_echo, "grasp": _run_grasp, "place": _run_place}


class ExecutorDemoNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_executor_demo")
        self.declare_parameter("heartbeat_rate_hz", 1.0)
        rate = float(self.get_parameter("heartbeat_rate_hz").value)

        for name, (desc, schema) in TOOL_SPECS.items():
            action_server = f"/{self.get_name()}/{name}/execute"
            self.create_action_server(MCPExecute, action_server, self._make_execute_cb(name))
            self.get_logger().info(f"tool {name} @ {action_server}")

        self._heartbeat_pub = self.create_publisher(ExecutorHeartbeat, HEARTBEAT_TOPIC, 1)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_heartbeat)
        self._publish_heartbeat()

    def _publish_heartbeat(self) -> None:
        hb = ExecutorHeartbeat()
        hb.executor_name = self.get_name()
        for name, (desc, schema) in TOOL_SPECS.items():
            tool = ToolDescriptor()
            tool.name = name
            tool.description = desc
            tool.params_schema_json = json.dumps(schema)
            tool.action_server_name = f"/{self.get_name()}/{name}/execute"
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
                result_json = _HANDLERS[name](params)
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
