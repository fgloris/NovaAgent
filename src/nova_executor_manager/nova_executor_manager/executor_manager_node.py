#!/usr/bin/env python3
# executor_manager:订阅各 executor 心跳维护工具注册表(热插拔),
# 提供 ListTools service 供 agentos 查询,并以 MCPExecute action 统一转发工具调用。
import time

import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from nova_interfaces.action import MCPExecute
from nova_interfaces.msg import ExecutorHeartbeat
from nova_interfaces.srv import ListTools

HEARTBEAT_TOPIC = "/nova/executors/heartbeat"


class ExecutorManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("nova_executor_manager")
        self.declare_parameter("heartbeat_timeout_sec", 5.0)
        self.declare_parameter("list_tools_service", "/nova/executor_manager/list_tools")
        self.declare_parameter("execute_action", "/nova/executor_manager/execute")
        self._timeout = float(self.get_parameter("heartbeat_timeout_sec").value)

        # tool_name -> {"desc": ToolDescriptor, "executor": str, "last_seen": float}
        self._registry: dict[str, dict] = {}
        # 转发用 ActionClient 缓存;注意不能叫 _clients,rclpy Node 内部占用该名字
        self._action_clients: dict[str, ActionClient] = {}
        # 独立 callback group:execute 回调内同步等待转发结果,避免与默认组互斥卡死
        self._fwd_cg = MutuallyExclusiveCallbackGroup()

        self.create_subscription(ExecutorHeartbeat, HEARTBEAT_TOPIC, self._on_heartbeat, 10)
        self.create_service(
            ListTools, str(self.get_parameter("list_tools_service").value), self._list_tools_cb
        )
        self._action_server = ActionServer(
            self, MCPExecute, str(self.get_parameter("execute_action").value), self._execute_cb
        )
        self.create_timer(1.0, self._expire_check)

    def _on_heartbeat(self, hb: ExecutorHeartbeat) -> None:
        now = time.time()
        for tool in hb.tools:
            entry = self._registry.get(tool.name)
            if entry is None:
                self.get_logger().info(f"发现工具 {tool.name}(executor: {hb.executor_name})")
                self._registry[tool.name] = {"desc": tool, "executor": hb.executor_name, "last_seen": now}
            else:
                entry["desc"] = tool
                entry["executor"] = hb.executor_name
                entry["last_seen"] = now

    def _expire_check(self) -> None:
        now = time.time()
        for name in [n for n, e in self._registry.items() if now - e["last_seen"] > self._timeout]:
            entry = self._registry.pop(name)
            self._action_clients.pop(entry["desc"].action_server_name, None)
            self.get_logger().warn(f"工具 {name} 下线(心跳超时,executor: {entry['executor']})")

    def _list_tools_cb(self, request, response):
        del request
        response.tools = [e["desc"] for e in self._registry.values()]
        return response

    def _get_client(self, action_server_name: str) -> ActionClient:
        client = self._action_clients.get(action_server_name)
        if client is None:
            client = ActionClient(self, MCPExecute, action_server_name, callback_group=self._fwd_cg)
            self._action_clients[action_server_name] = client
        return client

    def _execute_cb(self, goal_handle):
        goal = goal_handle.request
        entry = self._registry.get(goal.tool_name)
        result = MCPExecute.Result()
        if entry is None:
            result.success = False
            result.error = f"工具 '{goal.tool_name}' 未注册"
            goal_handle.abort()
            return result

        client = self._get_client(entry["desc"].action_server_name)
        if not client.wait_for_server(timeout_sec=5.0):
            result.success = False
            result.error = f"executor action server {entry['desc'].action_server_name} 不可用"
            goal_handle.abort()
            return result

        fwd_goal = MCPExecute.Goal()
        fwd_goal.tool_name = goal.tool_name
        fwd_goal.params_json = goal.params_json
        fwd_goal.trace_id = goal.trace_id
        self.get_logger().info(f"转发工具 {goal.tool_name} -> {entry['desc'].action_server_name}")

        send_future = client.send_goal_async(
            fwd_goal, feedback_callback=self._make_feedback_forward(goal_handle)
        )
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        if not send_future.done():
            result.success = False
            result.error = "发送 goal 超时"
            goal_handle.abort()
            return result
        goal_ref = send_future.result()
        if not goal_ref.accepted:
            result.success = False
            result.error = f"executor 拒绝 goal({goal.tool_name})"
            goal_handle.abort()
            return result

        result_future = goal_ref.get_result_async()
        while rclpy.ok() and not result_future.done():
            if goal_handle.is_cancel_requested:
                cancel_future = client.cancel_goal_async(goal_ref)
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
                result.success = False
                result.error = "任务被取消"
                goal_handle.canceled()
                return result
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=0.1)

        if not result_future.done():
            result.success = False
            result.error = "执行超时"
            goal_handle.abort()
            return result

        fwd_result = result_future.result().result
        result.success = fwd_result.success
        result.result_json = fwd_result.result_json
        result.error = fwd_result.error
        if fwd_result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _make_feedback_forward(self, goal_handle):
        def forward(feedback_msg):
            fb = MCPExecute.Feedback()
            fb.status = feedback_msg.feedback.status
            fb.message = feedback_msg.feedback.message
            goal_handle.publish_feedback(fb)

        return forward


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ExecutorManagerNode()
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
