# MCP 适配器:与 executor_manager 通信。
#   fetch_tools(): 查询工具注册表 -> 转成 LLM function/tool schema
#   execute(): 对 manager 的 MCPExecute action 发 goal 并等待结果
import json

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from nova_interfaces.action import MCPExecute
from nova_interfaces.msg import ToolDescriptor
from nova_interfaces.srv import ListTools


def to_llm_tools(descriptors: list) -> list[dict]:
    tools = []
    for d in descriptors:
        try:
            params = json.loads(d.params_schema_json) if d.params_schema_json else {}
        except json.JSONDecodeError:
            params = {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {"name": d.name, "description": d.description, "parameters": params},
            }
        )
    return tools


class McpAdapter:
    def __init__(self, node: Node, list_tools_srv: str, execute_action: str) -> None:
        self.node = node
        # 独立 callback group:回调内同步等待响应时不会被默认组占用而卡死
        cg = MutuallyExclusiveCallbackGroup()
        self._list = node.create_client(ListTools, list_tools_srv, callback_group=cg)
        self._client = ActionClient(node, MCPExecute, execute_action, callback_group=cg)

    def fetch_tools(self, timeout_sec: float = 10.0) -> list:
        if not self._list.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"executor_manager 服务 {self._list.srv_name} 不可用")
        future = self._list.call_async(ListTools.Request())
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        if not future.done():
            raise RuntimeError("查询工具列表超时")
        return future.result().tools

    def execute(self, tool_name: str, params: dict, trace_id: str, timeout_sec: float = 120.0) -> dict:
        if not self._client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"executor_manager action {self._client.action_name} 不可用")
        goal = MCPExecute.Goal()
        goal.tool_name = tool_name
        goal.params_json = json.dumps(params, ensure_ascii=False)
        goal.trace_id = trace_id

        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=10.0)
        if not send_future.done():
            raise RuntimeError(f"工具 {tool_name} 发送 goal 超时")
        goal_ref = send_future.result()
        if not goal_ref.accepted:
            raise RuntimeError(f"工具 {tool_name} goal 被拒绝")

        result_future = goal_ref.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout_sec)
        if not result_future.done():
            raise RuntimeError(f"工具 {tool_name} 执行超时")
        result = result_future.result().result
        if not result.success:
            raise RuntimeError(f"工具 {tool_name} 执行失败: {result.error}")
        return json.loads(result.result_json) if result.result_json else {}
