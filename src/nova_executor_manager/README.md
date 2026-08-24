# nova_executor_manager

NovaAgent 的 executor 管理器,agentos 只与它对话,不感知具体 executor。

## 职责

- **工具注册表**:订阅各 executor 的 `/nova/executors/heartbeat` 心跳,聚合全部可用工具(热插拔);心跳超时(`heartbeat_timeout_sec`)自动剔除。
- **`ListTools` 服务**:返回当前全部可用工具的工具描述,供 agentos 生成 LLM tool schema。
- **`MCPExecute` 动作转发**:收到工具调用 goal 后,按 `tool_name` 查表路由到对应 executor 的 `MCPExecute` action server,feedback 透传、支持取消。

## 运行

```bash
ros2 run nova_executor_manager nova_executor_manager_node
```

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `heartbeat_timeout_sec` | 5.0 | 心跳超时,超过则剔除该 executor 的工具 |
| `list_tools_service` | `/nova/executor_manager/list_tools` | 工具查询服务名 |
| `execute_action` | `/nova/executor_manager/execute` | 工具执行 action 名 |
