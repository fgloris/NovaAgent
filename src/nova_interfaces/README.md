# nova_interfaces

NovaAgent 共享的 ROS 2 消息/服务/动作接口定义(ament_cmake 包)。

## 消息 (msg)

| 名称 | 说明 |
| --- | --- |
| `ToolDescriptor` | 一个 MCP 工具的能力描述(`name, description, params_schema_json, action_server_name`) |
| `ExecutorHeartbeat` | executor 周期发布的心跳,供 manager 自动发现/剔除 |
| `TaskState` | 任务状态(`working/done/failed` + `done` 标志 + 最后 message) |

## 服务 (srv)

- `ListTools`:查询 executor_manager 当前全部可用工具。
- `RunTask`:提交一条用户指令给 agentos(入队后立即返回 task_id,状态经 `/nova/agentos/task_state/t_{task_id}` 观察)。
- `EnvInfo`:查询仿真环境完整规格(`action_spec/obs_spec/cameras`),供手动诊断。

## 动作 (action)

- `MCPExecute`:统一工具执行动作。所有 executor 都实现此 action,executor_manager 负责按 `tool_name` 路由转发。goal 含 `tool_name/params_json/trace_id`,feedback 提供中间状态,result 返回结构化结果。
