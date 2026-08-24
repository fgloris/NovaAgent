# nova_interfaces

NovaAgent 共享的 ROS 2 消息/服务/动作接口定义(ament_cmake 包)。

## 消息 (msg)

| 名称 | 说明 |
| --- | --- |
| `DAGNode` | 任务图节点,每个节点 = 一次工具调用(`id, type, tool_name, params_json, depends_on, goal`) |
| `TaskGraph` | 格式化后的任务 DAG 图 |
| `ToolDescriptor` | 一个 MCP 工具的能力描述(`name, description, params_schema_json, action_server_name`) |
| `ExecutorHeartbeat` | executor 周期发布的心跳,供 manager 自动发现/剔除 |
| `SkillInfo` / `SkillCommand` / `SkillResult` | skill 元数据与执行命令/结果 |
| `TaskState` | 任务执行状态(`planning/executing/done/failed`) |

## 服务 (srv)

- `ListTools`:查询 executor_manager 当前全部可用工具。
- `RunTask`:提交一条任务指令给 agentos(阻塞直到执行完成)。

## 动作 (action)

- `MCPExecute`:统一工具执行动作。所有 executor 都实现此 action,executor_manager 负责按 `tool_name` 路由转发。goal 含 `tool_name/params_json/trace_id`,feedback 提供中间状态,result 返回结构化结果。
