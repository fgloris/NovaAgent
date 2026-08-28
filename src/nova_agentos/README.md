# nova_agentos

NovaAgent 核心:LLM 驱动的持续 AgentOS。负责 skill 管理、agent 循环、工具调度执行。

## 数据流

```
用户指令 → RunTask 服务(入队,立即返回 task_id)
        → 后台 agent 循环(上下文跨任务累积)
            每轮:LLM 发一个函数调用(executor 工具 / load_skill / finish)
             ├─ executor 工具 → executor_manager → 具体 executor
             ├─ load_skill   → 本地注入 SKILL.md 全文
             ├─ finish       → 任务完成
             └─ 纯文本       → 等待用户下一条消息
        → 全局消息话题 /nova/agentos/agent_msg(规划文本/工具调用与结果/完成)
```

- **agent 循环**:无 DAG,每次调用一个工具,模型根据上一步结果决定下一步(闭环)。
- **上下文持久**:对话历史跨任务累积,之前的任务结果可复用。
- **全局消息**:`RunTask` 返回 `task_id`;agent 每轮规划、工具调用与结果、完成总结全部发布到 `/nova/agentos/agent_msg`(`TaskState`,字段:`task_id/status/done/kind/message`,`kind` ∈ status|text|tool_call|tool_result)。

## 模块

| 文件 | 职责 |
| --- | --- |
| `skill_store.py` | 扫描 `skills/<name>/{SKILL.yaml, SKILL.md}`,生成索引、按需加载正文 |
| `agent_loop.py` | 后台持续循环:消费消息队列、调用 LLM、执行工具、维护持久上下文 |
| `mcp_adapter.py` | 与 executor_manager 通信(查询工具 + 发 action goal) |
| `agentos_node.py` | ROS 2 节点:RunTask 入队服务 + agent_msg 消息发布 |
| `agent_cli.py` | 终端聊天 CLI:发消息 + 实时查看 agent 消息 + 调试命令 |

## Skill 说明

Skill 是**任务型领域经验**(纯文本),描述"怎么完成某个任务",而不是"怎么调用某个 API"。它作为 `load_skill` 工具在循环中按需注入 LLM 上下文。

## 运行

```bash
# 一键启动 demo executor + manager + agentos
ros2 launch nova_agentos system.launch.py

# 单独启动 agentos
ros2 run nova_agentos nova_agentos_node --ros-args -p skills_dir:=<skills目录>

# 终端聊天 CLI
ros2 run nova_agentos nova_agentos_cli
```

## 提交任务

```bash
# 命令行直接调服务
ros2 service call /nova/agentos/run nova_interfaces/srv/RunTask "{instruction: '把杯子放到桌上并等待2秒'}"
# 返回 task_id,立即返回

# 监听全部 agent 消息
ros2 topic echo /nova/agentos/agent_msg
```

推荐用 CLI 交互:

```
你> 请把桌面收拾干净
[abc123][agent] 我先加载桌面整理的领域流程...
[abc123][tool] 调用 load_skill: {"skill":"tidy_table"}
[abc123][result] load_skill -> # 收拾桌面的领域经验...
[abc123][tool] 调用 pi0_policy: {"instruction":"..."}
[abc123][result] pi0_policy -> {"ok": true, "infer_steps": 10, ...}
[abc123][agent] 完成,已收拾干净
你> /ping
  你的LLM: 850ms OK
你> /reset     # 重置仿真环境
你> /env       # 查询环境规格(相机/state/action 键)
你> /quit
```

多轮交互:直接再发一条消息即可,上下文会保留上一轮的对话。

## CLI 命令

| 命令 | 行为 |
| --- | --- |
| `/reset` | 重置仿真环境(`/nova/env/reset`) |
| `/ping` | 测每个 LLM provider 连接延迟 |
| `/env` | 查询仿真环境规格(相机/state/action 键) |
| `/help` | 显示帮助 |
| `/quit` `/exit` | 退出 |
