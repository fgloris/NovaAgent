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
        → 状态话题 /nova/agentos/task_state/t_{task_id}(最后 message + done)
```

- **agent 循环**:无 DAG,每次调用一个工具,模型根据上一步结果决定下一步(闭环)。
- **上下文持久**:对话历史跨任务累积,之前的任务结果可复用。
- **每任务状态**:`RunTask` 返回 `task_id`,进度/结果发布到 `/nova/agentos/task_state/t_{task_id}`(字段:status/done/message)。

## 模块

| 文件 | 职责 |
| --- | --- |
| `skill_store.py` | 扫描 `skills/<name>/{SKILL.yaml, SKILL.md}`,生成索引、按需加载正文 |
| `agent_loop.py` | 后台持续循环:消费消息队列、调用 LLM、执行工具、维护持久上下文 |
| `mcp_adapter.py` | 与 executor_manager 通信(查询工具 + 发 action goal) |
| `agentos_node.py` | ROS 2 节点:RunTask 入队服务 + 每任务状态发布 |

## Skill 说明

Skill 是**任务型领域经验**(纯文本),描述"怎么完成某个任务",而不是"怎么调用某个 API"。它作为 `load_skill` 工具在循环中按需注入 LLM 上下文。

## 运行

```bash
# 一键启动 demo executor + manager + agentos
ros2 launch nova_agentos system.launch.py

# 单独启动 agentos
ros2 run nova_agentos nova_agentos_node --ros-args -p skills_dir:=<skills目录>
```

## 提交任务

```bash
ros2 service call /nova/agentos/run nova_interfaces/srv/RunTask "{instruction: '把杯子放到桌上并等待2秒'}"
# 返回 task_id,立即返回

# 监听该任务状态
ros2 topic echo /nova/agentos/task_state/t_<task_id>
# status: working / done / failed;done=true 表示任务结束
```

多轮交互:直接再调一次 `RunTask` 即可,上下文会保留上一轮的对话。
