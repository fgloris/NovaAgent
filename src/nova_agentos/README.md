# nova_agentos

NovaAgent 核心:LLM 驱动的 AgentOS。负责 skill 管理、任务规划(DAG 生成)、调度执行。

## 数据流

```
用户指令 → planner 两阶段规划 → DAG 校验 → 拓扑序执行 → 工具调用(经 executor_manager)
```

- **阶段1**:注入 skill 索引 + 可用工具 schema,LLM 输出 `{selected_skills, dag 草稿}`。
- **阶段2**:按需加载选中 skill 的 `SKILL.md` 全文注入,LLM 依据领域经验补全每个节点的参数。
- **DAG**:每个节点是一次 executor 工具调用(`type=tool`),`depends_on` 表达依赖,参数支持 `"$ref": "<节点id>"` 引用前序节点结果。

## 模块

| 文件 | 职责 |
| --- | --- |
| `skill_store.py` | 扫描 `skills/<name>/{SKILL.yaml, SKILL.md}`,生成索引、按需加载正文 |
| `planner.py` | 两阶段 LLM 规划,function calling 产出 DAG |
| `dag_validator.py` | 校验: id 唯一 / type 合法 / 依赖存在 / 工具已注册 / 无环 |
| `dag_executor.py` | 拓扑序逐节点调用工具,支持 `$ref` 引用解析 |
| `mcp_adapter.py` | 与 executor_manager 通信(查询工具 + 发 action goal) |
| `agentos_node.py` | ROS 2 节点:RunTask 服务 + TaskState 状态发布 |

## Skill 说明

Skill 是**任务型领域经验**(纯文本),描述"怎么完成某个任务",而不是"怎么调用某个 API"。它只在规划时注入 LLM 上下文,不进入执行图。

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
```

监听状态:

```bash
ros2 topic echo /nova/agentos/task_state
```
