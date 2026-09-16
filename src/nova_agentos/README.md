# nova_agentos

NovaAgent 核心:VLM 驱动的持续 AgentOS。负责 skill 管理、读取当前多路相机画面、agent 循环和工具调度执行。

## 数据流

```
session start/resume → RunTask(session_id, 入队,立即返回 task_id)
        → 后台 agent 循环(上下文跨任务累积)
            每轮:VLM 结合最新多路相机画面发一个函数调用(executor 工具 / load_skill / finish)
             ├─ executor 工具 → executor_manager → 具体 executor
             ├─ load_skill   → 本地注入 SKILL.md 全文
             ├─ finish       → 任务完成
             └─ 纯文本       → 等待用户下一条消息
        → 全局消息话题 /nova/agentos/agent_msg(规划文本/工具调用与结果/完成)
```

- **agent 循环**:无 DAG,每次调用一个工具,模型根据上一步结果决定下一步(闭环)。
- **上下文持久**:对话历史跨任务累积,之前的任务结果可复用。
- **全局消息**:`RunTask` 必须携带 `session_id`;agent 每轮规划、工具调用、反馈与结果、完成总结全部发布到 `/nova/agentos/agent_msg`(`TaskState`,字段:`task_id/status/done/kind/message`,`kind` ∈ status|text|tool_call|tool_feedback|tool_result)。

## 模块

| 文件 | 职责 |
| --- | --- |
| `skill_store.py` | 扫描 `skills/<name>/{SKILL.yaml, SKILL.md}`,生成索引、按需加载正文 |
| `agent_loop.py` | 后台持续循环:消费消息队列、调用 VLM、执行工具、维护持久上下文;装配动态图像段与时间戳 |
| `memory/` | 记忆包:`session.py`(会话/任务记忆 + ContextBuilder)、`images.py`(图像记忆)、`sampler.py`(历史采样) |
| `vision_observer.py` | 订阅 `/nova/env/obs` 与 `/nova/env/camera/*/image_raw`,提供环境摘要与最新帧 |
| `robot_state_observer.py` | 从 `/nova/env/info` 发现机器人状态话题并订阅,每轮注入最新 EEF/关节/夹爪状态(与工具无关) |
| `mcp_adapter.py` | 与 executor_manager 通信(查询工具 + 发 action goal) |
| `agentos_node.py` | ROS 2 节点:RunTask/会话服务 + agent_msg 消息发布 + 首任务自动命名 + 图像记忆装配 |
| `cli/` | 终端 TUI 包(Textual):会话自动创建、对话渲染、命令处理、`--plain` REPL |

### 图像记忆

agentos 是图像记忆的唯一持有者,perception executor 无状态、按 `file://` 引用读文件。

- 落盘根目录 `memory_dir`(默认 `~/.cache/nova_agentos/memory`),每个 session 一个子目录:
  `current/<camera>.jpg`(最新帧)、`history/<epoch_ms>-<camera>.jpg`(采样历史)、`processed/<epoch_ms>-<camera>-<seq>.jpg`(工具返回图)。
- 元信息(相机/时间/来源/底图)写在 JPEG COM 段;模型只看到 `file://<kind>/<file>`。
- `ImageSampler` 每 `image_sample_period_sec` 采样一次,与上一张归一化 MSE 低于阈值则跳过。
- 每轮 context 动态尾部依次注入 current → processed → history 图段,最后追加时间戳(保留稳定前缀以提升缓存命中)。
- 模型可调本地工具 `list_accessible_images(history_depth=N)` 与 `fetch_history_image(time, topic)`。
- 调用 perception 工具时,agentos 自动注入隐藏参数 `image_root` 并把相机名解析为对应 current 图。

## Skill 说明

Skill 是**任务型领域经验**(纯文本),描述"怎么完成某个任务",而不是"怎么调用某个 API"。它作为 `load_skill` 工具在循环中按需注入 LLM 上下文。

## 运行

```bash
# 启动 bridge + raw/VLA/perception executor + manager + agentos。
# RoboCasa Python 3.11 sim server 和 Pi server 是外部前置进程。
ros2 launch nova_agentos system.launch.py

# 单独启动 agentos
ros2 run nova_agentos nova_agentos_node --ros-args -p skills_dir:=<skills目录>

# 终端聊天 TUI(启动即自动创建 session)
ros2 run nova_agentos nova_agentos_cli

# 行式 REPL(无 TUI,适合 ssh/调试)
ros2 run nova_agentos nova_agentos_cli --plain

# 启动时恢复指定 session
ros2 run nova_agentos nova_agentos_cli --resume sess_xxx
```

CLI 启动后自动创建新 session,首个任务由 AgentOS 用 VLM 概括为 `场景-做什么`(如「厨房-收拾桌面」)并改名;退出时标记 session 为 ended(文件保留,可 `--resume` 恢复)。

## 提交任务

```bash
# 命令行直接调服务
ros2 service call /nova/agentos/session/start nova_interfaces/srv/StartSession "{name: 'demo'}"
ros2 service call /nova/agentos/run nova_interfaces/srv/RunTask "{session_id: 'sess_...', instruction: '把杯子放到桌上并等待2秒'}"
# 返回 task_id,立即返回

# 监听全部 agent 消息
ros2 topic echo /nova/agentos/agent_msg
```

推荐用 CLI TUI 交互:

```
◆ NovaAgent CLI 已连接。输入指令开始任务,或 /help 查看命令。
✎ 已自动创建会话 (sess_xxx),首个任务后自动命名
你> 请把桌面收拾干净
◆ 我先加载桌面整理的领域流程...
⚙ 调用 load_skill: {"skill":"tidy_table"}
✔ load_skill -> # 收拾桌面的领域经验...
⚙ 调用 pi0_policy: {"instruction":"..."}
✔ pi0_policy -> {"ok": true, "infer_steps": 10, ...}
◆ 完成,已收拾干净
✎ 会话已命名: 厨房-收拾桌面
```

输入框支持多行(`Shift+Enter` 换行)、鼠标点击定位光标、↑/↓ 浏览历史、`/` 开头给出命令候选。

## CLI 命令

| 命令 | 行为 |
| --- | --- |
| `/session list` | 列出所有 session(名称 + id + 状态) |
| `/session new [name]` | 新建 session |
| `/session resume <id>` | 恢复指定 session |
| `/session rename <name>` | 重命名当前 session |
| `/session info` | 查看当前 session |
| `/session end` | 结束当前 session 并新建 |
| `/reset` | 重置仿真环境(`/nova/env/reset`) |
| `/ping` | 测每个 LLM provider 连接延迟 |
| `/env` | 查询仿真环境规格(相机/state/action 键) |
| `/clear` | 清空会话显示 |
| `/help` | 显示帮助 |
| `/quit` | 退出(结束当前 session) |
