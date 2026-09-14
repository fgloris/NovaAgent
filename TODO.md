# 8.24
![alt text](<Screenshot from 2026-08-24 20-11-46.png>)
opencode -s ses_fcc87a673ffes3DTdEUZaCf9Fn
未完成：
1. 未完成仿真执行闭环。skill里面都是假的，nova_executor也是假的。
2. manager现在没有verify action是否成功，对任务没有持续追踪。
3. 还没有确认各个话题是否合理。

计划：
1. 先用weijia h200服务器跑一下VLA backend,看看连过来到底能不能成。
2. 看看能不能把LIBERO的仿真连上。(已经完成环境libero的搭建)
3. 试一下GraspNet。

# 8.25 统一仿真话题 + 动态 topic 转发(plan.md 已执行,代码完成,待验证)
- [x] P1: `nova_common` 抽取 `obs_codec`(encode/decode/summarize/normalize/build_obs_spec)、`jsonline`(JsonLineClient/JsonLineServer)、`env_bridge`(EnvBridgeBase)
- [x] P2: 两 sim server 自描述(reset/step 响应带 `action_spec`/`obs_spec`/`sim_info`);robocasa 键名归一化 + `state.instruction` + 动作转换挪入 sim server
- [x] P3: 两 bridge 继承 `EnvBridgeBase`,统一发布 `/nova/env/*`;`random_action_client` 改发 `/nova/env/action_cmd` 且 `action_dim` 参数化
- [x] P4: `ToolDescriptor` 加 `obs_bindings`;新增 `MapTopics`/`UnmapTopics`/`EnvInfo` srv(需 colcon build 重新编译 nova_interfaces)
- [x] P5: `topic_router`(MapTopics/UnmapTopics 服务,独立节点) + `dag_executor` 前后钩子 + `agentos` 动态转发/命名空间注入/异常清理 + `system.launch.py`
- [x] P6: executor_demo 新增 `pi0_policy` 演示工具(bindings + session 订阅 + 动作回灌)
- [ ] 验证:先 `colcon build --symlink-install`,再按 plan.md §9 做 9.2(单桥)/9.3(router)/9.4(端到端)手动验证

![alt text](<Screenshot from 2026-08-25 18-21-30.png>)

打通闭环后做完整验证，再做其他工作。

# 8.27

## 闭环测试(在服务器NovaAgent/执行)

### 一、运行robocasa仿真
terminal 1:
```
. ../robocasa/.venv/bin/activate && python3 src/nova_robocasa_bridge/nova_robocasa_bridge/robocasa_sim_server.py
```
等显示
```
...
[render] quality=low shadowsize=1024 offsamples=0 nlight=1 ambient=0.40 diffuse=0.60 specular=0.20 shininess=1.00
...
warmup done in 34.9s
RoboCasa sim server listening on 127.0.0.1:8766
```
后，开terminal 2:

```bash
. install/setup.sh && ros2 run nova_robocasa_bridge robocasa_bridge_node
```

### 二、运行vla backend
```
. ../env.sh && . ../openpi/.venv/bin/activate && python src/nova_vla_executor/nova_vla_executor/pi_server.py
```
等显示
```
[pi0] loaded checkpoint: /home/ubuntu/data1/lxy/robocasa/robocasa365_checkpoints/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000 (model=pi0_robocasa_pretrain_human300)
[pi0] serving on ws://0.0.0.0:8767/predict
INFO:     Started server process [1549580]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8767 (Press CTRL+C to quit)
```
即可。

### 三、连接foxglove
如何不用sudo安装foxglove?
用 apt install 下载二进制包，然后用 dpkg-deb -x 解压缩到外部文件夹，再补上环境变量。
现在假设已经装好了foxglove。
在本地运行：
```
ssh -f -N -L 8765:localhost:8765 MaA6000
```
在远端运行：
```
. ../env.sh
. install/setup.sh
ros2 run foxglove_bridge foxglove_bridge
```

### 四、启动AgentOS
```
. ../env.sh && . install/setup.sh && ros2 launch nova_agentos system.launch.py
```

### 五、各类命令
```
ros2 service call /nova/agentos/session/start nova_interfaces/srv/StartSession "{name: 'demo'}"
ros2 service call /nova/agentos/run nova_interfaces/srv/RunTask "{session_id: 'sess_...', instruction: '把桌面收拾干净'}"
```
reset仿真环境：
```
ros2 service call /nova/env/reset std_srvs/srv/Trigger "{}"
```

# 8.28
1. 去掉过度设计部分。动态router，统一改成静态remapping，DAG也去掉。（已完成）
2. 需要找一个适合用于判断任务是否成功的模型（verifier）

# 8.31
1. 首先，pi0和pi0.5对多元化的指令follow很差，几乎只能抓取。我需要看一眼，找出他们的共同缺点，给出我的动机。
   现在我的路线当以VLM+preception tools+math tools+传统motion tools为主
2. 下一步是测试preception tools

# 9.3
本地测试流程

### 仿真
```bash
conda activate robocasa
python3 src/nova_robocasa_bridge/nova_robocasa_bridge/robocasa_sim_server.py
```

```bash
. install/setup.sh && ros2 launch nova_robocasa_bridge robocasa_bridge.launch.py 
```

### foxglove
```bash
. install/setup.sh && ros2 run foxglove_bridge foxglove_bridge 
```

### preception tools
```bash
. install/setup.sh && ros2 run nova_preception_executor nova_preception_executor_node
```

### AgentOS
```bash
. install/setup.sh
```
快速设计一个编排！

# 9.14
长期记忆 Long-Term Memory，需要持久化成文件。
  ├── 用户记忆       用户偏好、习惯、约束
  ├── 场景记忆       物体、位置、空间关系、环境状态
  └── 经验和技能记忆（模型检索读取）  哪种任务应该如何执行，哪些方法在什么条件下成功/失败。目前的存法是放在skills里面，经验相关还没加。

短期记忆 Context Memory，持久化成临时文件。
  ├── 会话记忆       会话开始以来所有与用户对话、执行任务的记忆，当用户结束会话时生成一个会话ID并存成临时文件，以供用户恢复。
  └── 任务记忆       当前用户发出的单个任务内的记忆

短期记忆怎么维护？
1. 每当一个新task入队，开启一个新任务记忆。该记忆过程中，所有的thinking和tool use反馈等应当保留。tool call内部记忆省略。
2. 会话记忆表现为新任务记忆的列表。每次都需要给vlm发完整任务记忆。当上下文装不下，就需要从前面的任务记忆开始压缩总结，压缩为用户指令-执行方法-结果。
3. 帮我加一个log（如果原来就有，把他做完），记录每次发给模型的context和模型的response。（即记录每次API调用完整内容。你可以在API Client那里加一层log）
4. 长期记忆先不用写。

```
  CLI / ROS client
        |
        | StartSession / ResumeSession
        v
  session_id
        |
        | RunTask(session_id, instruction)
        v
  AgentosNode
        |
        | 创建 TaskMemory
        | 放入全局 FIFO 队列
        v
  AgentLoop
        |
        | 一次取出一个 task
        v
  TaskRunner
        |
        | 构造上下文
        | 调用 VLM
        | 执行工具
        | 接收反馈
        | 回填结果
        | 再次调用 VLM
        v
  finish / error / max steps
        |
        v
  任务落盘 + TaskState 发布
```

1. Session 是任务的外层容器

  相关代码在：

  - src/nova_agentos/nova_agentos/agentos_node.py:92
  - src/nova_agentos/nova_agentos/memory.py:107

  AgentOS 提供三个 session 服务：

  /nova/agentos/session/start
  /nova/agentos/session/resume
  /nova/agentos/session/end

  start 会生成类似：

  sess_a1b2c3d4e5f6

  并创建：

  ~/.cache/nova_agentos/sessions/<session_id>/
  ├── session.json
  ├── context.json
  └── tasks/

  RunTask 必须携带 session：

  string session_id
  string instruction

  在 src/nova_agentos/nova_agentos/agentos_node.py:101：

  task = self.sessions.create_task(
      request.session_id,
      request.instruction,
  )

  这里会做两件事：

  1. 检查 session 是否存在并且是 active
  2. 创建一个新的 TaskMemory

  如果 session 不存在、已经结束或者 session_id 为空，任务不会入队。

  ———

  2. 所有任务进入同一个全局 FIFO 队列

  相关代码在：

  - src/nova_agentos/nova_agentos/agent_loop.py:224
  - src/nova_agentos/nova_agentos/agent_loop.py:263

  AgentLoop 内部只有一个队列：

  self.queue: queue.Queue = queue.Queue()

  提交任务时：

  self.loop.submit(
      task_id,
      request.session_id,
      request.instruction,
  )

  然后后台线程按照 FIFO 顺序执行：

  while self._running:
      item = self.queue.get()
      ...
      TaskRunner(...).run()

  因此当前行为是：

  session A 的 task 1
  session B 的 task 1
  session A 的 task 2

  仍然会全局串行执行，不会并发控制机器人。

  这样可以避免两个任务同时调用机器人动作，产生动作竞争。

  注意：session 是隔离的，但执行队列不是隔离的。

  ———

  3. TaskRunner 是单任务状态机

  相关代码在：

  src/nova_agentos/nova_agentos/agent_loop.py:50

  一个任务的执行主循环是：

  for round_no in range(1, MAX_STEPS_PER_TASK + 1):

  默认最多执行 20 轮：

  MAX_STEPS_PER_TASK = 20

  每一轮大致如下：

  1. 读取 session 历史任务
  2. 获取当前 task 事件
  3. 获取最新观测
  4. 获取当前工具 schema
  5. 构造 VLM messages
  6. 调用 VLM
  7. 如果返回 tool call，执行工具
  8. 记录工具结果
  9. 下一轮重新调用 VLM

  模型不是一次性生成完整计划，而是每次只决定下一步。

  例如用户输入：

  把杯子放到桌子上

  模型可能产生：

  第 1 轮: 调用 grasp_object
  第 2 轮: 调用 move_to
  第 3 轮: 调用 release_object
  第 4 轮: 调用 finish

  每一步工具执行完后，结果都会反馈给模型，由模型决定下一步。

  ———

  4. VLM 可调用三类内部工具

  定义在：

  src/nova_agentos/nova_agentos/agent_loop.py:17

  当前固定加入两个 AgentOS 内置工具：

  tools = [
      LOAD_SKILL_TOOL,
      FINISH_TOOL,
  ] + to_llm_tools(descriptors)

  分别是：

  ### load_skill

  模型可以请求加载技能文件：

  {
    "skill": "tidy_table"
  }

  执行后读取：

  skills/tidy_table/SKILL.md

  内容会作为工具结果回填到当前任务上下文。

  ### finish

  模型完成任务时调用：

  {
    "summary": "已经把杯子放到桌子上"
  }

  Agent 收到 finish 后：

  self.task.finish("success", summary)

  然后写入任务文件并发布完成消息。

  ### Executor 工具

  这些工具来自 executor manager：

  descriptors = self.adapter.fetch_tools()

  例如：

  grasp
  move
  release
  pi0_policy

  它们通过 ToolDescriptor 动态发现，再转换成 OpenAI function schema。

  ———

  5. 工具执行链路

  相关代码：

  - src/nova_agentos/nova_agentos/mcp_adapter.py:33
  - src/nova_executor_manager/nova_executor_manager/executor_manager_node.py:90

  完整调用链是：

  TaskRunner
    |
    | McpAdapter.execute()
    v
  /nova/executor_manager/execute
    |
    | MCPExecute action
    v
  ExecutorManagerNode
    |
    | 根据 tool_name 查 registry
    v
  具体 executor action server

  AgentOS 发出的 Action goal 包含：

  tool_name
  params_json
  trace_id

  例如：

  {
    "tool_name": "grasp",
    "params_json": "{\"object\":\"cup\"}",
    "trace_id": "task_xxx"
  }

  Executor Manager 根据心跳注册表找到真正的 executor，然后转发 Action goal。

  ———

  6. Action feedback 会实时进入 Agent

  McpAdapter.execute() 接收回调：

  feedback_callback=None

  调用 Action 时注册：

  send_goal_async(
      goal,
      feedback_callback=...
  )

  收到反馈后，会进入 TaskRunner：

  def feedback(status, message):
      self.task.add_event(
          "tool_feedback",
          tool_name=name,
          status=status,
          message=message,
      )

  例如 executor 发送：

  status = running
  message = 正在移动到桌面

  会产生一个任务事件：

  {
    "type": "tool_feedback",
    "tool_name": "move",
    "status": "running",
    "message": "正在移动到桌面"
  }

  同时发布 ROS 消息：

  TaskState.kind = tool_feedback

  所以 UI 或 CLI 可以实时看到：

  [task_xxx][feedback] move: 正在移动到桌面

  ———

  7. 工具结果和失败处理

  工具执行完成后，结果会统一转换成字符串：

  content = self._run_tool(name, args)

  成功示例：

  {
    "success": true,
    "summary": "抓取完成"
  }

  失败示例：

  工具执行失败: grasp action server 不可用

  无论成功还是失败，都会先写入任务事件：

  self.task.add_event(
      "tool_result",
      tool_name=name,
      success=not failed,
      summary=content[:2000],
  )

  然后发布：

  TaskState.kind = tool_result

  连续失败达到三次时，任务终止：

  MAX_TOOL_FAILS = 3

  执行顺序是：

  tool_feedback
  tool_result
  failed task

  也就是说，工具失败不会直接跳过 tool_result。

  ———

  8. 当前任务记忆和 session 历史记忆

  任务记忆结构大致是：

  {
    "task_id": "task_xxx",
    "session_id": "sess_xxx",
    "instruction": "把杯子放到桌上",
    "events": [
      {
        "type": "assistant_text",
        "content": "我先抓取杯子",
        "time": "..."
      },
      {
        "type": "tool_call",
        "tool_name": "grasp",
        "params": {},
        "time": "..."
      },
      {
        "type": "tool_feedback",
        "tool_name": "grasp",
        "status": "running",
        "message": "正在抓取",
        "time": "..."
      },
      {
        "type": "tool_result",
        "tool_name": "grasp",
        "success": true,
        "summary": "抓取成功",
        "time": "..."
      }
    ],
    "outcome": "success",
    "summary": "任务完成",
    "duration_ms": 3000
  }

  代码在：

  src/nova_agentos/nova_agentos/memory.py:37

  事件每发生一次就落盘：

  self.manager.save_task(self.task)

  因此进程中断时，至少可以保留已经发生的任务事件。

  任务结束后，session 的 context.json 会更新为任务列表：

  session.context["_tasks"] = [
      item.to_dict()
      for item in self.manager.tasks(self.task.session_id)
  ]

  ———

  9. 每轮 VLM 收到什么上下文

  上下文构造在：

  src/nova_agentos/nova_agentos/memory.py:271

  当前消息结构大致是：

  system prompt
  session 历史任务
  skill 索引
  当前用户指令
  当前任务事件
  最新环境观测
  工具 schema
  当前任务的运行时 assistant/tool 协议消息

  代码中：

  messages = [
      {
          "role": "system",
          "content": session.context.get("system_prompt", ""),
      },
      {
          "role": "user",
          "content": (
              f"# session 历史任务\n..."
              f"# skill 索引\n..."
              f"# 当前用户指令\n..."
              f"# 当前任务事件\n..."
          ),
      },
  ]

  然后追加视觉观测：

  messages.append(observation)

  最后追加工具 schema：

  messages.append({
      "role": "user",
      "content": f"# 可用工具 schema\n..."
  })

  当前相机图片不是历史记忆的一部分，而是每轮重新从 VisionObserver 获取。

  ———

  10. 历史任务会被压缩

  压缩逻辑在：

  src/nova_agentos/nova_agentos/memory.py:228

  参数是：

  context_budget_tokens: 12000
  context_compaction_enabled: true
  max_recent_tasks: 8

  如果超出预算：

  较早任务:
    用户指令 - 执行方法 - 结果

  最近任务:
    保留较详细事件

  当前任务:
    始终保留

  历史任务的摘要类似：

  用户指令: 把杯子放到桌上
  执行方法: grasp, move, release
  结果: success - 已完成

  原始 task JSON 不会删除，压缩只影响 context.json 中用于发送给 VLM 的上下文。

  ———

  11. 视觉观测如何进入模型

  代码在：

  src/nova_agentos/nova_agentos/vision_observer.py:63

  VisionObserver 订阅：

  /nova/env/obs
  /nova/env/camera/<camera_name>/image_raw

  每轮调用：

  self.observation_provider()

  得到类似：

  {
      "role": "user",
      "content": [
          {
              "type": "text",
              "text": "# 当前环境视觉观测 ..."
          },
          {
              "type": "text",
              "text": "camera: robot0_agentview_left ..."
          },
          {
              "type": "image_url",
              "image_url": {
                  "url": "data:image/jpeg;base64,..."
              }
          }
      ]
  }

  相机帧会：

  1. 转成 RGB
  2. 缩放到 vlm_max_image_size
  3. 压缩为 JPEG
  4. 作为多模态消息发送给 VLM

  历史任务文件只保存任务事件，不把原始图片塞进 session 上下文。

  ———

  12. 每次 VLM 请求都会记 API 日志

  日志接入点在：

  src/nova_common/nova_common/llm_client.py:94

  每个 provider 实际请求都会生成新的：

  request_id = req_xxx

  日志目录：

  ~/.local/share/nova_agentos/api_logs/YYYY-MM-DD/

  每个请求至少包含两行：

  {
    "type": "request",
    "request_id": "...",
    "task_id": "...",
    "session_id": "...",
    "provider": "...",
    "model": "...",
    "messages": "...",
    "tools": "..."
  }

  以及：

  {
    "type": "response",
    "request_id": "...",
    "duration_ms": 1234,
    "response": "...",
    "tool_calls": "..."
  }

  如果第一个 provider 失败并 fallback 到第二个 provider，每个实际 HTTP 请求都会单独记录。

  图片不会把 base64 重复写入 JSONL，而是保存到：

  ~/.local/share/nova_agentos/api_logs/images/<request_id>/

  并记录：

  camera
  timestamp
  width
  height
  sha256
  path

  ———

  13. ROS 状态消息如何发布

  消息定义在：

  src/nova_interfaces/msg/TaskState.msg

  当前支持：

  status
  text
  tool_call
  tool_feedback
  tool_result

  TaskRunner 通过回调通知节点：

  self.on_state(
      task_id,
      session_id,
      status,
      message,
      done,
      kind,
  )

  节点收到后发布到：

  /nova/agentos/agent_msg

  例如一次任务可能发布：

  status        收到指令
  text          我先观察桌面
  tool_call     调用 grasp: ...
  tool_feedback grasp: 正在执行
  tool_result   grasp -> 成功
  tool_call     调用 move: ...
  tool_result   move -> 成功
  status        任务完成

  done=true 只在最终成功或失败时出现。

  ———

  14. CLI 的编排方式

  CLI 不会自动创建 session。

  启动后必须执行：

  /session start demo

  之后普通输入会自动变成：

  RunTask.Request(
      session_id=self._session_id,
      instruction=instruction,
  )

  代码在：

  src/nova_agentos/nova_agentos/agent_cli.py:86

  支持：

  /session start [name]
  /session resume <session_id>
  /session end
  /session info

  普通输入的实际路径是：

  用户输入
    -> AgentCliNode.send_message()
    -> RunTask(session_id, instruction)
    -> AgentOS 入队
    -> TaskState 订阅回显

  ———

  当前实现的两个边界

  1. Console 还没有完整接入 AgentOS session 生命周期

  nova_console 已经改为发送 session_id，但它自身没有自动调用 StartSession。因此当前 CLI 是完整支持 session 的客
  户端，Console 仍需要外部注入有效 session_id，否则聊天请求会被拒绝。

  2. 运行中的 tool-call 协议没有跨 AgentOS 重启恢复

  任务事件会持久化，但 TaskRunner.runtime_messages 是内存变量：

  self.runtime_messages: list[dict] = []

  所以 AgentOS 重启后可以恢复任务摘要、工具结果等可读记忆，但不会恢复完整的 OpenAI assistant/tool message 协议
  链。当前设计依赖新任务从持久化事件重新构造上下文，而不是恢复中断中的同一个 VLM 调用链。

  简化来说，现在的 Agent 是：

  session 管理长期运行范围
  task 表示一次用户任务
  AgentLoop 保证所有任务全局串行
  TaskRunner 执行 VLM/tool/VLM 闭环
  Working memory 持久化当前任务事件
  ContextBuilder 每轮重建 VLM 上下文
  McpAdapter 负责 ROS Action 工具调用
  TaskState 负责实时状态输出
  ApiLogger 负责完整模型调用审计