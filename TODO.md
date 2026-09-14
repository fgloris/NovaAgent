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
