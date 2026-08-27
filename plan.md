# NovaAgent:统一仿真话题 + 动态 topic 转发 实施计划

> 目标:两个 sim(LIBERO / RoboCasa)的 bridge 统一到规范话题 `/nova/env/*`(只有维度不同,格式一致 + 自发现),并在 AgentOS 内实现动态 topic 转发,让 VLA 类 executor 在执行期间通过专属命名空间收发数据,执行完即关闭。

---

## 0. 现状(关键文件)

| 文件 | 角色 | 关键点 |
|---|---|---|
| `src/nova_libero_bridge/nova_libero_bridge/libero_sim_server.py` | LIBERO TCP 仿真服务 | `normalize_obs`(:67)已归一化为 `video.*`/`state.*`;动作透传;TCP 8766 |
| `src/nova_robocasa_bridge/nova_robocasa_bridge/robocasa_sim_server.py` | RoboCasa TCP 仿真服务 | obs **未归一化**;动作收 dict;TCP 8766 |
| `src/nova_libero_bridge/nova_libero_bridge/libero_bridge_node.py` | LIBERO 桥 | `ACTION_DIM=7`(:23)硬编码;发 `/nova/libero/*` |
| `src/nova_robocasa_bridge/nova_robocasa_bridge/robocasa_bridge_node.py` | RoboCasa 桥 | `ACTION_KEYS`/12 维映射(:22,:32);发 `/nova/robocasa/*`;`video.*` 前缀匹配不到 robosuite 原始键 → 相机/state 实为空 |
| `src/nova_agentos/nova_agentos/agentos_node.py` | AgentOS 主节点 | RunTask → 规划 → `DagExecutor.execute`(:65) |
| `src/nova_agentos/nova_agentos/dag_executor.py` | DAG 执行器 | 逐节点 `adapter.execute`(:19);有 `on_step` 钩子 |
| `src/nova_agentos/nova_agentos/mcp_adapter.py` | executor_manager 客户端 | `fetch_tools` / `execute` |
| `src/nova_interfaces/msg/ToolDescriptor.msg` | 工具描述 | 需加 `obs_bindings` 字段 |
| `src/nova_executor_demo/nova_executor_demo/executor_demo_node.py` | 示例 executor | 需加一个带 bindings 的 VLA 演示工具 |

现状问题:
1. 两个桥话题名、观测格式、动作维度全部不同,且维度硬编码。
2. robocasa 桥的相机/state 发布实际为空(键名不匹配)。
3. `encode_value`/`decode_array`/`decode_observation`/`summarize_value`/`JsonLineClient` 在两 sim server + 两 bridge 中完全重复。

---

## 1. 目标架构

```
┌───────────┐ TCP(JSON) ┌─────────────┐  规范话题 /nova/env/*   ┌──────────────┐
│ sim server│──────────▶│  bridge     │──────────┬────────────▶│  AgentOS     │
│ (libero / │◀──────────│ (统一基类)   │◀─────────┘ (obs/action) │  (规划+调度) │
│  robocasa)│           └─────────────┘                        └──────┬───────┘
└───────────┘                                                         │ MapTopics
                                                 ┌──────────┐        │ service
                                                 │  router  │◀───────┘ (nova_agentos 包内)
                                                 │ 独立节点  │        │ UnmapTopics
                                                 └────┬─────┘
                                      /nova/session/{task_id}/{nid}/*
                                             ▲      │
                                             │      ▼ (action_cmd 回灌 /nova/env/action_cmd)
                                     ┌───────┴──────┐
                                     │  VLA executor│(如 pi0,执行期间订阅 session 话题)
                                     └──────────────┘
```

- **bridge = 纯管道**:sim ↔ ROS 之间的收发、规范动作/观测映射,统一逻辑,子类只填 sim 特定细节。
- **router = 转发**:AgentOS 执行某个 executor 前开启转发,执行完关闭。router 是独立可执行文件(放 `nova_agentos` 包),避免 AgentOS 单线程阻塞导致转发卡死。
- **自发现**:动作 spec(维度+每维含义)与观测 spec(相机/state 键)由 sim server 从 env 自省,随 reset 响应返回,bridge 缓存并暴露为 `/nova/env/info` service;obs 每帧 JSON 自描述(shape/dtype)。

---

## 2. 统一话题协议(规范 `/nova/env/*`)

### 2.1 观测(桥 → agent/executor)

| topic | 类型 | 说明 |
|---|---|---|
| `/nova/env/obs` | `std_msgs/String` | 结构化 JSON 摘要,自描述(见 2.3) |
| `/nova/env/camera/{name}/image_raw` | `sensor_msgs/Image` | 每相机一个,`{name}` 来自 obs_spec |
| `/nova/env/reward` | `std_msgs/Float32` | |
| `/nova/env/success` | `std_msgs/Bool` | |
| `/nova/env/info` | service | 查询动作/观测规格(自发现核心) |

### 2.2 动作(executor/agent → 桥)

| topic | 类型 | 说明 |
|---|---|---|
| `/nova/env/action_cmd` | `std_msgs/Float32MultiArray` | 规范动作向量,长度 = `action_spec.dim` |

动作采用"布局约定 + spec 描述",**不硬编码维度**:

```
[ee_pos(3), ee_rot(3), gripper(N), base(4), control_mode(1)]
LIBERO OSC_POSE     → dim 7,  meaning [dx,dy,dz,droll,dpitch,dyaw,gripper]
RoboCasa PandaOmron → dim 12, meaning [pos3, rot3, gripper, base4, mode]
```

- **动作映射放 sim server 端**(bridge 是纯管道):sim server 收规范向量,自己转 native 动作(libero 直接传数组;robocasa 向量→dict)。
- 换机器人 → 只有 sim server 的 `action_spec` 和映射变化,bridge/agent/executor 不动。

### 2.3 obs JSON 结构(自描述)

```json
{
  "sim": "libero", "robots": ["Panda"], "controller": "OSC_POSE",
  "step_count": 3, "reward": 0.0, "success": false, "done": false,
  "instruction": "open the drawer",
  "action_spec": {"dim": 7, "meaning": ["dx","dy","dz","droll","dpitch","dyaw","gripper"]},
  "state": {
    "robot0_eef_pos": {"shape": [3], "dtype": "float64", "min": -0.2, "max": 0.8},
    "robot0_eef_quat": {"shape": [4], "dtype": "float64", "min": -1.0, "max": 1.0}
  },
  "cameras": {
    "agentview": {"shape": [256, 256, 3], "dtype": "uint8"},
    "robot0_eye_in_hand": {"shape": [256, 256, 3], "dtype": "uint8"}
  }
}
```

### 2.4 观测键名统一规则(公共 `normalize_obs`)

```
3D 且 shape[-1]==3 的数组 → video.{key};若 key 以 "_image" 结尾则先剥掉
其余(含字符串/标量/低维数组) → state.{key}
```
- LIBERO 键 `agentview` → `video.agentview`(不变)
- RoboCasa 键 `agentview_image` → `video.agentview`(剥 `_image`,相机名统一)
- `annotation.human.task_description` → `state.annotation.human.task_description`(保留)
- 指令统一写入 `state.instruction`(libero 已有;robocasa sim server 补上,值来自 `annotation.human.task_description`)

### 2.5 自发现流程

1. sim server `reset` 后从 `env.action_space` 自省 `action_spec`;从归一化 obs 自省 `obs_spec`,随响应返回。
2. bridge 缓存 spec;提供 `/nova/env/info` service 返回完整 spec(`sim/robots/controller/action_spec/obs_spec`);obs 每帧 JSON 内嵌 `action_spec`。
3. AgentOS 每次 RunTask 查询 `/nova/env/info`,获得当前环境动作维度与相机列表。

---

## 3. Phase 1:公共代码抽取(`nova_common`)

### 3.1 新建 `src/nova_common/nova_common/obs_codec.py`(无 ROS 依赖)
- `encode_value(value)` ← 从 `libero_sim_server.py:48` / `robocasa_sim_server.py:19` 抽取
- `decode_array(payload)` / `decode_observation(value)` ← 从 bridge 抽取
- `summarize_value(value)` ← 从 bridge 抽取(>32 元素只给 shape/dtype/min/max)
- `normalize_obs(obs)` ← 从 `libero_sim_server.py:67` 抽取 + 剥 `_image` 后缀规则

### 3.2 新建 `src/nova_common/nova_common/jsonline.py`(无 ROS 依赖)
- `JsonLineClient(host, port, timeout_sec)`: `request(payload)` / `close()` ← 从两 bridge 抽取(`JsonLineClient` 两处完全一致)

### 3.3 新建 `src/nova_common/nova_common/env_bridge.py`(ROS 节点基类,依赖 rclpy)
`class EnvBridgeBase(Node)` 抽两 bridge 公共逻辑:
- 公共参数:`server_host/server_port/camera_width/camera_height/publish_rate_hz/auto_reset/zero_action_on_start/request_timeout_sec`
- 状态:`obs/action_spec/obs_spec/latest_action/last_reward/last_success/last_done/step_count/camera_publishers`
- 公共方法:
  - `_reset_env()`(调 `_build_reset_request()`,解码响应,缓存 spec,`_ensure_camera_publishers`,`_publish_observation`)
  - `_step_once()`(发 step,更新状态,发布)
  - `_publish_observation()`(发布 `/nova/env/obs|reward|success` + 相机图像)
  - `_ensure_camera_publishers()`(从 obs_spec 动态建/复用 `Image` publisher)
  - `action_callback(msg)`(**用 `action_spec["dim"]` 校验长度**,丢弃则 `action_vector_to_native` 返回 None)
  - `info_callback()`(service,返回完整 spec)
  - `reset_callback/step_zero_callback/timer_callback/_numpy_rgb_to_image_msg`
- 子类必须实现:
  - `_build_reset_request()` → dict
  - `action_vector_to_native(values: np.ndarray)` → 发送给 sim server 的 `request["action"]`(libero 返回 list;robocasa 返回 dict)
  - `_extra_info()` → dict(`sim/robots/controller` 等 sim 字段)

### 3.4 依赖与清理
- `nova_common/package.xml` 加 `exec_depend`:`std_msgs`, `sensor_msgs`, `std_srvs`, `nova_interfaces`
- 两 bridge / 两 sim server 删除重复函数,改用 `nova_common`
- 两 bridge 的 `package.xml` 加 `exec_depend`:`nova_common`

---

## 4. Phase 2:sim server 自描述与动作映射

### 4.1 `libero_sim_server.py`
- `_ensure_env` 内基于 `controller` 生成 `action_spec`:
  - `OSC_POSE`/`IK_POSE` → dim 7,meaning `["dx","dy","dz","droll","dpitch","dyaw","gripper"]`
  - `OSC_POSITION` → dim 4,meaning `["dx","dy","dz","gripper"]`
  - dim 优先从 `env.action_space.shape[0]` 自省,fallback 配置
- `reset`/`step` 响应加顶层字段 `action_spec`、`obs_spec`(从归一化 obs 自省:video 键→cameras 键+shape/dtype;state 键→列表)
- 用公共 `normalize_obs`

### 4.2 `robocasa_sim_server.py`
- 用公共 `normalize_obs`(修掉桥侧相机/state 为空的隐患);补 `state.instruction`(取 `annotation.human.task_description`)
- `action_spec`:dim = `env.action_space.shape[0]`(fallback 12),meaning 固定布局 `["pos_x","pos_y","pos_z","rot_x","rot_y","rot_z","gripper","base_vx","base_vy","base_wz","base_rz","control_mode"]`
- **动作转换从 bridge 挪到这里**:`action_vector_to_dict`(`robocasa_bridge_node.py:32` 挪入,收规范向量转 dict,缺省补零、clip [-1,1]);`step` 里 `request["action"]` 改为收向量
- `reset`/`step` 响应同样加 `action_spec`/`obs_spec`

> 两 sim server 可共用 TCP server 代码(`ThreadingTcpServer`/`handle`/`dispatch`),可抽到 `nova_common/jsonline.py` 的 `JsonLineServer` 基类(handler 子类只实现 `dispatch`)。

---

## 5. Phase 3:bridge 统一到 `/nova/env/*`

- `libero_bridge_node.py` / `robocasa_bridge_node.py` 改继承 `EnvBridgeBase`,子类只保留:
  - 节点名、话题命名空间参数、`_build_reset_request`、`action_vector_to_native`、`_extra_info`
- **删除旧话题** `/nova/libero/*`、`/nova/robocasa/*`(直接替换,不保留兼容),改用 `/nova/env/*`
- `random_action_client.py`(两包)改发 `/nova/env/action_cmd`;`ACTION_DIM` 改为从参数 `action_dim` 读(默认 7),不硬编码
- 更新 `config/bridge.yaml`、`launch/*.launch.py`(端口/参数不变;话题已统一)

---

## 6. Phase 4:消息接口扩展

### 6.1 `nova_interfaces/msg/ToolDescriptor.msg` 加字段
```
string obs_bindings    # JSON,如 {"cameras": ["agentview"], "state": ["robot0_eef_pos"]};空串表示纯工具调用、不需要转发
```

### 6.2 新建 `nova_interfaces/srv/MapTopics.srv`
```
string mapping_id
string[] src_topics
string[] dst_topics
string[] msg_types      # 与 src/dst 一一对应:image|string|float32multi|float32|bool
---
bool success
string message
```

### 6.3 新建 `nova_interfaces/srv/UnmapTopics.srv`
```
string mapping_id
---
bool success
string message
```

> `ListTools.srv` 不用改(已返回 `ToolDescriptor[]`)。改完需 `colcon build` 重新编译 `nova_interfaces`。

---

## 7. Phase 5:router + AgentOS 集成

### 7.1 新建 `src/nova_agentos/nova_agentos/topic_router.py`
`class TopicRouter(Node)`:
- 服务 `MapTopics` / `UnmapTopics`(话题名做成参数,默认 `/nova/topic_router/map`、`/nova/topic_router/unmap`)
- 内部 `self._mappings: dict[mapping_id, list[(sub, pub)]]`
- `MapTopics`:动态 `create_subscription(src, msg_type, ...)` + `create_publisher(dst, ...)`;回调里把 msg **复制并重发布**(Image 需重建 msg);同一 mapping_id 重复 map 报错
- msg_type 映射表:`image→sensor_msgs/Image`, `string→std_msgs/String`, `float32multi→std_msgs/Float32MultiArray`, `float32→std_msgs/Float32`, `bool→std_msgs/Bool`
- QoS:图像类转发统一 `best_effort + keep_last(1)`(camera 用;obs/reward/success/action 用 `reliable`)
- `UnmapTopics`:销毁对应 sub/pub
- `main`:用 `MultiThreadedExecutor`(多 mapping 并发安全)
- `setup.py` 加 console_script:`nova_agentos_topic_router = nova_agentos.topic_router:main`

### 7.2 改造 `dag_executor.py`
`DagExecutor.execute` 增加两个钩子,在 `adapter.execute` 前/后调用:
```python
def execute(self, graph, task_id, on_step=None, on_before_node=None, on_after_node=None):
    ...
    if on_before_node: on_before_node(nid, n)
    result = self.adapter.execute(...)
    if on_after_node: on_after_node(nid, n)
```

### 7.3 改造 `agentos_node.py`
- 初始化 router 的 service clients(`MapTopics`/`UnmapTopics`)
- 创建 `/nova/env/info` client;`_run_task_cb` 里查询并缓存 `env_info`(action_spec/obs_spec/cameras)
- `fetch_tools` 后缓存 `tool_name → obs_bindings`(解析 `ToolDescriptor.obs_bindings` JSON)
- `executor.execute(..., on_before_node=..., on_after_node=...)`:

  `on_before_node(nid, node)`:
  1. 取 `bindings = bindings_of(node["tool_name"])`;为空则直接返回
  2. `mapping_id = f"{task_id}_{nid}"`, `ns = f"/nova/session/{mapping_id}"`
  3. 组装 entries(相机名取 bindings 指定∩env_info 实际;bindings 相机为空则全部)并调用 `MapTopics`
  4. 把 `ns` 注入该节点参数:`node_params["topic_namespace"] = ns`(供 executor 读取)
  5. 等待 router 确认(动态 topic 需要 discovery 时间,MapTopics 响应前短暂等待/重试)

  `on_after_node(nid, node)`:调用 `UnmapTopics(mapping_id)`,**必须在异常路径也要执行**(try/finally 或 ensure)

- **动作回路**:router 转发 `/nova/session/{mapping_id}/action_cmd` → `/nova/env/action_cmd`,bridge 订阅它,executor 只往自己命名空间发动作。

### 7.4 `system.launch.py`
加 `nova_agentos_topic_router` 节点。

---

## 8. Phase 6:executor 演示(VLA 语义)

### 8.1 改 `executor_demo_node.py`
- 每个工具 `ToolDescriptor.obs_bindings` 填充:如 `wait/echo` 为空;新增 `pi0_policy` 演示工具:
  - `obs_bindings = {"cameras": ["agentview"], "state": ["robot0_eef_pos"]}`
  - 参数 schema 声明 `topic_namespace`(由 AgentOS 注入)
  - 执行时:订阅 `<ns>/camera/agentview`(Image)+ `<ns>/state`(String),收到一帧后周期发布随机动作到 `<ns>/action_cmd`,运行 `duration_sec` 秒返回;没收到帧则记录并跳过
- 说明文档在计划完成后更新 TODO.md

---

## 9. 验证（不自己跑，通知用户进行）

### 9.1 构建
```
colcon build --symlink-install
```

### 9.2 单桥手动验证(以 libero 为例,robocasa 同理)
```
# 终端1:sim server
python src/nova_libero_bridge/nova_libero_bridge/libero_sim_server.py --port 8766
# 终端2:bridge(先 source install/setup.bash)
ros2 run nova_libero_bridge libero_bridge_node
# 检查
ros2 topic list                          # 应见 /nova/env/obs, /nova/env/reward, /nova/env/success, /nova/env/camera/agentview/image_raw, /nova/env/action_cmd
ros2 topic echo /nova/env/obs --once     # 自描述 JSON(action_spec/state/cameras)
ros2 service call /nova/env/info nova_interfaces/srv/...  # 返回完整 spec
ros2 run nova_libero_bridge random_action_client           # 动作维度用参数指定,仿真推进
```

### 9.3 router 手动验证
```
ros2 run nova_agentos nova_agentos_topic_router
# 开两个终端 echo 源/目的话题,再手动调 MapTopics 验证转发、UnmapTopics 验证关闭
```

### 9.4 端到端(AgentOS)
```
ros2 launch nova_agentos system.launch.py
# 另起 bridge + sim server,再起 executor_demo
ros2 service call /nova/agentos/run nova_interfaces/srv/RunTask "{instruction: '演示:调用 pi0_policy 3 秒'}"
# 执行期间:ros2 topic list 应出现 /nova/session/{id}/* 系列,结束后消失
# 观察 /nova/env/action_cmd 有数据(来自 executor 回灌)
```

---

## 10. 风险点与注意

1. **AgentOS 单线程阻塞**:DAG 执行在 service callback 里阻塞,router 必须是独立节点(不能同 node 回调),已通过独立可执行文件解决。
2. **动态 topic discovery 延迟**:MapTopics 后 executor 需要短暂等待;bridge 端 camera publisher 也是动态创建,executor 订阅需容忍首帧延迟;相机 QoS 用 best_effort 防背压。
3. **异常清理**:UnmapTopics 必须覆盖 execute 抛异常路径(用 try/finally),否则 topic 泄漏。
4. **robocasa 换机器人**:`env.action_space` 自省是唯一来源,meaning 布局若与真实不符需在 sim server 端调整;PandaOmron 布局已在 config 注释说明。
5. **base64 链路性能**:sim server↔bridge 的 JSON+base64 是全链路大头(比 router 高一个量级),本期不动,后续可换二进制协议。
6. **多 VLA 并行**:DAG 允许并行节点,`mapping_id` 用 `task_id:nid` 天然隔离;executor 实现需支持并行实例。
7. **旧话题删除**:`random_action_client` 等客户端全部要同步改到 `/nova/env/*`,否则静默失联。
8. **QoS 匹配**:bridge 相机发布与 router 相机转发的 QoS 需一致(best_effort);obs/reward/success 用 reliable。

---

## 11. 实施顺序清单(供新会话逐步执行)

- [ ] P1: `nova_common/obs_codec.py`(encode/decode/summarize/normalize)
- [ ] P1: `nova_common/jsonline.py`(JsonLineClient;JsonLineServer 基类)
- [ ] P1: `nova_common/env_bridge.py`(EnvBridgeBase)
- [ ] P1: `nova_common/package.xml` 依赖;两 bridge/两 sim server 清理重复代码
- [ ] P2: libero_sim_server:action_spec/obs_spec + 公共 normalize
- [ ] P2: robocasa_sim_server:同上 + normalize + instruction + 动作转换挪入
- [ ] P3: 两 bridge 继承 EnvBridgeBase,发 `/nova/env/*` + `/nova/env/info`;删旧话题
- [ ] P3: random_action_client ×2 改 `/nova/env/action_cmd` + 参数化 dim;更新 config/launch
- [ ] P4: ToolDescriptor 加 obs_bindings;新建 MapTopics/UnmapTopics srv;重建 nova_interfaces
- [ ] P5: `topic_router.py` + console_script
- [ ] P5: `dag_executor.py` 钩子 + `agentos_node.py` 集成 + launch
- [ ] P6: executor_demo 加 `pi0_policy` 演示工具(bindings + session 订阅 + 动作回灌)
- [ ] 验证 9.2/9.3/9.4
