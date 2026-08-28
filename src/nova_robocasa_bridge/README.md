# nova_robocasa_bridge

RoboCasa ROS 2 桥接器，用于将 RoboCasa 作为 NovaAgent 的仿真后端。

RoboCasa 默认不暴露外部的 MuJoCo socket / RPC 服务。稳定的集成入口是其 Gymnasium 包装器：

```py
import gymnasium as gym
import robocasa
import robocasa.wrappers.gym_wrapper

env = gym.make("robocasa/PickPlaceCounterToCabinet", split="target")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action_dict)
```

在本工作区中，ROS 2 Humble 和 RoboCasa 不应在同一个 Python 进程中被导入：ROS 2 使用 Python 3.10，而配置的 `robocasa` conda 环境使用 Python 3.11。因此桥接器被拆分为两个进程：

```text
ROS 2 进程, Python 3.10
  robocasa_bridge_node
  - 发布/订阅 ROS topic
  - 通过 TCP 与 sim server 进行 JSON 通信

conda robocasa 进程, Python 3.11
  robocasa_sim_server
  - 导入 gymnasium / robocasa
  - 负责 env.reset() 和 env.step()
```

TCP 协议为按行分隔的 JSON。观测中的 numpy 数组以 base64 编码，并附带 dtype 和 shape 元数据。

## Topics

统一规范话题 `/nova/env/*`(与 LIBERO 一致,只维度不同;动作/观测规格通过自发现获取):

- `/nova/env/action_cmd`（`std_msgs/Float32MultiArray`）
  - 长度 = `action_spec.dim`(默认 12,`[-1, 1]` 归一化向量;缺失值以 0 填充):
    `[pos3, rot3, gripper, base4, control_mode]`
  - 动作转换在 sim server 端完成(规范向量 → RoboCasa action dict)。
- `/nova/env/obs`（`std_msgs/String`）
  - 自描述 JSON：含 `sim/robots/controller`、`instruction`、`action_spec`、精简后的 `state` 与 `cameras`。大型数组以 shape/dtype/min/max 概括。
- `/nova/env/reward`（`std_msgs/Float32`）
- `/nova/env/success`（`std_msgs/Bool`）
- `/nova/env/camera/<name>/image_raw`（`sensor_msgs/Image`）
  - 为每一个 `video.*` RGB 观测键动态创建(键名统一剥掉 `_image` 后缀)。

## Services

- `/nova/env/info`（`nova_interfaces/srv/EnvInfo`）
  - 自发现：返回 `action_spec`（维度+含义）与 `obs_spec`（相机/state 键），供 AgentOS 规划转发。
- `/nova/env/reset`（`std_srvs/Trigger`）
- `/nova/env/step_zero`（`std_srvs/Trigger`）

## 运行

构建 ROS 包：

```bash
cd /home/ginger/Documents/workspace/CapX/NovaAgent
colcon build --symlink-install
source install/setup.bash
ros2 launch nova_robocasa_bridge robocasa_bridge.launch.py
```

启动文件只启动 ROS 桥接器。需要在 `robocasa` conda 环境（Python 3.11）中单独启动 RoboCasa sim server。它会通过相对路径自动定位项目的 `config/scene.yaml`，因此无需 `--scene-config` 参数（可通过 `--scene-config <path>` 或 `NOVA_SCENE_CONFIG` 环境变量覆盖）：

```bash
/home/ginger/miniconda3/envs/robocasa/bin/python \
  /home/ginger/Documents/workspace/CapX/NovaAgent/install/nova_robocasa_bridge/lib/python3.10/site-packages/nova_robocasa_bridge/robocasa_sim_server.py \
  --host 127.0.0.1 --port 8766
```

然后在另一个终端启动 ROS 桥接器。

在另一个终端中，可以手动或使用随机动作驱动机器人：

```bash
source /opt/ros/humble/setup.bash
source /home/ginger/Documents/workspace/CapX/NovaAgent/install/setup.bash
ros2 run nova_robocasa_bridge random_action_client
```

## 场景配置

机器人与场景布局/风格在 `config/scene.yaml` 中配置，该文件**由 sim server 直接读取**（`--scene-config`）——它不是 ROS 参数，因此不会通过桥接器转发：

- `robots`：robosuite 机器人名称（默认为 `PandaOmron`）。RoboCasa gym 包装器的动作/相机映射是为 PandaOmron 硬编码的，因此更换机器人需要谨慎。
- `split`：`target`（固定的 10 个测试场景）、`pretrain`、`all`（随机），或 `null` 以使用下方的显式 layout/style id。
- `layout_ids` / `style_ids`：显式 id（int 或 list）。负数 id 表示组：`-1` 测试组（1-10）、`-2` 训练组（11-60）、`-3` 全部。`style_ids` 也可以是 dict，用于自定义单个 fixture 的风格。
- `layout_and_style_ids`：显式的 `(layout, style)` 组合，或 `"5x5"` / `"5x1"`。
- `render_quality`：`low` / `medium` / `high` / `ultra`。增强 MuJoCo 离屏渲染器（阴影分辨率、反射、光照、材质光泽）。`ultra` 效果最好但渲染更慢。

当 `split` 为 `null` 时，至少提供 `layout_ids`+`style_ids` 或 `layout_and_style_ids` 之一，否则 RoboCasa 会回退到对所有 60x60 组合进行采样。不要为 `layout_ids` 等写入空数组（`[]`）；应直接省略该键。

不要用 `ros2 run` 启动 sim server。`ros2 run` 使用 ROS 2 的 Python 环境，因此无法找到 RoboCasa 的 Python 3.11 包（如 `gymnasium`）。

`robocasa_sim_server` 在导入 RoboCasa 前默认设置 `NUMBA_DISABLE_JIT=1`。在当前可编辑安装的 RoboSuite/RoboCasa 环境中，这可以避免导入时出现 numba 缓存定位器失败。

## Agent 架构

将该桥接器用作底层仿真边界：
- LLM 规划器：读取 `/nova/env/obs`，尤其是 `instruction`，并选择下一个高层技能。
- VLA 类 executor：启动时静态绑定 `/nova/env/*`，常驻订阅相机 topic 和 state JSON，并把动作发布到 `/nova/env/action_cmd`（回灌 bridge，无动态路由）。
- 桥接器：以 `publish_rate_hz` 步进 RoboCasa/MuJoCo，发布 reward/success，并对 NovaAgent 其余部分隐藏 RoboCasa 特有的动作维度（默认 12 维）细节。
