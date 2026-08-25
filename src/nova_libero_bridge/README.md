# nova_libero_bridge

LIBERO ROS 2 桥接器，用于将 LIBERO 作为 NovaAgent 的仿真后端。

LIBERO 基于 robosuite/BDDL，官方通过 `OffScreenRenderEnv` 提供离屏渲染接口：

```py
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

bench = benchmark.get_benchmark_dict()["libero_spatial"]()
bddl_file = bench.get_task_bddl_file_path(0)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
obs, reward, done, info = env.step([0.0] * 7)  # 旧 gym API,动作 7 维 OSC_POSE
```

ROS 2 Humble（Python 3.10）与配置的 `libero` conda 环境（Python 3.8）不在同一个 Python 进程中，因此桥接器拆分为两个进程，与 `nova_robocasa_bridge` 架构一致：

```text
ROS 2 进程, Python 3.10
  libero_bridge_node
  - 发布/订阅 ROS topic
  - 通过 TCP 与 sim server 进行 JSON 通信

conda libero 进程, Python 3.8
  libero_sim_server
  - 导入 libero / robosuite
  - 负责 env.reset() 和 env.step()
```

TCP 协议为按行分隔的 JSON。观测中的 numpy 数组以 base64 编码，并附带 dtype 和 shape 元数据。观测键统一为 robocasa 风格：图像加 `video.` 前缀，其余加 `state.` 前缀。

## Topics

- `/nova/libero/action_cmd`（`std_msgs/Float32MultiArray`）
  - 7 个值：`[eef_dx, eef_dy, eef_dz, eef_droll, eef_dpitch, eef_dyaw, gripper]`（OSC_POSE）
- `/nova/libero/state`（`std_msgs/String`）
  - JSON 载荷，包含 `instruction`、标量的 rollout 状态以及精简后的 `state.*` 观测。大型数组以 shape/dtype/min/max 概括。
- `/nova/libero/reward`（`std_msgs/Float32`）
- `/nova/libero/success`（`std_msgs/Bool`）
  - LIBERO 没有内置 success 标志，来自 `env.check_success()`。
- `/nova/libero/cameras/<name>/image_raw`（`sensor_msgs/Image`）
  - 为每一个 `video.*` RGB 观测键动态创建（默认 `agentview`、`robot0_eye_in_hand`）。

## Services

- `/nova/libero/reset`（`std_srvs/Trigger`）
- `/nova/libero/step_zero`（`std_srvs/Trigger`）

## 运行

构建 ROS 包：

```bash
cd /home/ginger/Documents/workspace/CapX/NovaAgent
colcon build --symlink-install
source install/setup.bash
ros2 launch nova_libero_bridge libero_bridge.launch.py
```

启动文件只启动 ROS 桥接器。需要在 `libero` conda 环境（Python 3.8）中单独启动 LIBERO sim server。它会自动定位 LIBERO 源码根目录（`LIBERO_ROOT` 环境变量，或 `~/.libero/config.yaml` 的 `benchmark_root` 向上推断），因此无需额外参数：

```bash
/home/ginger/miniconda3/envs/libero/bin/python \
  /home/ginger/Documents/workspace/CapX/NovaAgent/install/nova_libero_bridge/lib/python3.10/site-packages/nova_libero_bridge/libero_sim_server.py \
  --host 127.0.0.1 --port 8767
```

然后在另一个终端启动 ROS 桥接器。可用 launch 参数覆盖任务与相机：

```bash
ros2 launch nova_libero_bridge libero_bridge.launch.py benchmark:=libero_object task_id:=3
```

在另一个终端中，可以手动或使用随机动作驱动机器人：

```bash
source /opt/ros/humble/setup.bash
source /home/ginger/Documents/workspace/CapX/NovaAgent/install/setup.bash
ros2 run nova_libero_bridge random_action_client
```

## 场景配置

机器人与控制器在 `config/scene.yaml` 中配置，该文件**由 sim server 直接读取**（`--scene-config`）——它不是 ROS 参数：

- `robots`：robosuite 机器人名（默认为 `Panda`）。
- `controller`：`OSC_POSE` / `OSC_POSITION` / `IK_POSE` 等（默认为 `OSC_POSE`，7 维动作）。
- `renderer`：渲染后端（默认为 `mujoco`）。
- `camera_names`：相机名列表，为每个相机创建 `/nova/libero/cameras/<name>/image_raw`。

不要用 `ros2 run` 启动 sim server。`ros2 run` 使用 ROS 2 的 Python 环境，因此无法找到 LIBERO 的 Python 3.8 包（如 `robosuite`）。

## Agent 架构

将该桥接器用作底层仿真边界：
- LLM 规划器：读取 `/nova/libero/state`，尤其是 `instruction`，并选择下一个高层技能。
- 技能后端 / VLA 策略：消费相机 topic 和 state JSON，并发布 `/nova/libero/action_cmd`。
- 桥接器：以 `publish_rate_hz` 步进 LIBERO/MuJoCo，发布 reward/success，并对 NovaAgent 其余部分隐藏 LIBERO 特有的 7 维 OSC_POSE 动作细节。
