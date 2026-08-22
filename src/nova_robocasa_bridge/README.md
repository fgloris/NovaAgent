# nova_robocasa_bridge

ROS 2 bridge for using RoboCasa as the simulator backend for NovaAgent.

RoboCasa does not expose an external MuJoCo socket / RPC server by default. The
stable integration point is its Gymnasium wrapper:

```py
import gymnasium as gym
import robocasa
import robocasa.wrappers.gym_wrapper

env = gym.make("robocasa/PickPlaceCounterToCabinet", split="target")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action_dict)
```

ROS 2 Humble and RoboCasa should not be imported in the same Python process in
this workspace: ROS 2 uses Python 3.10, while the configured `robocasa` conda
environment uses Python 3.11. The bridge is therefore split into two processes:

```text
ROS 2 process, Python 3.10
  robocasa_bridge_node
  - publishes/subscribes ROS topics
  - talks JSON-over-TCP to the sim server

conda robocasa process, Python 3.11
  robocasa_sim_server
  - imports gymnasium / robocasa
  - owns env.reset() and env.step()
```

The TCP protocol is newline-delimited JSON. Numpy arrays in observations are
encoded as base64 payloads with dtype and shape metadata.

## Topics

- `/nova/robocasa/action_cmd` (`std_msgs/Float32MultiArray`)
  - Up to 12 normalized values in `[-1, 1]`; missing values are padded with 0:
    `[eef_dx, eef_dy, eef_dz, eef_rx, eef_ry, eef_rz, gripper_close, base_x, base_y, base_yaw, torso, control_mode]`
  - `gripper_close` and `control_mode` are thresholded at `0.5` because the
    RoboCasa wrapper maps them to binary controller commands.
- `/nova/robocasa/state` (`std_msgs/String`)
  - JSON payload with `instruction`, scalar rollout status, and compact
    `state.*` observations. Large arrays are summarized by shape/dtype/min/max.
- `/nova/robocasa/reward` (`std_msgs/Float32`)
- `/nova/robocasa/success` (`std_msgs/Bool`)
- `/nova/robocasa/cameras/<name>/image_raw` (`sensor_msgs/Image`)
  - Created dynamically for every `video.*` RGB observation key.

## Services

- `/nova/robocasa/reset` (`std_srvs/Trigger`)
- `/nova/robocasa/step_zero` (`std_srvs/Trigger`)

## Run

Build the ROS package:

```bash
cd /home/ginger/Documents/workspace/CapX/NovaAgent
colcon build --symlink-install
source install/setup.bash
ros2 launch nova_robocasa_bridge robocasa_bridge.launch.py
```

The launch file starts only the ROS bridge. Start the RoboCasa sim server
separately in the `robocasa` conda environment (Python 3.11), passing the
scene config file:

```bash
/home/ginger/miniconda3/envs/robocasa/bin/python \
  /home/ginger/Documents/workspace/CapX/NovaAgent/install/nova_robocasa_bridge/lib/python3.10/site-packages/nova_robocasa_bridge/robocasa_sim_server.py \
  --host 127.0.0.1 --port 8766 \
  --scene-config /home/ginger/Documents/workspace/CapX/NovaAgent/install/nova_robocasa_bridge/share/nova_robocasa_bridge/config/scene.yaml
```

Then launch the ROS bridge in another terminal.

In another terminal, drive the robot manually or with random actions:

```bash
source /opt/ros/humble/setup.bash
source /home/ginger/Documents/workspace/CapX/NovaAgent/install/setup.bash
ros2 run nova_robocasa_bridge teleop_keyboard
# or: ros2 run nova_robocasa_bridge random_action_client
```

## Scene configuration

Robot and scene layout/style are configured in `config/scene.yaml`, which is
read **directly by the sim server** (`--scene-config`) — it is not a ROS
parameter, so it is not forwarded through the bridge:

- `robots`: robosuite robot name (default `PandaOmron`). The RoboCasa gym
  wrapper's action/camera mapping is hardcoded for PandaOmron, so switching
  robots needs care.
- `split`: `target` (fixed 10 test scenes), `pretrain`, `all` (random), or
  `null` to use explicit layout/style ids below.
- `layout_ids` / `style_ids`: explicit ids (int or list). Negative ids mean
  groups: `-1` test group (1-10), `-2` train group (11-60), `-3` all. `style_ids`
  may also be a dict to customize a single fixture's style.
- `layout_and_style_ids`: explicit `(layout, style)` pairs, or `"5x5"` / `"5x1"`.
- `render_quality`: `low` / `medium` / `high` / `ultra`. Enhances the MuJoCo
  offscreen renderer (shadow resolution, reflection, lighting, material gloss).
  `ultra` looks best but renders noticeably slower.

When `split` is `null`, provide at least one of `layout_ids`+`style_ids` or
`layout_and_style_ids`, otherwise RoboCasa falls back to sampling all 60x60
combinations. Do not write empty arrays (`[]`) for `layout_ids` etc.; omit the
key entirely instead.

Do not start the sim server with `ros2 run`. `ros2 run` uses the ROS 2 Python
environment, so it will not see RoboCasa's Python 3.11 packages such as
`gymnasium`.

`robocasa_sim_server` sets `NUMBA_DISABLE_JIT=1` by default before importing
RoboCasa. In the current editable RoboSuite/RoboCasa setup this avoids a numba
cache locator failure during import.

## Agent architecture

Use this bridge as the low-level simulator boundary:

- LLM planner: reads `/nova/robocasa/state`, especially `instruction`, and
  chooses the next high-level skill.
- Skill backend / VLA policy: consumes camera topics plus state JSON and
  publishes `/nova/robocasa/action_cmd`.
- Bridge: steps RoboCasa/MuJoCo at `publish_rate_hz`, publishes reward/success,
  and hides RoboCasa-specific action dict details from the rest of NovaAgent.
