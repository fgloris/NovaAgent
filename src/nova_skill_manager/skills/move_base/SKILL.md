---
name: move_base
description: |
  控制 PandaOmron 底盘朝指定方向 (vx, vy, wz) 持续移动若干秒,用于短距离位移。

  **Use this skill when:**
  - 机器人需要前进/后退/横移/原地转一定距离
  - 需要靠近或离开某个家具/物体
metadata: '{"nova": {"available": true, "kind": "builtin"}}'
---

# Move Base

## Workflow

1. 读取 `vx/vy/wz/duration` 参数。
2. 以 20Hz 向 `/nova/robocasa/action_cmd` 发布底盘速度指令持续 `duration` 秒。

## Interfaces

- Publish: `/nova/robocasa/action_cmd` (std_msgs/Float32MultiArray, 12 维)

## Params

- `vx` (float, 默认 0.0): 前进速度 [-1,1]。
- `vy` (float, 默认 0.0): 横移速度 [-1,1]。
- `wz` (float, 默认 0.0): 旋转速度 [-1,1]。
- `duration` (float, 默认 1.0): 移动秒数。

## Safety Rules

- 速度不超过 1.0,先小幅试动再加大。
- 移动期间不能同时做机械臂动作。

## Examples

- `{"vx": 0.5, "duration": 2.0}` 前进 2 秒。
- `{"wz": 0.3, "duration": 1.0}` 原地旋转 1 秒。
