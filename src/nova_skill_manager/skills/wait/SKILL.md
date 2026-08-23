---
name: wait
description: |
  让机器人等待若干秒,期间持续发送零动作让仿真继续推进。

  **Use this skill when:**
  - 需要给运动留出缓冲时间
  - 任务流程中需要短暂停顿
metadata: '{"nova": {"available": true, "kind": "builtin"}}'
---

# Wait

## Workflow

1. 读取 `duration` 参数(秒)。
2. 持续发布零动作直到等待结束。

## Interfaces

- Publish: `/nova/robocasa/action_cmd` (std_msgs/Float32MultiArray, 12 维零向量)

## Params

- `duration` (float, 默认 1.0): 等待秒数。
- `zero_action` (bool, 默认 true): 等待期间是否发零动作。

## Safety Rules

- 等待时长不宜过长,避免长时间无进展。

## Examples

- `{"duration": 2.0}`
