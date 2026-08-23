---
name: reset
description: |
  重置 robocasa 仿真环境到初始状态,清空 step_count 并发布初始观测。

  **Use this skill when:**
  - 任务开始前需要初始化环境
  - 当前环境状态已 done 或卡住需要重新开始
  - 用户要求"重新开始/重置环境"
metadata: '{"nova": {"available": true, "kind": "builtin"}}'
---

# Reset

## Workflow

1. 调用 `/nova/robocasa/reset` 服务。
2. 等待 reset 完成并确认成功标志。

## Interfaces

- Service: `/nova/robocasa/reset` (std_srvs/Trigger)

## Safety Rules

- 重置会丢弃当前环境状态与已执行步骤,重置前确认没有未完成任务。

## Examples

- 无参数。
