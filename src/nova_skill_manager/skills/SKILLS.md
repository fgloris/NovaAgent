# NovaAgent Skills

技能是 Markdown 指令包,扩展 AgentOS 的能力。执行由 skill_manager 节点的执行器完成,这里只放契约与说明。

## 目录位置

- 内置技能: 包内 `skills/<skill-name>/SKILL.md`
- 自定义技能: `<workspace_skills_dir>/<skill-name>/SKILL.md`(默认 `~/novaagent/skills`),同名覆盖内置

## Frontmatter

```yaml
---
name: skill-name
description: |
  简短的能力说明,含 **Use this skill when:** 触发条件。
metadata: '{"nova": {"available": true, "kind": "builtin", "executor": "module:ExecutorClass"}}'
---
```

- `executor`: 执行器 `module:Class`,缺省按技能名匹配内置执行器(reset/wait/move_base)。
- `requires.bins / requires.env`: 可用性检查(二进制是否存在、环境变量是否设置)。

## 内置技能

- `reset`: 重置 robocasa 环境。
- `wait`: 等待若干秒,期间持续发零动作。
- `move_base`: 底盘朝 (vx, vy, wz) 移动若干秒。
