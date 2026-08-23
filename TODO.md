一、HoloAgent 的 skill 管理
位置：agentic_robot/agentOS/holoagent_skills/，核心思路是 *"Markdown 即注册表"*，没有运行时、没有 ROS 节点、没有可执行注册。
目录契约（每个技能一个目录）：
skills/<skill-name>/
├── SKILL.md      # 机器/LLM 面向的技能规范（必需）
├── README.md     # 人读的使用指南
├── scripts/      # 可执行辅助脚本
└── assets/       # 示例命令/提示词片段
SKILL.md 格式（skills/workflow/SKILL.md、skills/rel-move-skill/SKILL.md）：
- YAML frontmatter：name + description，description 里强制写 **Use this skill when:** 触发条件（这就是喂给 LLM 的"何时用"信号）
- 正文段落：## Workflow、## Safety Rules、## Interfaces（HTTP / ROS topic / script 入口）、## Examples
管理工具（scripts/ 下的 CLI 脚本，CRUD）：
- create_skill.py：按模板生成目录骨架
- list_skills.py：扫描 skills/，用正则提取 frontmatter description
- show_skill.py / delete_skill.py / validate_skills.py / sync_legacy_docs.py
- validate_skills.py 校验规则：frontmatter 头、description: |、**Use this skill when:**、## Workflow、
## Safety Rules 是否齐全
实际消费方式：真正的 LLM 循环在 chatbot/g1/g1chat_demo_multirobot_dag.py，LLM 输出的是一个 DAG（节点含 skill + target 字段），但 skill 枚举是代码里硬编码的 {navigation, arm}，SKILL.md 更多是给 LLM 的提示文档，执行派发还是手写 if/else。即：文档系统是开放的，执行系统是封闭的。
特点小结：轻量、纯文件、无状态；无技能可用性/依赖检查、无 session 协议、无运行时注册；技能本质是"教 LLM 怎么调用机器人 HTTP/ROS 接口的指令包"。
二、PhyAgentOS 的 skill 管理
核心是 认知/物理解耦 + Session 中心运行时，把技能拆成两层：
1. Agent Skills（认知层，LLM 面向）
- 位置：workspace 的 skills/<name>/SKILL.md（可覆盖内置 PhyAgentOS/skills/），由 agent/skills.py 的 SkillsLoader 管理
- 渐进式加载（agent/context.py 组装 system prompt）：
1. 元数据（name+description）常驻上下文（约 100 词）
2. always:true 的技能直接全文进 context
3. 其余技能以 XML 摘要 形式列出：<skills><skill available="true"><name>..<description>..<location>..<requires>..
4. LLM 按需用 read_file 读完整 SKILL.md
- 依赖/可用性检查：frontmatter 里可写 metadata: {"PhyAgentOS":{"always":false,"available":true,"requires":{"bins":[...],"env":[...],"runtime":{"enabled":true,"target_kind":"simulation","benchmark":true}}}}，_check_requirements 会检查 CLI 二进制、环境变量、以及 TARGETS.md/SKILLRUNTIME.md 里的运行时能力，不满足就标 available="false" 并列出缺什么
2. Skill Runtimes（物理层，执行契约）
- SKILLRUNTIME.md 是注册表，每项是 pydantic SkillRuntimeSpec（schemas/skillruntime.py）：
id, runtime(类名), runtime_kind(policy|builtin), loop_mode, agent_exposure, supported_target_kinds, policy(客户端+adapter), observation_contract, target_tool_policy, benchmark, input/output contract, adapter_requirements
- 两类运行时，都继承 runtime/skillruntime/base.py 的 BaseSkillRuntime（start/cancel/snapshot）：
- PolicySkillRuntime（如 OpenPISkillRuntime）：策略闭环（观测→VLA→动作块）
- BuiltinSkillRuntime（如 LiberoBenchmarkSkillRuntime、CommandSimSkillRuntime）：确定性内置循环
- Session 执行管线：Agent 把 Session 追加到 SESSIONS.md → WatchdogSupervisor 认领 → SessionRunner → SkillRuntime.start → TargetSessionHandle（走 target adapter）→ 结果回写 SESSIONS.md。全程 Markdown/YAML 可审计
- 配套：TARGETS.md（目标注册）、多级安全（Critic 校验→Preflight 契约检查→Target 侧 SafetyGuard→Operator override）
特点小结：最成熟。技能=文档（认知）+ 注册的运行时（物理），新机器人只写一个 adapter（~100 行），新技能=一个类+一条注册表项；可用性检查、渐进披露、会话协议一应俱全，但较重，且核心是"Agent 写文件、watchdog 轮询"而非 ROS 事件驱动。
三、对比结论

你的场景（robocasa + ROS2）最适合：取 PhyAgent 的"认知/物理分层 + 可用性检查 + 渐进披露"，取 HoloAgent 的"SKILL.md 轻量契约 + 校验脚本"，用 ROS2 服务/话题替换文件协议做事件驱动。
四、NovoAgent 计划（两个 ROS2 节点）
```
                        ┌──────────────────────────────────────────────┐
 robocasa 观测 --------->│  nova_agentos  (LLM 驱动的 AgentOS 节点)      │
 state/相机/success      │  ① 构建上下文(技能XML摘要+SKILL.md+观测+指令)   │
                        │  ② LLM 选技能+参数, 长任务分解成技能序列         │
                        │  ③ 发 /nova/skills/cmd, 订阅 result 校验重规划 │
                        └──────┬──────────────────────────┬────────────┘
                      /nova/skills/cmd (String JSON)      │
                               ▼                          │ result
                     ┌────────────────────────────────────┐   │
                     │ nova_skill_manager (SKILL节点)      │◄──┘
                     │  注册表: skills/<name>/SKILL.md     │
                     │  CRUD服务+可用性检查(仿PhyAgent)      │
                     │  执行: builtin(指令类) / policy(VLA) │
                     └─────────────┬───────────────────────┘
                                   ▼
                          /nova/robocasa/action_cmd (12维动作)
```

技能注册表（磁盘目录，放 nova_skill_manager 包内）：
```
skills/<skill-name>/
├── SKILL.md        # frontmatter: name/description(+**Use when**)/metadata{nova{kind,executor,requires}}
├── executors/<name>.py   # 执行器类, 可选
├── references/     # 需按需加载的文档
└── assets/
```
SKILL.md frontmatter（融合两者）：
---
name: pick_and_place
description: |
  把物体从台面拿到柜子里。**Use this skill when:** 用户要求抓取/放置某物体…
metadata: '{"nova": {"always": false, "available": true,
           "kind": "builtin",   # 或 policy(接VLA闭环)
           "executor": "nova_skill_manager.executors.pick_place:PickPlaceExecutor",
           "requires": {"runtime": {"robocasa": true}},
           "inputs": {"object": "str", "target": "str"}}}'

## Workflow / Safety Rules / Interfaces(ROS topic,service) / Examples
接口设计（沿用你现有 String JSON 风格，避免自定义 msg 编译）：
- /nova/skills/list、/nova/skills/get、/nova/skills/load、/nova/skills/validate、/nova/skills/create、/nova/skills/delete（服务，String 进出）
- /nova/skills/cmd（AgentOS 发）、/nova/skills/result（skill manager 回）
- /nova/agentos/task（任务状态，供调试/周报录屏）
实现顺序：
1. nova_skill_manager 包：注册表 + CRUD/validate 服务 + 可用性检查（移植 PhyAgent SkillsLoader 思路到 ROS）
2. 内置 builtin 技能：reset、wait、move_base（直接驱动 /nova/robocasa/action_cmd），先打通"指令→执行→result"闭环
3. nova_agentos 包：上下文构建（XML 技能摘要 + 按需读 SKILL.md）+ 技能选择 + 结果校验/重规划
4. 长任务分解（workflow 技能），后续可挂 PDDL/HTN 校验（对应你 TODO 的 TAMP 方向）
5. 接 policy 型技能：VLA/diffusion policy（robocasa365 checkpoint 已有），skill manager 只做 client
开始前有几个决策想确认：

# Questions
  两个节点放在哪个 ROS2 包结构里？ -- 新建两个独立包 (推荐)
  AgentOS 的 LLM 用哪个？ -- 远程API,可在配置文件里配多个provider,支持OpenAI和Anthropic.
  ROS 接口用什么形式？ -- 自定义 msg/srv 接口包
  第一个要写死的技能用什么验证闭环？ -- 纯 builtin 指令类 (推荐)

已完成三个包的代码并编译通过，纯 Python 逻辑（loader/validator/LLM 回退/JSON 解析）都自测通过。剩下 ROS 相关的验证需要你执行。
我做了什么
```
NovaAgent/src/
├── nova_interfaces/          # 自定义 msg/srv (已编译)
│   ├── msg/  SkillInfo SkillCommand SkillResult TaskState
│   └── srv/  ListSkills GetSkill LoadSkills ValidateSkills CreateSkill DeleteSkill
├── nova_skill_manager/       # SKILL 管理节点
│   ├── skills_loader.py      # 仿 PhyAgent: workspace 覆盖 builtin + bins/env 可用性检查
│   ├── skills_validator.py   # 仿 HoloAgent: SKILL.md 结构校验
│   ├── executors/            # reset / wait / move_base 内置执行器
│   ├── skill_manager_node.py # 注册表服务 + /nova/skills/cmd 派发 + 线程池执行
│   └── skills/               # reset wait move_base 的 SKILL.md
└── nova_agentos/             # AgentOS 节点
    ├── llm_client.py         # 零依赖 urllib, openai/anthropic 多 provider 回退
    ├── context_builder.py    # 技能 XML 摘要(仿 PhyAgent 渐进披露)
    └── agentos_node.py       # 状态机: 任务→LLM选技能→下发→result校验→重规划
```
话题/服务：/nova/skills/cmd、/nova/skills/result、/nova/agentos/task、/nova/agentos/goal，服务 /nova/skills/{list,get,load,validate,create,delete}。

1. 启动 skill_manager 并调服务（终端 A）
ros2 run nova_skill_manager skill_manager_node
（终端 B）
ros2 service call /nova/skills/list nova_interfaces/srv/ListSkills "{}"
ros2 service call /nova/skills/validate nova_interfaces/srv/ValidateSkills "{}"
2. 不发仿真也能测 cmd→result 闭环（skill_manager 终端 A 保持运行，终端 B）
ros2 topic echo /nova/skills/result &
ros2 topic pub -1 /nova/skills/cmd nova_interfaces/msg/SkillCommand \
  "{task_id: 't1', skill_id: 'wait', goal: '等待', params_json: '{\"duration\": 2}'}"
预期看到 result：先 running 再 succeeded（wait 只需 sleep，不依赖 sim）。
3. AgentOS 冒烟（需要设 LLM key，如 export DEEPSEEK_API_KEY=...；先只跑节点不发任务，验证不崩）
ros2 run nova_agentos agentos_node
若没配 key，节点启动时会因 llm.yaml 缺 key 在首个决策时报错——这是预期行为，可用 provider 回退。