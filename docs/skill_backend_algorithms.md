# NovoAgent Skill Backend 算法调研汇总

> 适用环境: robocasa (MuJoCo 厨房场景) + Panda Omron 移动单臂 + ROS2 bridge
> 目标: LLM 驱动 AgentOS 选择技能, skill backend 完成可执行运动。
> 本文分两部分:**A. 传统规划算法**(确定性、无需训练,可直接跑) / **B. 非 VLA 的 learning-based 方法**(GraspNet 一类,需权重/推理,但无需大语言模型)。
> 鲁棒性评级 ★1-5。

---

## Part A. 传统规划算法

### A1. 任务规划层(符号级,给 LLM 当"大脑校验器")

| 算法/框架 | 说明 | 鲁棒性 | 备注 |
|---|---|---|---|
| PDDL + Fast Downward / FF / LAMA | 经典 STRIPS 规划器,符号状态搜索 | ★★★★★ | 依赖准确状态提取;厨房半结构化场景非常契合 |
| HTN (SHOP2) | 分层任务网络,用领域知识分解任务 | ★★★★★ | 对"把X放进柜子"这类层级技能天然匹配 |
| PDDLStream | 符号规划 + 采样流,经典 TAMP 框架(MIT) | ★★★★★ | Python 可跑,社区成熟,是你 TODO 里 TAMP 方向的最佳起点 |
| Logic-Geometric Programming (LGP) | 符号层与几何层交替求解 | ★★★★ | 理论优美,实现/调参较重 |
| 规划即 SAT/CP (HPlan) | 编码成约束求解 | ★★★ | 长任务求解快,但建模成本高 |

### A2. 运动规划层(采样类,OMPL/MoveIt 主力)

| 算法 | 说明 | 鲁棒性 | 备注 |
|---|---|---|---|
| RRT-Connect | 双向快速扩展 | ★★★★★ | 概率完备 + 快,工业最常用,首选 |
| RRT* / Informed RRT* / BIT* | 渐近最优 | ★★★★ | 质量好但慢;高维 C 空间退化明显 |
| PRM / PRM* / Lazy PRM | 多查询先建图 | ★★★★ | 适合静态场景,动态障碍弱 |
| FMT* | 批次启发式,近最优 | ★★★★ | 实现复杂 |
| SBL | 双向懒评估,少碰撞检测 | ★★★★ | 比 RRT 省算力 |

### A3. 轨迹优化层(给采样结果做平滑/最优)

| 算法 | 说明 | 鲁棒性 | 备注 |
|---|---|---|---|
| TrajOpt (SQP) | 序贯二次规划 | ★★★★★ | 对初始解不敏感,操作类任务最稳 |
| CHOMP | 梯度下降 + 碰撞梯度 | ★★★★ | 局部最优,需好初值 |
| STOMP | 策略搜索,抗局部极小 | ★★★★ | 更鲁棒但慢 |
| ITOMP | 增量式,动态重规划 | ★★★★ | 适合有扰动的仿真 |
| GPMP2 | 高斯过程 + 因子图 | ★★★★ | 轨迹平滑自然,实现重 |
| OCS2 / Crocoddyl (DDP/MPC) | 最优控制 | ★★★★★ | 需动力学模型,适合真机部署前仿真验证 |

### A4. 底层技能/控制原语层(skill backend 落地点)

| 方案 | 说明 | 鲁棒性 | 备注 |
|---|---|---|---|
| 笛卡尔点对点 + RRT-Connect (MoveIt) | 端到端关节规划 | ★★★★★ | 最成熟基线 |
| OSC / 阻抗控制 | 末端阻抗跟踪笛卡尔轨迹 | ★★★★★ | 对模型误差鲁棒,建议执行层用它 |
| DMP (动态运动基元) | 示教轨迹参数化 | ★★★★ | 适合插/取/放置原语,可避障变形 |
| A* / TEB / DWA | Omron 底盘导航 | ★★★★ | nav2 标配 |
| Behavior Tree / State Machine | 组织技能 + 失败重试 | ★★★★★ | 工程鲁棒性的关键一环 |

### A5. 完整 TAMP 框架(可直接参考的开源 repo)

- **pddlstream** (MIT, Python) — 符号 + 采样流
- **FFRob / OSCAR** (CMU) — 深度强化学习 + TAMP
- **HLS** (层次化 LGP)
- 思路: LLM 出高层意图 → PDDL 校验/补全 → 采样求解几何层 → 轨迹优化 → 阻抗执行

### A6. 传统层鲁棒性结论

1. 任务层: LLM 出计划 + **PDDL/HTN 校验** 远比纯 LLM 可靠。
2. 运动层: 首选 **RRT-Connect**(OMPL 有 Python binding,robocasa 里直接用 MuJoCo 做 FK/碰撞)。
3. 执行层: **OSC/阻抗控制** 而不是纯位置跟踪。
4. 碰撞检测可直接用 MuJoCo 自身状态,无需另建 SDF。

---

## Part B. 非 VLA 的 Learning-Based 方法

> 分类: B1 抓取位姿检测 / B2 交互与 Affordance(铰接物体,适配橱柜抽屉) / B3 Pick-Place 与视觉操作策略 / B4 灵巧手抓取 / B5 数据与评测。

### B1. 抓取位姿检测(平行夹爪 6-DoF 抓取)

| 方法 | 输入→输出 | 鲁棒性 | 实现/部署 | 对你场景的契合度 |
|---|---|---|---|---|
| **GraspNet-1Billion + GraspNet-Baseline** | 点云→抓取位姿+置信度,1.1B 标注 | ★★★★★ 大规模数据,泛化好 | 开源,需GPU推理,可离线生成抓取候选 | 高,robocasa 物体多为已知类别,直接可用 |
| **AnyGrasp** (GraspNet 家族) | 深度图→稠密抓取+运动学过滤,实时闭环 | ★★★★★ 对乱堆/未知物体强 | 开源 SDK,实时,可接入闭环 | 高,仿真里替代"机械式采样抓取" |
| **Contact-GraspNet** (NVIDIA) | 点云→接触抓取,zero-shot 新物体 | ★★★★★ 零样本泛化强 | 开源,较吃显存 | 高,新物体多时适用 |
| **6-DoF GraspNet** (Mousavian) | 点云+VAE→多样抓取位姿 | ★★★★ | 开源 | 中高 |
| **PointNetGPD** | 点云→抓取质量打分 | ★★★★ | 轻量 | 中,可当打分器 |
| **VGN (Volumetric Grasping Network)** | 截断 SDF→抓取+碰撞+质量,实时 | ★★★★ | 开源,ETH 实现 | 中高,体素化与 MuJoCo 网格契合 |
| **Dex-Net 2.0** (Berkeley) | 深度图→鲁棒抓取,抗传感器噪声 | ★★★★ | 开源,Amazon 系 | 中,对仿真噪声鲁棒性已验证 |

### B2. 交互与 Affordance(铰接物体/橱柜/抽屉)

| 方法 | 说明 | 鲁棒性 | 备注 |
|---|---|---|---|
| **Where2Act** (NVIDIA) | 从被动观测学交互 affordance,给定单张图即可选动作 | ★★★★★ 跨类别泛化 | 开源;橱柜开关门/抽屉这类 robocasa 任务正合适 |
| **UMPNet** | 通用操作策略网络,铰接物体交互推理 | ★★★★ | 与 Where2Act 同源思路 |
| **GIGA** | 交互图学习,铰接物体的因果关系 | ★★★★ | 学术参考价值高 |
| **Form2Fit** (Caltech) | 关键点驱动的装配/放置,sim2real 强 | ★★★★ | 适合"放物体进容器"类技能 |
| **FlowBot3D** (NVIDIA) | 场景流→稠密抓取方向,乱堆鲁棒 | ★★★★ | 与 B1 抓取可互补 |

### B3. Pick-Place 与视觉操作策略(非 VLA 的策略网络)

| 方法 | 说明 | 鲁棒性 | 备注 |
|---|---|---|---|
| **Transporter Nets** (Google) | 抓取+放置双模块,关键点匹配,RLBench 标配 | ★★★★★ | 开源 (ravens);robocasa 的 pick-place 任务可先跑通它 |
| **CLIPort** | Transporter + CLIP 语言条件,指令化 pick-place | ★★★★★ | 开源;可直接对接你 LLM 下发的中层指令 |
| **Diffusion Policy** | 条件去噪扩散生成动作序列,多模态鲁棒 | ★★★★★ SOTA 之一 | 开源;robosuite 系环境现成支持 |
| **DP3 (3D Diffusion Policy)** | Diffusion Policy + 点云表示,干扰下极稳 | ★★★★★ 更强 | 开源;与 robocasa 3D 场景契合 |
| **ACT (Action Chunking with Transformers)** | CVAE + Transformer 动作块,稳定 pick-place | ★★★★★ | 开源;ALOHA 移动操作验证过 |
| **IBC** (隐式行为克隆) | 能量函数建模动作 | ★★★★ | 与 Diffusion Policy 同源流派 |

### B4. 灵巧手抓取(参考,当前 Panda 平行夹爪用不上)

| 方法 | 说明 |
|---|---|
| **DexGraspNet** | 灵巧手抓取数据集 + 优化生成器 |
| **UniDexGrasp** | 统一灵巧抓取学习框架 |

### B5. 数据与评测基准

- **GraspNet-1Billion benchmark**: 抓取算法标准评测集。
- **Dex-Net 数据集**: 鲁棒抓取度量。
- **robocasa365 (RoboCasa)**: 你已在用的仿真/数据集,可直接喂给 B1/B3 方法做验证。

### B6. Learning-Based 层鲁棒性结论

1. 抓取: 首选 **AnyGrasp / GraspNet-Baseline**,robocasa 物体类别已知、场景受控,不需要 VLA 也能拿到高质量的 6-DoF 抓取位姿。
2. 铰接/容器交互: **Where2Act** 或 **Transporter 放置模块** 处理开柜/放入。
3. 端到端策略验证: **DP3 / Diffusion Policy** 是在 robosuite 系环境(robocasa 同源)上最鲁棒的基线,可做 ablation 对照。
4. 关键点: 这些方法都只需 **点云/深度 + 类别标注**,可在你的 CPU/单卡上先跑通数据闭环,再决定是否引入 VLA。

---

## Part C. 推荐组合(与你的 AgentOS 分工)

```
LLM AgentOS (高层任务分解 + 技能选择)
        │  下发中层指令 (自然语言 / 技能名 + 参数)
        ▼
skill backend 路由器
   ├─ 符号校验: PDDL/HTN (可选,长任务时启用)
   ├─ 感知: 点云/深度 → GraspNet/AnyGrasp 出抓取位姿,Where2Act 出交互点
   ├─ 规划: RRT-Connect (OMPL) 出关节路径 → TrajOpt 平滑
   ├─ 执行: OSC/阻抗跟踪, Behavior Tree 编排 + 失败重试
   └─ 可选对照: Diffusion Policy / DP3 / ACT 端到端策略
```

**快速验证路径建议**: robocasa 里先做 `RRT-Connect 规划 + 阻抗执行` 的确定性闭环(跑通 skill 层),再用 `GraspNet-Baseline/AnyGrasp + Transporter` 替换机械式抓取,最后用 `DP3` 作为端到端对照。这与"Hybrid WM + TAMP"的论文方向可衔接。


以下是传统（非深度学习）机械臂 manipulation 规划算法的常见分类整理，基于 OMPL、MoveIt 及经典教材（LaValle《Planning Algorithms》、Latombe）的主流内容：
1. 任务与操作规划层（Task & Manipulation Planning, TAMP）
- 符号任务规划 + 底层运动规划的混合求解：如 PDDL 任务规划器（FastDownward、FF）与运动规划器耦合，代表作 TMKit、PDDLStream、Asai & Fukunaga 的符号化方法。
- 多模态运动规划（Multi-modal / manipulation-aware planning）：处理"抓取→携带→放置"等模式切换，OMPL 支持 multi-modal 规划。
- 经典思路：先在离散任务空间排动作序列，再为每个动作解连续运动。
2. 抓取规划（Grasp Planning）
- 抓取候选生成：基于几何的 GraspIt!（Graspit/Dart）、以手模型枚举抓取姿态。
- 力封闭（Force Closure）判定：经典抓取质量评估。
- 从物体 CAD/点云生成抓取：分解凸包 + 接触点采样（如 dex-net 之前的几何方法）。
- 常用库：GraspIt!、MoveIt 的 Pick & Place 管线（Gripper 姿态 + 逆运动学 + 轨迹）。
3. 逆运动学与笛卡尔规划（IK & Cartesian Planning）
- 解析/数值 IK（KDL、IKFast、Trac-IK），MoveIt 的 Cartesian path 规划（增量式笛卡尔插值 + IK 校验）。
4. 运动规划（Motion Planning）——最核心部分
采样类（sampling-based，机械臂标配）：
- 单查询树：RRT、RRT-Connect（双向，OMPL/MoveIt 默认常用）、EST、KPIECE/BKPIECE/LBKPIECE（OMPL 默认）、SBL。
- 渐近最优：RRT\*、PRM\*、Informed RRT\*、BIT\*、AIT\*、FMT\*、LazyPRM\*。
- 多查询路图：PRM、LazyPRM、SPARS/SPARS2。
- 多层级/商空间：QRRT/QMP（multilevel，提速高维规划）。
图搜索类：A\*、Dijkstra、D\*（动态重规划）、Theta\*（多用于移动底盘，机械臂高维用得少）。
势场/局部规划：人工势场法（APF）、弹性带 Elastic Band、动态窗口 DWA（主要用于移动机器人，机械臂局部避碰也可用）。
轨迹优化（Trajectory Optimization）：
- 后处理平滑：Path Simplifier（shortcutting）、hybridization。
- 直接优化：CHOMP（协变哈密顿轨迹优化）、STOMP（随机优化）、TrajOpt（序列凸优化）、TOPP-RA（时间最优参数化）。
- 用时基样条插值 + 加加速度限制的轨迹生成（如 MoveIt 的 Time-optimal Trajectory Generation）。
5. 约束运动规划（Constrained Planning）
机械臂常见末端位姿约束/保持接触：CBiRRT2（约束双向 RRT）、OMPL 的 constrained planning 支持（约束流形上的采样）。
6. 经典库与工具链（传统方法实践标准）
- OMPL（Open Motion Planning Library）——采样类算法最全。
- MoveIt（ROS）——OMPL + IK + 碰撞检测 + Pick&Place 完整管线。
- OpenRAVE / Robowflex / TMKit——研究和 TAMP 实验平台。
- KDL / Pinocchio / FCL——运动学与碰撞检测底层。
一句话选型建议：工程上最常用 RRT-Connect / BKPIECE 找初解 + 轨迹优化（CHOMP/STOMP）做平滑 + 笛卡尔段拼接 + TAMP（PDDLStream 类）排任务；追求最优解则用 RRT\*/Informed RRT\*/BIT\*。