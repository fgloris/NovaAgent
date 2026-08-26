# 8.24
![alt text](<Screenshot from 2026-08-24 20-11-46.png>)
opencode -s ses_fcc87a673ffes3DTdEUZaCf9Fn
未完成：
1. 未完成仿真执行闭环。skill里面都是假的，nova_executor也是假的。
2. manager现在没有verify action是否成功，对任务没有持续追踪。
3. 还没有确认各个话题是否合理。

计划：
1. 先用weijia h200服务器跑一下VLA backend,看看连过来到底能不能成。
2. 看看能不能把LIBERO的仿真连上。(已经完成环境libero的搭建)
3. 试一下GraspNet。

# 8.25 统一仿真话题 + 动态 topic 转发(plan.md 已执行,代码完成,待验证)
- [x] P1: `nova_common` 抽取 `obs_codec`(encode/decode/summarize/normalize/build_obs_spec)、`jsonline`(JsonLineClient/JsonLineServer)、`env_bridge`(EnvBridgeBase)
- [x] P2: 两 sim server 自描述(reset/step 响应带 `action_spec`/`obs_spec`/`sim_info`);robocasa 键名归一化 + `state.instruction` + 动作转换挪入 sim server
- [x] P3: 两 bridge 继承 `EnvBridgeBase`,统一发布 `/nova/env/*`;`random_action_client` 改发 `/nova/env/action_cmd` 且 `action_dim` 参数化
- [x] P4: `ToolDescriptor` 加 `obs_bindings`;新增 `MapTopics`/`UnmapTopics`/`EnvInfo` srv(需 colcon build 重新编译 nova_interfaces)
- [x] P5: `topic_router`(MapTopics/UnmapTopics 服务,独立节点) + `dag_executor` 前后钩子 + `agentos` 动态转发/命名空间注入/异常清理 + `system.launch.py`
- [x] P6: executor_demo 新增 `pi0_policy` 演示工具(bindings + session 订阅 + 动作回灌)
- [ ] 验证:先 `colcon build --symlink-install`,再按 plan.md §9 做 9.2(单桥)/9.3(router)/9.4(端到端)手动验证

![alt text](<Screenshot from 2026-08-25 18-21-30.png>)

打通闭环后做完整验证，再做其他工作。

# 8.26