# nova_executor_demo

示例 executor,演示如何把一个能力接入 NovaAgent 系统,也用于端到端验证。

## 提供的工具(全部为模拟实现)

| 工具 | 说明 | 参数 |
| --- | --- | --- |
| `wait` | 阻塞指定秒数 | `duration_sec` |
| `echo` | 返回传入文本 | `text` |
| `grasp` | 模拟抓取物体 | `object` |
| `place` | 模拟放置物体 | `object, surface` |

真实环境里,把工具换成 VLA / 传统机械臂控制即可(如 openpi 的 `act`、RTT 的 `plan/grasp`)。

## 接入方式

一个 executor 节点只需:

1. 为每个工具提供一个 `MCPExecute` action server(本例命名 `/nova_executor_demo/<tool>/execute`)。
2. 周期发布 `/nova/executors/heartbeat`(`ExecutorHeartbeat`,携带工具描述列表),executor_manager 便会自动发现;心跳超时则自动剔除。

## 运行

```bash
ros2 run nova_executor_demo nova_executor_demo_node
```
