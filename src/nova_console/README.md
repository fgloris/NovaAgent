# nova_console

NovaAgent 的 Web 控制台:PTY 会话管理器 + agent 聊天面板。

- **会话管理**:按 `config/sessions.yaml` 以 PTY 子进程编排整套栈(sim server / bridge / pi0 / agentos),支持 `depends_on` 依赖 DAG 与 `wait_for` 启动成功判断,按序自动拉起。
- **Web 界面**:左侧会话侧边栏(状态色点,点击切换);右侧上方 xterm.js 终端(实时输出,可向会话 stdin 输入);底部常驻 agent 聊天面板(发消息走 `RunTask`,实时显示 agent 规划/工具调用/结果,来自 `/nova/agentos/agent_msg`)。
- **CLI 联动**:`nova_agentos_cli` 增加 `/setup [profile]`(拉起整套)、`/sessions`(查会话状态)。

## 运行

```bash
# 需要 ROS 环境(已装 fastapi/uvicorn/ptyprocess/websockets)
ros2 run nova_console nova_console_server            # 默认 127.0.0.1:8090
# 或指定配置/参数
nova_console_server --config <sessions.yaml> --port 8090 \
  --agent-msg-topic /nova/agentos/agent_msg --run-task-service /nova/agentos/run
```

浏览器访问(远程则先 ssh 转发):
```bash
ssh -f -N -L 8090:localhost:8090 <gpu-server>
# 打开 http://localhost:8090
```

## YAML 配置(`config/sessions.yaml`)

```yaml
profiles:
  robocasa_loop:
    sessions:
      - id: sim
        name: RoboCasa 仿真
        venv: ~/data1/lxy/robocasa/.venv
        workdir: ~/data1/lxy/NovaAgent
        command: python3 src/nova_robocasa_bridge/nova_robocasa_bridge/robocasa_sim_server.py
        wait_for: "warmup done in"
        wait_timeout_sec: 300
      - id: bridge
        name: robocasa bridge
        depends_on: [sim]
        pre: ["source ~/data1/lxy/env.sh"]
        command: ros2 run nova_robocasa_bridge robocasa_bridge_node
      # ...
```

会话字段:
| 字段 | 说明 |
| --- | --- |
| `id` / `name` | 会话标识 / 显示名 |
| `venv` | 可选,启动前 `source <venv>/bin/activate` |
| `workdir` | 可选,工作目录 |
| `pre` | 可选,主命令前执行的前置命令(`source env.sh` 等) |
| `command` | 主命令 |
| `depends_on` | 依赖会话 id 列表,全部 ready 后才启动 |
| `wait_for` | 输出中出现该文本(正则)即 ready;不配则 running 即视为 ready |
| `wait_timeout_sec` | 匹配超时,超时标记 failed(进程继续跑) |

启动语义:无依赖会话先起 → 依赖会话 `ready` 后,DAG 逐级拉起。

## API

```
GET  /                      # Web 页面
GET  /api/profiles
POST /api/start/{profile}
GET  /api/sessions
POST /api/sessions/{sid}/stop|restart
POST /api/sessions/{sid}/input   body: {"text": "..."}
GET  /api/chat?after=N
POST /api/chat               body: {"message": "..."}
WS   /ws                     # 会话输出/状态 + 聊天消息统一推送
```

## 模块

| 文件 | 职责 |
| --- | --- |
| `config.py` | YAML 加载/校验(profiles + sessions) |
| `session.py` | PTY 会话运行时(ptyprocess):状态机 / wait_for / 环形缓冲 + 日志 / stdin / 整组 stop |
| `orchestrator.py` | 会话注册 + depends_on DAG + 事件出口 |
| `server.py` | FastAPI(静态页 + REST + WS)+ 内嵌 ROS 节点(聊天桥) |
| `web/` | 前端(index.html / app.js / style.css,xterm.js 走 CDN) |

## 注意

- server 需在 ROS 环境跑(rclpy);会话编排本身不依赖 ROS,各会话用各自 venv 自洽。
- 聊天面板需要 agentos 在跑(`/nova/agentos/agent_msg` 才有数据)。
- 绑定 `127.0.0.1`,靠 ssh 转发访问;本地开发工具,无认证。
- xterm.js 走 CDN,浏览器需联网;内网环境可后续 vendor 到 `web/`。
