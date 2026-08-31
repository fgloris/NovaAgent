# nova_console:Web 会话管理器 + agent 聊天面板 实施计划

> 目标:新增 ROS 包 `nova_console`。用 FastAPI + WebSocket + xterm.js 做一个 Web 界面:
> 1) 按 YAML 配置以"会话"(PTY 子进程)管理整套 NovaAgent 栈(sim server / bridge / pi0 / agentos 等),
>    支持 `depends_on` 依赖 DAG 与 `wait_for` 启动成功判断,按序自动拉起;
> 2) 左侧边栏切换会话、实时查看输出(可向会话 stdin 输入);
> 3) 底部常驻 agent 聊天面板:发送消息走 `RunTask`,实时显示 agent 每轮规划/工具调用/结果(`/nova/agentos/agent_msg`)。
> 4) agent CLI 增加 `/setup`(拉起整套)、`/sessions`(查会话状态)。

---

## 0. 背景与决策(已与用户确认)

| 决策点 | 结论 |
|---|---|
| UI 形态 | Web 页面(浏览器 + ssh 转发访问,默认 `127.0.0.1:8090`) |
| 前端技术栈 | FastAPI serve 原生 HTML/JS(零构建链)+ xterm.js(CDN)做终端面板 |
| 后端 | 复用 `fastapi` + `uvicorn`(**ROS 环境已装**,零新增依赖)+ `ptyprocess`(PTY) |
| 聊天面板 | 本次嵌入(server 内嵌 ROS 节点:订阅 `/nova/agentos/agent_msg`、调 `RunTask`) |
| 默认配置 | `config/sessions.yaml` 写 robocasa 闭环栈(sim→bridge/pi0→agentos) |
| 布局 | 左侧会话侧边栏;右侧上方终端;底部常驻聊天面板 |

---

## 1. 目标架构

```
浏览器(FastAPI 页面)
   │  WS /ws(会话输出 + 状态 + 聊天消息统一推送)
   │  REST(启动/停止/输入/发消息)
   ▼
nova_console server(FastAPI + uvicorn,跑在 ROS 环境)
   ├─ orchestrator:会话注册 + depends_on DAG + 按序启动
   │     └─ 每个会话 = PTY 子进程(bash -lc:source venv → pre → command)
   │          状态机:starting → running → ready(wait_for 匹配)→ exited/failed
   │          输出:后台线程读 PTY → 环形缓冲(内存)+ 每会话日志文件
   └─ ROS 节点(后台线程 spin):订阅 /nova/agentos/agent_msg(聊天数据源)+ RunTask client(发消息)
```

---

## 2. 包结构(`src/nova_console`)

```
src/nova_console/
  nova_console/
    config.py         # YAML 加载/校验(profiles + sessions)
    session.py        # PTY 会话运行时
    orchestrator.py   # 会话注册 + 依赖 DAG + 事件总线
    server.py         # FastAPI 应用 + ROS 节点 + main
    web/              # index.html / app.js / style.css(静态前端)
  config/sessions.yaml   # 默认 robocasa 闭环栈配置
  setup.py
  package.xml
  resource/nova_console
```

entry point:`nova_console_server = nova_console.server:main`

---

## 3. YAML 配置(`config/sessions.yaml`)

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
        wait_timeout_sec: 240
      - id: bridge
        name: robocasa bridge
        depends_on: [sim]
        pre: ["source ~/data1/lxy/env.sh"]
        command: ros2 run nova_robocasa_bridge robocasa_bridge_node
      - id: pi0
        name: pi0 server
        depends_on: [sim]
        venv: ~/data1/lxy/openpi/.venv
        workdir: ~/data1/lxy/NovaAgent
        command: python3 src/nova_vla_executor/nova_vla_executor/pi_server.py --checkpoint ...
        wait_for: "Application startup complete"
      - id: agentos
        name: AgentOS
        depends_on: [bridge, pi0]
        pre: ["source ~/data1/lxy/env.sh"]
        command: ros2 launch nova_agentos system.launch.py
```

字段:`id / name / venv? / workdir? / env? / pre[] / command / depends_on[] / wait_for? / wait_timeout_sec?`

启动语义:无依赖会话先起;某会话 `ready` 后,满足全部依赖的会话才拉起(DAG);`wait_for` 超时未匹配 → 标记 failed(进程可继续跑,由用户观察)。

---

## 4. 模块实现

### 4.1 `config.py`
- `load_profiles(path) -> dict`:读取 YAML,校验必填字段与依赖 id 存在;提供 `Profile` 数据结构。

### 4.2 `session.py`
- `class Session`:
  - `__init__(cfg)`:解析配置,构建 shell 脚本
    `source {venv}/bin/activate && {pre.join(' && ')} && {command}`
  - `start()`:用 `ptyprocess.PtyProcess.spawn(script, cwd=workdir, env=...)`(已装,自动 setsid);
    后台读线程 `os.read(proc.fd, ...)` → 增量解码(容忍跨块 UTF-8)→ 追加环形缓冲(`collections.deque`,保留 ~64KB)+ 追加日志文件;扫描 `wait_for` 正则 → 置 `ready` 并通知
  - `send_input(text)` / `stop()`(先 SIGINT,再 SIGTERM,超时 SIGKILL,按进程组 `os.killpg`)/ `restart()`
  - 状态:`created / starting / running / ready / exited / stopped / failed`;记录 exit code
- 进程组管理:整组发信号,避免 shell 子进程残留。

### 4.3 `orchestrator.py`
- `class Orchestrator`:
  - `load(profile)`,`start_profile(name)`:启动全部入度为 0 的会话;订阅会话状态变更,就绪后尝试启动依赖者
  - `sessions()` 列表;`stop_all()`
  - 事件总线:每个会话的输出/状态增量进一个线程安全队列,由 FastAPI 侧转成 asyncio 任务推给所有 WS 客户端

### 4.4 `server.py`
- 内嵌 ROS 节点 `ConsoleRosNode`:订阅 `agent_msg`(topic 可配,默认 `/nova/agentos/agent_msg`)、`RunTask` client(独立 callback group)
- FastAPI:
  - 静态:`GET /` → `web/index.html`
  - REST:
    - `GET /api/profiles`
    - `POST /api/start/{profile}`
    - `GET /api/sessions`
    - `POST /api/sessions/{id}/stop|restart`
    - `POST /api/sessions/{id}/input` body `{"text": "..."}`
    - `GET /api/chat?after=N`(聊天历史,轮询兜底)
    - `POST /api/chat` body `{"message": "..."}` → 调 `RunTask`,返回 task_id
  - `WS /ws`:推送 `{"type": "session", ...}` 与 `{"type": "chat", ...}`
- 线程模型:ROS 节点在后台线程 `spin`;会话读线程向队列写;FastAPI 事件循环用 `run_coroutine_threadsafe` 广播
- `main()`:uvicorn 起在 `127.0.0.1:8090`(端口可配)

### 4.5 `web/`
- `index.html`:布局 = 顶部工具栏(profile 启动/全部停止)+ 左侧会话侧边栏 + 右侧上方终端 `div` + 底部聊天面板
- `app.js`:
  - xterm.js(CDN)+ fit;WS 连接;按 session_id 分发输出到对应 Terminal 实例或切换重建
  - 侧边栏:会话名 + 状态色点,点击切换;每会话 stop/restart 按钮
  - 聊天:按 `kind` 着色渲染(agent绿/tool黄/result青/done粗绿/failed红);输入框发送
  - `POST /api/start/{profile}` 触发整套启动
- `style.css`:简洁暗色主题

### 4.6 CLI 扩展(`nova_agentos/agent_cli.py`)
- `/setup [profile]` → `POST {console}/api/start/{profile}`(console 地址可配参数,默认 `http://127.0.0.1:8090`)
- `/sessions` → `GET {console}/api/sessions` 打印各会话状态

---

## 5. 依赖变化(已调研可复用项)

**本机 ROS 环境已存在,零新增安装:**
- `fastapi 0.115.12` + `uvicorn 0.34.0`(后端 HTTP/WS)
- `websockets 15.0.1`(WS 库)
- `pexpect 4.8.0` + `ptyprocess 0.7.0`(PTY 子进程:用 `PtyProcess.spawn` 取代手写 `pty.openpty`)

**前端:** xterm.js 走 CDN,GPU 服务器无需安装 node/额外文件。

**调研结论(未采用):**
- `tmux` / `screen`(已装):可作会话基座,但无自定义侧边栏/WS 推送/`wait_for` 流式判断,不采用。
- `ttyd` / `gotty` / `wetty`(未装):web 终端服务器,单命令一个进程,难以做自定义侧边栏切换与整套 DAG 编排,不采用。
- `pyte`(未装):服务端 VT100 渲染,前端已用 xterm.js,不需要。

---

## 6. 验证

1. `colcon build --symlink-install`
2. 起 `nova_console_server`(ROS 环境)→ 浏览器(ssh 转发 8090)打开页面
3. 点"启动 robocasa_loop" → 观察 sim(等 `warmup done`)→ bridge/pi0 → agentos 依次 ready
4. 侧边栏切换会话看实时输出;向会话输入(如 agent CLI 会话)
5. 聊天面板发"请把桌面收拾干净" → 实时看到规划/工具调用/结果/完成
6. CLI `/setup`、`/sessions` 打通

---

## 7. 风险与注意

1. **server 需要 ROS 环境**:在 `env.sh` 环境里 `pip install fastapi uvicorn`;rclpy 依赖 ROS。
2. **ROS 与 asyncio 线程桥接**:订阅回调(ROS 线程)→ WS(asyncio)必须用 `run_coroutine_threadsafe`/线程安全队列,防止跨线程直接操作 asyncio。
3. **进程组清理**:会话 stop 必须杀整个进程组,防僵尸/残留(尤其 sim server 占 GPU)。
4. **wait_for 正则匹配**:在原始输出流上做,大小写敏感;`warmup done in` 等标记随实际输出调整。
5. **PTY 输出解码**:可能遇到部分 UTF-8 字符跨块,用增量解码(`errors='replace'`)。
6. **xterm.js CDN**:浏览器需联网;若内网无网则后续 vendor 到 `web/`。
7. **安全**:绑定 `127.0.0.1`,靠 ssh 转发;不做认证(本地开发工具)。
