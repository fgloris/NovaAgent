/* NovaAgent Console 前端:会话侧边栏 + xterm 终端 + 聊天面板。
 * 输出/状态优先走 WS;WS 断连时自动切 REST 轮询兜底(增量 out_seq),保证输出始终可见。 */
"use strict";

const state = {
  profiles: [],
  activeProfile: null,
  sessions: {},   // sid -> {id,name,status,exit_code,depends_on,out_seq}
  current: null,  // 当前选中的 sid
  terminals: {},  // sid -> {term, fit}
  outSeq: {},     // sid -> 已写入的最新输出 seq
  unread: {},     // sid -> 后台 session 新输出块数量
  ws: null,
  pollTimer: null,
  keepTimer: null,
  chatSeq: 0,
};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(path + " -> " + r.status);
  return r.json();
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function setConn(cls) {
  const el = document.getElementById("conn");
  el.className = "conn " + cls;
  el.textContent = { on: "已连接", off: "未连接", poll: "未连接(轮询中)", conn: "连接中…" }[cls] || cls;
}

/* ---------------- 会话侧边栏 ---------------- */
function statusColor(s) {
  return { created: "created", starting: "starting", running: "running",
           ready: "running", exited: "exited", stopped: "stopped", failed: "failed" }[s] || "created";
}

function statusText(s) {
  return { created: "待启动", starting: "启动中", running: "运行中",
           ready: "就绪", exited: "已退出", stopped: "已停止", failed: "失败" }[s] || s;
}

function fmt(v) {
  if (v === undefined || v === null || v === "") return "-";
  if (Array.isArray(v)) return v.length ? v.join(", ") : "-";
  return String(v);
}

function sessionBanner(s) {
  const lines = [
    `session: ${s.name || s.id} (${s.id})`,
    `command: ${s.script || s.command || "-"}`,
  ];
  if (s.workdir) lines.push(`workdir: ${s.workdir}`);
  if (s.venv) lines.push(`venv: ${s.venv}`);
  if (s.depends_on && s.depends_on.length) lines.push(`depends_on: ${s.depends_on.join(", ")}`);
  if (s.wait_for) lines.push(`ready_when: ${s.wait_for}`);
  return `\x1b[38;5;110m${lines.map(x => "# " + x).join("\r\n")}\x1b[0m\r\n\r\n`;
}

function renderSessions() {
  const list = document.getElementById("session-list");
  list.innerHTML = "";
  const sessions = Object.values(state.sessions);
  document.getElementById("session-count").textContent = sessions.length;
  sessions.forEach(s => {
    const row = document.createElement("div");
    row.className = "sess" + (state.current === s.id ? " active" : "");
    row.innerHTML = `<div class="sess-main">
        <span class="dot ${statusColor(s.status)}"></span>
        <span class="name">${esc(s.name)}</span>
        <span class="status">${esc(statusText(s.status))}</span>
      </div>
      <div class="sess-command">${esc(s.script || s.command || "")}</div>
      <div class="sess-foot">
        <span>${esc(s.id)}</span>
        <span>${state.unread[s.id] ? "+" + state.unread[s.id] + " 输出" : (s.exit_code === null || s.exit_code === undefined ? "" : "exit " + esc(s.exit_code))}</span>
      </div>
      <span class="ops">
        <button title="重启" data-act="restart" data-sid="${s.id}">↻</button>
        <button title="停止" data-act="stop" data-sid="${s.id}">■</button>
      </span>`;
    row.onclick = () => selectSession(s.id);
    row.querySelector("[data-act=stop]").onclick = (e) => { e.stopPropagation(); doStop(s.id); };
    row.querySelector("[data-act=restart]").onclick = (e) => { e.stopPropagation(); doRestart(s.id); };
    list.appendChild(row);
  });
  renderCurrent();
}

/* ---------------- xterm 终端 ---------------- */
function getTerm(sid) {
  if (!state.terminals[sid]) {
    const term = new Terminal({ convertEol: true, fontSize: 13, fontFamily: "monospace" });
    const fit = new FitAddon.FitAddon();
    term.loadAddon(fit);
    const el = document.createElement("div");
    el.className = "terminal";
    el.dataset.sid = sid;
    el.style.display = state.current === sid ? "" : "none";
    document.getElementById("terminals").appendChild(el);
    term.open(el);
    fit.fit();
    term.onData(d => {  // 终端输入转发到会话 stdin
      if (state.current === sid) {
        api(`/api/sessions/${sid}/input`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: d }),
        }).catch(() => {});
      }
    });
    state.terminals[sid] = { term, fit, fresh: true };
  }
  return state.terminals[sid];
}

function resetTermWithSnapshot(s) {
  const t = getTerm(s.id);
  t.term.reset();
  t.term.write(sessionBanner(s));
  if (s.tail) t.term.write(s.tail);
  t.fresh = false;
}

function selectSession(sid) {
  state.current = sid;
  state.unread[sid] = 0;
  document.querySelectorAll(".terminal").forEach(e => {
    e.style.display = e.dataset.sid === sid ? "" : "none";
  });
  getTerm(sid);
  renderSessions();
  setTimeout(onResize, 0);
}

function writeOutput(sid, data) {
  const t = getTerm(sid);
  t.term.write(data);
  if (state.current !== sid) {
    state.unread[sid] = (state.unread[sid] || 0) + 1;
    renderSessions();
  }
}

function onResize() {
  Object.values(state.terminals).forEach(t => {
    try { t.fit.fit(); } catch (e) {}
  });
}
window.addEventListener("resize", onResize);

/* ---------------- 聊天 ---------------- */
function appendChat(m) {
  const log = document.getElementById("chat-log");
  const div = document.createElement("div");
  let cls = "msg kind-" + (m.kind || "status");
  if (m.status === "done") cls += " status-done";
  if (m.status === "failed") cls += " status-failed";
  const label = { status: "sys", text: "agent", tool_call: "tool", tool_result: "result" }[m.kind] || m.kind;
  div.className = cls;
  div.innerHTML = `<span class="tag">[${esc(m.task_id || "?")}][${label}]</span>${esc(m.message)}`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function renderCurrent() {
  const s = state.sessions[state.current];
  const setText = (id, text) => { document.getElementById(id).textContent = text; };
  const restart = document.getElementById("btn-current-restart");
  const stop = document.getElementById("btn-current-stop");
  restart.disabled = !s;
  stop.disabled = !s;
  if (!s) {
    setText("current-name", "未选择 session");
    setText("current-meta", "启动 profile 后查看命令与输出");
    setText("terminal-session", "-");
    ["current-command", "current-workdir", "current-venv", "current-depends", "current-ready"].forEach(id => setText(id, "-"));
    return;
  }
  const exit = s.exit_code === null || s.exit_code === undefined ? "" : ` · exit ${s.exit_code}`;
  setText("current-name", s.name || s.id);
  setText("current-meta", `${s.id} · ${statusText(s.status)}${exit}`);
  setText("terminal-session", s.id);
  setText("current-command", fmt(s.script || s.command));
  setText("current-workdir", fmt(s.workdir));
  setText("current-venv", fmt(s.venv));
  setText("current-depends", fmt(s.depends_on));
  setText("current-ready", fmt(s.wait_for));
}

async function sendChat() {
  const box = document.getElementById("chat-box");
  const text = box.value.trim();
  if (!text) return;
  box.value = "";
  appendChat({ task_id: "你", kind: "text", status: "working", message: text });
  try {
    const r = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!r.ok) appendChat({ task_id: "sys", kind: "status", status: "failed", message: "发送失败: " + r.error });
    else appendChat({ task_id: r.task_id, kind: "status", status: "working", message: "已入队,等待 agent 响应…" });
  } catch (e) {
    appendChat({ task_id: "sys", kind: "status", status: "failed", message: "发送失败: " + e });
  }
}

/* ---------------- 工具栏动作 ---------------- */
async function refreshProfiles() {
  const r = await api("/api/profiles");
  state.profiles = r.profiles || [];
  state.activeProfile = r.active;
  const sel = document.getElementById("profile-select");
  sel.innerHTML = "";
  state.profiles.forEach(p => {
    const o = document.createElement("option");
    o.value = p; o.textContent = p;
    sel.appendChild(o);
  });
  if (state.activeProfile) sel.value = state.activeProfile;
}

// 初次加载:写入各会话 tail 快照,并记录基线 out_seq
async function refreshSessions() {
  const r = await api("/api/sessions");
  const s = r.sessions || [];
  state.sessions = {};
  s.forEach(x => { state.sessions[x.id] = x; });
  Object.keys(state.terminals).forEach(sid => {
    if (!state.sessions[sid]) {
      const el = document.querySelector(`.terminal[data-sid="${sid}"]`);
      if (el) el.remove();
      delete state.terminals[sid];
      delete state.outSeq[sid];
      delete state.unread[sid];
    }
  });
  s.forEach(x => {
    const t = getTerm(x.id);
    if (t.fresh) resetTermWithSnapshot(x);
    if (state.outSeq[x.id] === undefined) state.outSeq[x.id] = x.out_seq || 0;
  });
  if (state.current && !state.sessions[state.current]) state.current = null;
  if (!state.current && s.length) selectSession(s[0].id);
  renderSessions();
}

// WS 断连时的轮询兜底:状态 + 增量输出
async function pollSessions() {
  let list;
  try {
    const r = await api("/api/sessions");
    list = r.sessions || [];
  } catch (e) { return; }
  const nextSessions = {};
  list.forEach(x => {
    nextSessions[x.id] = x;
    state.sessions[x.id] = x;
    // 重启后 out_seq 归零:清掉旧终端,从 0 重新累积
    if (state.outSeq[x.id] !== undefined && x.out_seq < state.outSeq[x.id]) {
      delete state.outSeq[x.id];
      if (state.terminals[x.id]) state.terminals[x.id].term.reset();
    }
    const after = state.outSeq[x.id] === undefined ? 0 : state.outSeq[x.id];
    api(`/api/sessions/${x.id}/out?after=${after}`).then(res => {
      if (res && res.ok) {
        if (res.data) writeOutput(x.id, res.data);
        state.outSeq[x.id] = res.seq;
      }
    }).catch(() => {});
  });
  Object.keys(state.terminals).forEach(sid => {
    if (!nextSessions[sid]) {
      const el = document.querySelector(`.terminal[data-sid="${sid}"]`);
      if (el) el.remove();
      delete state.terminals[sid];
      delete state.outSeq[sid];
      delete state.unread[sid];
    }
  });
  state.sessions = nextSessions;
  if (state.current && !state.sessions[state.current]) state.current = null;
  if (!state.current && list.length) selectSession(list[0].id);
  renderSessions();
}

function startPolling() {
  if (state.pollTimer) return;
  state.pollTimer = setInterval(pollSessions, 1000);
  setConn("poll");
}

function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

async function doStartProfile() {
  const name = document.getElementById("profile-select").value;
  if (!name) return;
  const r = await api(`/api/start/${name}`, { method: "POST" });
  if (!r.ok) appendChat({ task_id: "sys", kind: "status", status: "failed", message: "启动失败: " + r.error });
  await refreshSessions();
}

async function doStopAll() {
  await Promise.all(Object.keys(state.sessions).map(sid =>
    api(`/api/sessions/${sid}/stop`, { method: "POST" }).catch(() => {})));
  await refreshSessions();
}

async function doStop(sid) { await api(`/api/sessions/${sid}/stop`, { method: "POST" }); }
async function doRestart(sid) { await api(`/api/sessions/${sid}/restart`, { method: "POST" }); }

/* ---------------- WebSocket ---------------- */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
  ws.onopen = () => { stopPolling(); setConn("on"); };
  ws.onclose = (ev) => {
    if (state.ws !== ws) return;
    setConn("off");
    if (ev.code && ev.code !== 1000) console.error("WS 关闭:", ev.code, ev.reason);
    startPolling();       // WS 挂了 -> 轮询兜底
    setTimeout(connectWS, 1500);
  };
  ws.onerror = () => console.error("WS 连接错误(若反复出现,可能是代理拦截了 ws:// 或 CDN 未加载)");
  ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (m.type === "session_output") {
      writeOutput(m.id, m.data);
      if (typeof m.seq === "number") state.outSeq[m.id] = m.seq;
    } else if (m.type === "session_status") {
      if (m.status === "stopped" || m.status === "starting") {
        // 重启后输出从零开始
        delete state.outSeq[m.id];
        state.unread[m.id] = 0;
      }
      state.sessions[m.id] = { ...(state.sessions[m.id] || {}), ...m };
      if (m.status === "starting" && state.terminals[m.id]) {
        resetTermWithSnapshot(state.sessions[m.id]);
      } else if (m.status === "stopped" && state.terminals[m.id]) {
        state.terminals[m.id].term.reset();
      }
      renderSessions();
    } else if (m.type === "chat") appendChat(m);
  };
  if (state.keepTimer) clearInterval(state.keepTimer);
  state.keepTimer = setInterval(() => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send("ping");
  }, 25000);
}

/* ---------------- init ---------------- */
document.getElementById("btn-start").onclick = doStartProfile;
document.getElementById("btn-stopall").onclick = doStopAll;
document.getElementById("btn-current-stop").onclick = () => { if (state.current) doStop(state.current); };
document.getElementById("btn-current-restart").onclick = () => { if (state.current) doRestart(state.current); };
document.getElementById("chat-send").onclick = sendChat;
document.getElementById("chat-box").addEventListener("keydown", e => { if (e.key === "Enter") sendChat(); });

(async () => {
  try { await refreshProfiles(); } catch (e) { console.error(e); }
  try { await refreshSessions(); } catch (e) { console.error(e); }
  connectWS();
})();
