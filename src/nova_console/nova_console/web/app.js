/* NovaAgent Console 前端:会话侧边栏 + xterm 终端 + 聊天面板。 */
"use strict";

const state = {
  profiles: [],
  activeProfile: null,
  sessions: {},   // sid -> {id,name,status,exit_code,depends_on}
  current: null,  // 当前选中的 sid
  terminals: {},  // sid -> {term, fit}
  ws: null,
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

/* ---------------- 会话侧边栏 ---------------- */
function statusColor(s) {
  return { created: "created", starting: "starting", running: "running",
           ready: "running", exited: "exited", stopped: "stopped", failed: "failed" }[s] || "created";
}

function renderSessions() {
  const list = document.getElementById("session-list");
  list.innerHTML = "";
  Object.values(state.sessions).forEach(s => {
    const row = document.createElement("div");
    row.className = "sess" + (state.current === s.id ? " active" : "");
    row.innerHTML = `<span class="dot ${statusColor(s.status)}"></span>
      <span class="name">${esc(s.name)}</span>
      <span class="ops">
        <button title="重启" data-act="restart" data-sid="${s.id}">↻</button>
        <button title="停止" data-act="stop" data-sid="${s.id}">■</button>
      </span>`;
    row.onclick = () => selectSession(s.id);
    row.querySelector("[data-act=stop]").onclick = (e) => { e.stopPropagation(); doStop(s.id); };
    row.querySelector("[data-act=restart]").onclick = (e) => { e.stopPropagation(); doRestart(s.id); };
    list.appendChild(row);
  });
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
    state.terminals[sid] = { term, fit };
  }
  return state.terminals[sid];
}

function selectSession(sid) {
  state.current = sid;
  document.querySelectorAll(".terminal").forEach(e => {
    e.style.display = e.dataset.sid === sid ? "" : "none";
  });
  getTerm(sid);
  renderSessions();
}

function writeOutput(sid, data) {
  const t = getTerm(sid);
  t.term.write(data);
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
    }
  });
  s.forEach(x => {
    const t = getTerm(x.id);
    if (x.tail) t.term.write(x.tail);
  });
  if (!state.current && s.length) selectSession(s[0].id);
  renderSessions();
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
  state.ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws.onopen = () => document.getElementById("conn").className = "conn on";
  state.ws.onclose = () => {
    document.getElementById("conn").className = "conn off";
    setTimeout(connectWS, 1500);
  };
  state.ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (m.type === "session_output") writeOutput(m.id, m.data);
    else if (m.type === "session_status") {
      if (state.sessions[m.id]) {
        state.sessions[m.id].status = m.status;
        state.sessions[m.id].exit_code = m.exit_code;
      } else {
        state.sessions[m.id] = { id: m.id, name: m.name, status: m.status, exit_code: m.exit_code, depends_on: m.depends_on };
      }
      renderSessions();
    } else if (m.type === "chat") appendChat(m);
  };
  const keep = setInterval(() => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send("ping");
  }, 25000);
}

/* ---------------- init ---------------- */
document.getElementById("btn-start").onclick = doStartProfile;
document.getElementById("btn-stopall").onclick = doStopAll;
document.getElementById("chat-send").onclick = sendChat;
document.getElementById("chat-box").addEventListener("keydown", e => { if (e.key === "Enter") sendChat(); });

(async () => {
  try { await refreshProfiles(); } catch (e) { console.error(e); }
  try { await refreshSessions(); } catch (e) { console.error(e); }
  connectWS();
})();
