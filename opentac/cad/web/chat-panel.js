// "Describe" mode: say what you want in plain English and the assistant
// designs it, then hands the result straight into this editor.
//
// The design itself is never invented by the model. The backend runs the real
// OpenTactileCAD pipeline against the OpenTactileCAD Skill (SKILL.md as system
// instructions, outline/preflight/generate as function-calling tools), so what
// comes back is a board that was actually built and DRC-checked, with a real
// taxel count.
//
// The finished board is loaded through the same "otc:load-example" event the
// example gallery and the URL handoff use, so a described design is editable
// exactly like a hand-drawn one and there is no third code path to keep in
// sync.
//
// Chat is optional on the server: without a GEMINI_API_KEY the routes report
// unavailable and this file hides the mode entirely, rather than leaving a
// button that fails when pressed.

import { API_BASE } from "./config.js";

const CHAT_API = `${API_BASE}/api/chat`;

const log = document.getElementById("chat-log");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("chat-send");
const resetBtn = document.getElementById("chat-reset");
const status = document.getElementById("chat-status");

function hideChatMode() {
  document.getElementById("mode-btn-chat")?.remove();
  document.querySelector('.mode-card__cta[data-mode="chat"]')
    ?.closest(".mode-card")?.remove();
  document.getElementById("chat-sidebar-group")?.remove();
}

if (form) {
  const SESSION = (() => {
    try {
      let s = localStorage.getItem("otc_chat_session");
      if (!s) {
        s = "s" + Math.random().toString(36).slice(2, 9);
        localStorage.setItem("otc_chat_session", s);
      }
      return s;
    } catch {
      return "s" + Math.random().toString(36).slice(2, 9);
    }
  })();

  function bubble(who, text, cls) {
    const el = document.createElement("div");
    el.className = "chat-msg " + (cls || "");
    const w = document.createElement("div");
    w.className = "chat-msg__who";
    w.textContent = who;
    const b = document.createElement("div");
    b.className = "chat-msg__body";
    b.textContent = text;
    el.append(w, b);
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }

  function toolTrace(node, calls) {
    if (!calls?.length) return;
    const det = document.createElement("details");
    det.className = "chat-tools";
    const sum = document.createElement("summary");
    sum.textContent = calls.map((c) => c.name.replace(/_/g, " ")).join(" → ");
    const pre = document.createElement("pre");
    pre.textContent = calls
      .map((c) => `${c.name}\n${JSON.stringify(c.args, null, 1)}`).join("\n\n");
    det.append(sum, pre);
    node.appendChild(det);
  }

  // Same event the example gallery fires, so everything downstream — canvas,
  // parameter fields, Generate, downloads — behaves as it does for a
  // hand-drawn board. autoGenerate re-runs the pipeline here so the editor
  // holds a live result it can edit, rather than a picture of one.
  function loadDesign(design) {
    window.dispatchEvent(new CustomEvent("otc:load-example", {
      detail: {
        outline: design.outline,
        params: design.params,
        label: design.name,
        requireCableEdge: false,
        autoGenerate: true,
      },
    }));
  }

  // A design turn runs 10-60s behind one request. Poll the server for what it
  // is actually doing so the wait shows the pipeline's real steps and numbers
  // instead of a motionless "thinking…", which reads as a hang.
  function trackProgress(node) {
    let stopped = false;
    const body = node.querySelector(".chat-msg__body");
    const tick = async () => {
      if (stopped) return;
      try {
        const r = await fetch(
          `${CHAT_API}/progress?session_id=${encodeURIComponent(SESSION)}`);
        const d = await r.json();
        if (!stopped && d.steps?.length) body.textContent = d.steps.join("\n");
      } catch { /* a dropped poll is not worth surfacing; the POST decides */ }
      if (!stopped) setTimeout(tick, 1000);
    };
    tick();
    return () => { stopped = true; };
  }

  async function send(text) {
    bubble("you", text, "chat-msg--user");
    input.value = "";
    sendBtn.disabled = true;
    status.textContent = "designing… a board takes 10-40s";
    const pending = bubble("assistant", "Thinking…");
    const stopProgress = trackProgress(pending);

    let data;
    try {
      const r = await fetch(CHAT_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION, message: text }),
      });
      data = await r.json();
    } catch (err) {
      stopProgress();
      pending.remove();
      bubble("error", `Could not reach the design assistant. ${err}`,
             "chat-msg--error");
      sendBtn.disabled = false;
      status.textContent = "";
      return;
    }

    stopProgress();
    pending.remove();
    if (data.error) {
      bubble("error", data.error, "chat-msg--error");
      status.textContent = "";
    } else {
      const node = bubble("assistant", data.text || "(no reply)");
      toolTrace(node, data.tool_calls);
      if (data.design) {
        loadDesign(data.design);
        status.textContent = `${data.design.name} loaded into the editor`;
      } else {
        status.textContent = "";
      }
    }
    sendBtn.disabled = false;
    input.focus();
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (text) send(text);
  });

  resetBtn?.addEventListener("click", async () => {
    try {
      await fetch(`${CHAT_API}/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION, message: "" }),
      });
      localStorage.removeItem("otc_chat_session");
    } catch { /* clearing the transcript is still worth doing */ }
    log.innerHTML = "";
    status.textContent = "Started a new conversation.";
  });

  // Advertise the mode only if the server can actually serve it.
  (async () => {
    try {
      const r = await fetch(`${CHAT_API}/status`);
      const d = await r.json();
      if (!d.available) hideChatMode();
    } catch {
      hideChatMode();
    }
  })();
}
