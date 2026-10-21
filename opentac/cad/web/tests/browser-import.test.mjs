// End-to-end check of "open a saved design", driving the REAL page in
// headless Chrome over CDP. No test framework and no browser-automation
// dependency: just Chrome's own debugging protocol over the WebSocket that
// node has had built in since v22.
//
// Run it against a local server, which serves the frontend too:
//
//   uv run opentac cad --port 8123 --no-browser
//   node opentac/cad/web/tests/browser-import.test.mjs
//
// ...or against the deployed site, which also exercises the hosted backend
// and the GitHub Pages sub-path:
//
//   OTC_APP=https://opentac.github.io/cad/ node opentac/cad/web/tests/browser-import.test.mjs
//
//   OTC_APP       the page to drive (default http://127.0.0.1:8123/)
//   OTC_CDP_PORT  Chrome's debug port (default 9333)
//
// It generates its own export in the page rather than reading a fixture, and
// resolves modules and the API relative to the PAGE, so the same script works
// on both. Root-relative paths do not: Pages serves the app under /cad/,
// where "/design-import.js" returns an HTML 404 that lands in JSON.parse.
//
// The import flow is mostly wiring -- a form field written by the wrong DOM
// call, an event nobody listens for, a promise nobody awaits -- and none of
// that shows up in a unit test of either side. Faults such as a ZIP reader
// that returns the wrong entry, or a checkbox set through .value, appear only
// when the page actually runs.
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const APP = process.env.OTC_APP || "http://127.0.0.1:8123/";
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const PORT = Number(process.env.OTC_CDP_PORT || 9333);

const profile = mkdtempSync(join(tmpdir(), "otc-cdp-"));
const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--no-first-run", "--no-default-browser-check", "--disable-gpu",
  "--window-size=1400,1000", "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function target() {
  for (let i = 0; i < 60; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(250);
  }
  throw new Error("chrome never exposed a debug target");
}

let id = 0;
const pending = new Map();
const consoleErrors = [];
const dialogs = [];
let ws;

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const msg = { id: ++id, method, params };
    pending.set(msg.id, { resolve, reject });
    ws.send(JSON.stringify(msg));
  });
}

async function evaluate(expression, awaitPromise = true) {
  const r = await send("Runtime.evaluate", {
    expression, awaitPromise, returnByValue: true, allowUnsafeEvalBlockedByCSP: true,
  });
  if (r.exceptionDetails) {
    throw new Error("page threw: " + (r.exceptionDetails.exception?.description
      || r.exceptionDetails.text));
  }
  return r.result.value;
}

let fails = 0, checks = 0;
function ok(cond, what, extra = "") {
  checks++;
  const mark = cond ? "  ok  " : "  FAIL";
  if (!cond) fails++;
  console.log(`${mark}  ${what}${extra && !cond ? `  <- ${extra}` : ""}`);
}

try {
  ws = new WebSocket(await target());
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) {
      const { resolve, reject } = pending.get(m.id);
      pending.delete(m.id);
      m.error ? reject(new Error(m.error.message)) : resolve(m.result);
    } else if (m.method === "Runtime.exceptionThrown") {
      consoleErrors.push(m.params.exceptionDetails?.exception?.description
        || m.params.exceptionDetails?.text || "unknown");
    } else if (m.method === "Page.javascriptDialogOpening") {
      // The app asks before discarding an outline that is already on the
      // canvas -- correct behaviour, and it blocks a headless page forever.
      dialogs.push(m.params.message);
      send("Page.handleJavaScriptDialog", { accept: true });
    } else if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") {
      consoleErrors.push(m.params.args.map((a) => a.value ?? a.description).join(" "));
    }
  };
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Page.navigate", { url: APP });
  await sleep(2500);

  // ---- the UI exists ----
  ok(await evaluate(`!!document.getElementById("design-import-group")`, false),
     "the Open-a-design panel is in the page");
  ok(await evaluate(`!!document.getElementById("mode-btn-open")`, false),
     "the Open mode button is in the switcher");
  ok(await evaluate(`
    document.getElementById("mode-btn-open").click();
    !document.getElementById("design-import-group").hidden
      && document.getElementById("dxf-import-group").hidden`, false),
     "picking Open shows its panel and hides DXF's");

  // Everything below resolves modules and the API relative to the page, not
  // to the origin root: GitHub Pages serves this app under /cad/, so a
  // root-relative "/design-import.js" returns 404 there and the page gets an
  // HTML error document where it expected JSON. For the same reason the export
  // is generated here rather than read from a fixture file: the fixture is
  // gitignored and does not exist on the deployed site.
  await evaluate(`(async () => {
    window.__otc = {
      mod: await import(new URL("design-import.js", location.href).href),
      api: (await import(new URL("config.js", location.href).href)).API_BASE,
    };
    window.__otc.gen = async (body) => {
      const r = await fetch(window.__otc.api + "/generate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error("generate HTTP " + r.status);
      return r.json();
    };
    window.__otc.zipBytes = (b64) => Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    return true;
  })()`);

  // ---- exact restore from the downloaded ZIP ----
  const restore = await evaluate(`(async () => {
    const made = await window.__otc.gen({
      outline: [[-20,-20],[20,-20],[20,20],[-20,20]], router: "hug",
      board_mode: "expand", pixel_w_mm: 4, pixel_h_mm: 4,
      pitch_x_mm: 4.2, pitch_y_mm: 4.2,
    });
    const bytes = window.__otc.zipBytes(made.zip_b64);
    const dt = new DataTransfer();
    dt.items.add(new File([bytes], "my_sensor.zip", { type: "application/zip" }));
    const input = document.getElementById("design-input");
    input.files = dt.files;
    input.dispatchEvent(new Event("change"));
    const info = document.getElementById("design-import-info");
    for (let i = 0; i < 240; i++) {
      await new Promise(r => setTimeout(r, 250));
      const t = info.textContent || "";
      if (/Reopened|Could not open|failed/i.test(t)) break;
    }
    const svg = document.querySelector("#pcb-preview svg");
    return {
      info: info.textContent.trim(),
      status: document.getElementById("gen-status").textContent.trim(),
      hasSvg: !!svg,
      traces: document.querySelectorAll("#pcb-preview svg *").length,
      downloadEnabled: !document.getElementById("download").disabled,
      stats: document.getElementById("stats").textContent.replace(/\\s+/g, " ").slice(0, 160),
      outlineInfo: (document.getElementById("outline-info")||{}).textContent || "",
      boardMode: (document.querySelector('input[name="board_mode"]:checked')||{}).value,
      pixelW: document.getElementById("pixel_w_mm").value,
      pitchX: document.getElementById("pitch_x_mm").value,
      routeTools: !document.getElementById("route-tools").hidden,
    };
  })()`);
  console.log("    restore ->", JSON.stringify(restore, null, 0).slice(0, 400));
  ok(/Reopened/i.test(restore.info), "the import reports a successful reopen", restore.info);
  ok(restore.hasSvg, "a board preview rendered");
  ok(restore.traces > 20, `the preview has real copper (${restore.traces} elements)`);
  ok(restore.downloadEnabled, "the download button is enabled");
  ok(/Reopened/i.test(restore.status), "the generate status announces the reopen", restore.status);
  ok(restore.boardMode === "expand", "board mode came back as expand", restore.boardMode);
  ok(restore.pixelW === "4" || Number(restore.pixelW) === 4, "taxel width restored", restore.pixelW);
  ok(Number(restore.pitchX) === 4.2, "pitch restored", restore.pitchX);
  ok(restore.routeTools, "the route editor is live on the restored board");

  // ---- recipe path, from a loose manifest.json with non-default settings ----
  const recipe = await evaluate(`(async () => {
    const data = await window.__otc.gen({
      outline: [[-18,-18],[18,-18],[18,18],[-18,18]],
      router: "hug", board_mode: "fixed_keepout",
      pixel_w_mm: 3, pixel_h_mm: 3, pitch_x_mm: 3.4, pitch_y_mm: 3.4,
      cable_length_mm: 15,
    });
    // Pull manifest.json out of the ZIP the same way the app does.
    const bin = window.__otc.zipBytes(data.zip_b64);
    const entry = await window.__otc.mod.readZipEntry(bin.buffer, ["manifest.json"]);
    const dt = new DataTransfer();
    dt.items.add(new File([entry.text], "recipe.json", { type: "application/json" }));
    const input = document.getElementById("design-input");
    input.files = dt.files;
    input.dispatchEvent(new Event("change"));
    const gs = document.getElementById("gen-status");
    for (let i = 0; i < 240; i++) {
      await new Promise(r2 => setTimeout(r2, 250));
      if (/^Done\\.|Error/i.test(gs.textContent.trim())) break;
    }
    return {
      status: gs.textContent.trim(),
      info: document.getElementById("design-import-info").textContent.trim(),
      boardMode: (document.querySelector('input[name="board_mode"]:checked')||{}).value,
      cable: document.getElementById("cable_length_mm").value,
      hasSvg: !!document.querySelector("#pcb-preview svg"),
      originalActive: data.stats.active_pixels,
      statsText: document.getElementById("stats").textContent.replace(/\\s+/g," "),
    };
  })()`);
  console.log("    recipe ->", JSON.stringify(recipe).slice(0, 500));
  ok(recipe.status === "Done.", "the recipe re-generated cleanly", recipe.status);
  ok(recipe.boardMode === "fixed_keepout", "fixed_keepout survived the import", recipe.boardMode);
  ok(Number(recipe.cable) === 15, "cable length survived", recipe.cable);
  ok(recipe.hasSvg, "the recipe rebuilt a board");
  ok(Number((recipe.statsText.match(/Taxels(\d+)/) || [])[1]) === recipe.originalActive,
     `the rebuild has the original taxel count (${recipe.originalActive})`,
     recipe.statsText.slice(0, 120));

  // ---- a pre-design-block manifest opens as a finished board ----
  // A board that was already fabricated, whose owner needs a case for it. The
  // case builder must be enabled; the download must not, because manufacturing
  // files that match the fabricated board cannot be produced from it.
  const legacy = await evaluate(`(async () => {
    const made = await window.__otc.gen({
      outline: [[-20.5,0],[20.5,0],[20.5,41],[-20.5,41]], router: "hug",
      board_mode: "expand", pixel_w_mm: 3, pixel_h_mm: 3,
      pitch_x_mm: 3.2, pitch_y_mm: 3.2,
    });
    const bin = window.__otc.zipBytes(made.zip_b64);
    const entry = await window.__otc.mod.readZipEntry(bin.buffer, ["manifest.json"]);
    // Strip it back to what a version-3 manifest held.
    const man = JSON.parse(entry.text);
    delete man.design; man.v = 3;

    const dt = new DataTransfer();
    dt.items.add(new File([JSON.stringify(man)], "fabricated_board.json",
                          { type: "application/json" }));
    const input = document.getElementById("design-input");
    input.files = dt.files;
    input.dispatchEvent(new Event("change"));
    const info = document.getElementById("design-import-info");
    for (let i = 0; i < 240; i++) {
      await new Promise(r => setTimeout(r, 250));
      if (/finished board|Could not open/i.test(info.textContent)) break;
    }
    return {
      info: info.textContent.trim(),
      status: document.getElementById("gen-status").textContent.trim(),
      taxelsShown: /Taxels/.test(document.getElementById("stats").textContent),
      stats: document.getElementById("stats").textContent.replace(/\\s+/g," ").slice(0, 200),
      hasSvg: !!document.querySelector("#pcb-preview svg"),
      svgTaxels: document.querySelectorAll("#pcb-preview svg rect.pixel").length,
      downloadDisabled: document.getElementById("download").disabled,
      caseDisabled: document.getElementById("case-generate").disabled,
      storedTaxels: made.stats.active_pixels,
    };
  })()`);
  console.log("    legacy ->", JSON.stringify(legacy).slice(0, 460));
  ok(/finished board/i.test(legacy.info), "a legacy manifest opens as a finished board", legacy.info);
  ok(legacy.hasSvg, "the recovered board renders a preview");
  ok(legacy.svgTaxels === legacy.storedTaxels,
     `the preview draws every stored taxel (${legacy.svgTaxels}/${legacy.storedTaxels})`);
  ok(legacy.taxelsShown, "the stats panel describes the recovered board", legacy.stats);
  ok(legacy.downloadDisabled === true,
     "Download ZIP is disabled: matching manufacturing files cannot be produced");
  ok(legacy.caseDisabled === false, "the case builder is enabled", String(legacy.caseDisabled));
  ok(/case|bump/i.test(legacy.status), "the page says what can still be built", legacy.status);

  // ---- a junk file is refused, not swallowed ----
  const junk = await evaluate(`(async () => {
    const dt = new DataTransfer();
    dt.items.add(new File(["{\\"hello\\":1}"], "junk.json", { type: "application/json" }));
    const input = document.getElementById("design-input");
    input.files = dt.files;
    input.dispatchEvent(new Event("change"));
    for (let i = 0; i < 40; i++) {
      await new Promise(r => setTimeout(r, 250));
      if (/Could not open/i.test(document.getElementById("design-import-info").textContent)) break;
    }
    return document.getElementById("design-import-info").textContent.trim();
  })()`);
  ok(/Could not open/i.test(junk), "a junk file is rejected with a message", junk);
  console.log("    junk ->", junk);

  ok(dialogs.length >= 1, "importing over an existing design asks first",
     JSON.stringify(dialogs));
  console.log("    dialogs ->", JSON.stringify(dialogs));
  ok(consoleErrors.length === 0, "no uncaught page errors", consoleErrors.join(" | "));
  if (consoleErrors.length) consoleErrors.forEach((e) => console.log("      " + e));

  console.log(`\n${checks - fails}/${checks} browser checks passed`);
} catch (err) {
  console.log("HARNESS ERROR:", err.message);
  fails++;
} finally {
  try { ws?.close(); } catch {}
  chrome.kill();
}
process.exit(fails ? 1 : 0);
