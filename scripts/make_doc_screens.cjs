// Screenshots of OpenTac-CAD for the website's design and get-started pages.
//
//   uv run opentac cad --host 127.0.0.1 --port 8231 --no-browser      # in one terminal
//   uv run --extra docs python scripts/preview_site.py --port 8217 --no-browser   # in another
//   NODE_PATH=/path/to/node_modules node scripts/make_doc_screens.cjs
//
// Needs node and puppeteer-core (`npm install puppeteer-core` in any folder, and
// point NODE_PATH at its node_modules), and Google Chrome. It drives the current
// OpenTac-CAD the way a visitor would (the example cards, the robot gallery, the
// case panel) and the website's QR decoder page, and writes WebP screenshots at
// twice the CSS size into docs/assets/screens/. Nothing is edited by hand.
//
//   OT_CAD   the running OpenTac-CAD (default http://127.0.0.1:8231/)
//   OT_SITE  the previewed website, for the QR decoder (default http://127.0.0.1:8217/)
//   CHROME   the Chrome binary
"use strict";
const path = require("path");
const puppeteer = require("puppeteer-core");

const CAD = process.env.OT_CAD || "http://127.0.0.1:8231/";
const SITE = process.env.OT_SITE || "http://127.0.0.1:8217/";
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const OUT = path.join(__dirname, "..", "docs", "assets", "screens");
const QUALITY = 82;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function page(browser, w = 1440, h = 900) {
  const p = await browser.newPage();
  await p.setViewport({ width: w, height: h, deviceScaleFactor: 2 });
  p.on("pageerror", (e) => console.log("page error:", e.message));
  return p;
}

// Screenshot the union of some elements (CSS selectors or handles), with a margin.
async function clip(p, targets, file, pad = 0, inset = null) {
  const boxes = [];
  for (const t of targets) {
    const el = typeof t === "string" ? await p.$(t) : t;
    if (!el) throw new Error("not found: " + t);
    boxes.push(await p.evaluate((e) => {
      const r = e.getBoundingClientRect();
      return { x: r.left + scrollX, y: r.top + scrollY, r: r.right + scrollX, b: r.bottom + scrollY };
    }, el));
  }
  let x = Math.min(...boxes.map((b) => b.x)) - pad, y = Math.min(...boxes.map((b) => b.y)) - pad;
  let r = Math.max(...boxes.map((b) => b.r)) + pad, b = Math.max(...boxes.map((b) => b.b)) + pad;
  if (inset) {                       // [fraction off each side, fraction off top and bottom]
    const dx = (r - x) * inset[0], dy = (b - y) * inset[1];
    x += dx; r -= dx; y += dy; b -= dy;
  }
  await p.mouse.move(1, 1);          // no hover highlight in the picture
  await p.evaluate((y) => window.scrollTo(0, Math.max(0, y - 40)), y);
  await sleep(400);
  const out = path.join(OUT, file);
  await p.screenshot({ path: out, type: "webp", quality: QUALITY, clip: { x, y, width: r - x, height: b - y } });
  console.log("wrote", path.relative(path.join(__dirname, ".."), out));
}

async function byText(p, selector, text) {
  const handles = await p.$$(selector);
  for (const h of handles) if ((await p.evaluate((e) => e.textContent, h)).includes(text)) return h;
  throw new Error(`no ${selector} with "${text}"`);
}

async function landing(p) {
  await p.goto(CAD, { waitUntil: "networkidle2" });
  await sleep(500);
}

async function openExample(p, title) {
  await landing(p);
  await p.evaluate((title) => {
    const card = [...document.querySelectorAll("#landing-examples > *")].find((c) => c.textContent.includes(title));
    [...card.querySelectorAll("button")].find((x) => /Open this design/.test(x.textContent)).click();
  }, title);
  for (let t = 0; t < 180; t++) {
    const s = await p.evaluate(() => document.querySelector("#gen-status").textContent);
    if (/Done/.test(s)) return;
    await sleep(500);
  }
  throw new Error("generate did not finish");
}

async function shots() {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new",
    args: ["--use-angle=metal", "--enable-webgl", "--ignore-gpu-blocklist"] });
  try {
    // 1. The start screen: the robot gallery (first row) and the ways to start.
    let p = await page(browser);
    await landing(p);
    const head = await byText(p, "h3.landing-modes__label", "robot part");
    await p.evaluate((e) => e.scrollIntoView(), head);
    await sleep(2500);                                    // thumbnails load as they come into view
    const cards = (await p.$$("#landing-stl-examples > *")).slice(0, 4);
    await clip(p, [head, ...cards], "cad_robot_gallery.webp", 12);
    const ways = await byText(p, "h3.landing-modes__label", "Other ways to start");
    await clip(p, [ways, ".landing-modes"], "cad_ways_to_start.webp", 12);
    await p.close();

    // 2. A generated example: the result panel, the board preview and the download bar.
    p = await page(browser);
    await openExample(p, "Yubi finger pad");
    const gen = await p.evaluate(() => {
      const g = document.querySelector("#generate").closest(".panel") || document.querySelector("#generate").parentElement;
      g.setAttribute("data-shot", "gen");
      document.querySelector(".download-panel").setAttribute("data-shot", "dl");
      document.querySelector(".download-bar").setAttribute("data-shot", "dlbar");
      document.querySelector("#pcb-preview-panel").setAttribute("data-shot", "pcb");
      return true;
    });
    if (gen) {
      await clip(p, ['[data-shot="gen"]', '[data-shot="pcb"]', '[data-shot="dl"]'], "cad_generated.webp", 8);
      await clip(p, ['[data-shot="dlbar"]'], "cad_download_bar.webp", 2);
    }
    // 3. The printed case: the three lids of the Square patch example.
    await p.close();
    p = await page(browser);
    await openExample(p, "Square patch");
    await p.click("#case-enable");
    await sleep(400);
    await p.addStyleTag({ content: "#case-container { width: 520px !important; height: 400px !important; }" });
    await p.evaluate(() => window.dispatchEvent(new Event("resize")));
    for (const lid of ["flat", "hatch", "bumps"]) {
      await p.evaluate((v) => document.querySelector(`input[name=case-top][value=${v}]`).click(), lid);
      await p.click("#case-generate");
      for (let t = 0; t < 240; t++) {
        if (/^Done/.test(await p.evaluate(() => document.querySelector("#case-status").textContent))) break;
        await sleep(500);
      }
      await p.select("#case-show", "top");
      await sleep(2500);
      await clip(p, ["#case-container"], `cad_case_${lid}.webp`, 0, [0.04, 0.07]);
    }
    await p.close();

    // 4. A surface from the robot gallery: picked on the model, and unfolded flat.
    p = await page(browser);
    await landing(p);
    await p.evaluate(() => {
      const card = [...document.querySelectorAll("#landing-stl-examples > *")].find((c) => c.textContent.includes("AgileX Piper"));
      card.scrollIntoView();
      [...card.querySelectorAll("button")].find((x) => /Use this shape/.test(x.textContent)).click();
    });
    await sleep(8000);
    await clip(p, ["#stl-container"], "cad_stl_surface.webp", 0, [0.08, 0.1]);
    await clip(p, ["#outline-canvas"], "cad_stl_unfolded.webp");
    await p.close();

    // 5. The website's QR decoder on the 9 x 9 patch example's code.
    p = await page(browser, 1100, 900);
    await p.goto(SITE + "qr/#YJ77YN9I2O-QN0AOBT.:VW3NG90", { waitUntil: "networkidle2" });
    await sleep(1500);
    await clip(p, [".ot-qr__fig .ot-qr__drawing"], "qr_decoder_drawing.webp", 6);
    await p.close();
  } finally {
    await browser.close();
  }
}

shots().catch((e) => { console.error(e); process.exit(1); });
