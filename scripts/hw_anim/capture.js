// Frame capture for the hardware animations (run by scripts/make_hw_animations.py).
//
//   PUPPETEER_CORE=<path to puppeteer-core> CHROME=<chrome binary> node capture.js URL OUTDIR FPS [SECONDS [START]]
//
// The page must be loaded with ?capture=1: it then draws only when asked, through
// window.renderAt(t), instead of playing in real time. Frames are written as
// OUTDIR/f0000.png ... at the page's w x h times its dpr, for t = START, START + 1/FPS, ...
const puppeteer = require(process.env.PUPPETEER_CORE || "puppeteer-core");
const fs = require("fs");
const [url, out, fpsArg, secArg, startArg] = process.argv.slice(2);
const fps = +(fpsArg || 30), t0 = +(startArg || 0);
(async () => {
  const browser = await puppeteer.launch({
    executablePath: process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new", args: ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "--font-render-hinting=none"],
  });
  const page = await browser.newPage();
  const u = new URL(url);
  const w = +(u.searchParams.get("w") || 1280), h = +(u.searchParams.get("h") || 960);
  const dpr = +(u.searchParams.get("dpr") || 1);
  await page.setViewport({ width: w, height: h, deviceScaleFactor: dpr });
  page.on("console", m => console.log("page:", m.text()));
  page.on("pageerror", e => { console.log("pageerror:", e.message); process.exitCode = 1; });
  await page.goto(url);
  await page.waitForFunction("window.ready === true", { timeout: 180000 });
  const tEnd = secArg ? +secArg : await page.evaluate("window.T_END");
  fs.mkdirSync(out, { recursive: true });
  const n = Math.round(tEnd * fps);
  for (let i = 0; i < n; i++) {
    await page.evaluate(t => window.renderAt(t), t0 + i / fps);
    await page.screenshot({ path: `${out}/f${String(i).padStart(4, "0")}.png`, clip: { x: 0, y: 0, width: w, height: h } });
    if (i % 60 === 0) console.log(`  frame ${i} / ${n}`);
  }
  await browser.close();
})();
