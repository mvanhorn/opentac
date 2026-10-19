"""Render the animations next to the feature list on the Readout boards page (docs/hardware.md).

    uv run python scripts/make_hw_animations.py                  # all three clips
    uv run python scripts/make_hw_animations.py imu              # one clip
    uv run python scripts/make_hw_animations.py --sheets DIR     # also a 12-frame contact sheet per clip
    uv run python scripts/make_hw_animations.py --serve imu      # serve a scene to look at in a browser

Writes into docs/assets/boards/:

    zero_potential.mp4, zero_potential_poster.jpg   a 4 x 4 matrix scanned with the other rows
                                                    left open, then held at 0 V (2D, the network
                                                    is solved for every step)
    imu.mp4, imu_poster.jpg                         a recorded take: the board turned by its
                                                    quaternion, the accelerometer, the tactile frame
    daisy_chain.mp4, daisy_chain_poster.jpg         three 30-pin pairs chained to a Teensy 4.0: close
                                                    up the JST-SH plugs go in, then the whole chain
    usb.jpg                                         a still: both boards with a micro-USB cable in

The scenes are in scripts/hw_anim/ and use the three.js in opentac/cad/web/vendor/three/. Each
clip is drawn frame by frame in headless Chrome (window.renderAt(t)) and encoded with ffmpeg.

Needs kicad-cli (KiCad 10), ffmpeg, Google Chrome, and Node.js with puppeteer-core:

    npm install --prefix ~/.cache/opentac-anim puppeteer-core
    uv run python scripts/make_hw_animations.py --puppeteer ~/.cache/opentac-anim/node_modules/puppeteer-core

The board models need the third-party 3D models in hardware/libs/3dmodels/ (see hardware/README.md;
--models DIR reads them from another folder). The IMU clip reads a recording that is not in this
repository (--recording .../tactile2.npz); only the rendered clip is committed.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = Path(os.environ.get("OPENTAC_SITE_ROOT", ROOT.parent / "opentac.github.io")).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
from make_board_media import BOT30, PCB60, TOP30, footprints, kicad_cli  # noqa: E402

SCENES = ROOT / "scripts" / "hw_anim"
THREE = ROOT / "opentac" / "cad" / "web" / "vendor" / "three"
OUT = SITE_ROOT / "docs" / "assets" / "boards"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

RECORDING = (Path.home() / "Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Arduino/"
             "stm_test/tactile_imu_fast_cfg/recordings/20260911_183119_demo/tactile2.npz")
# The part of the take that is shown: the gripper turns toward a can, grasps it while a hand moves
# the can (the knocks in the accelerometer), and turns back.
IMU_WINDOW = (10.5, 22.5)      # s on the recording's clock (sum of the board's dt_us)
IMU_HISTORY = 4.0              # s of trace already on screen at the first frame
IMU_XFADE = 0.3                # s: the clip's last frames dissolve into the frames just before its start
VIB_HIGHPASS_HZ = 8.0          # vibration = acceleration minus gravity and slow motion

W, H, FPS = 1280, 960, 30
CLIPS = {
    # name: (seconds (None: the scene's own length), poster time, crf); a None crf is a still (JPEG, W x 640)
    "zero_potential": (None, 8.4, 20),
    "imu": (IMU_WINDOW[1] - IMU_WINDOW[0], 7.85, 23),
    "daisy_chain": (None, 9.0, 20),
    "usb": (None, None, None),
}
STILL_H = 640

# Parts the scenes need to know the positions of (KiCad mm), per board.
REFS = {"readout_60pin": ["IC5", "LED1", "J4", "USB2"],
        "readout_30pin_top": ["CN1", "CN2", "LED1", "USB2", "IC5", "J1", "J3"],
        "readout_30pin_bottom": ["J2", "J3", "J4"]}
STACK_GAP_MM = 5.7          # 30-pin pair: board-to-board gap, as in make_board_media.py


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{r.stdout}\n{r.stderr}")
    return r.stdout


# ---------------------------------------------------------------- board meshes

def board_outline(text):
    """Bounding box [x0, y0, x1, y1] of the Edge.Cuts graphics, KiCad mm."""
    from make_board_media import block_end
    pts = []
    for m in re.finditer(r"\(gr_(?:line|arc|rect|circle|poly)\b", text):
        blk = text[m.start():block_end(text, m.start())]
        if '(layer "Edge.Cuts")' in blk:
            pts += [(float(a), float(b)) for a, b in re.findall(r"\((?:start|mid|end|xy) ([-\d.]+) ([-\d.]+)\)", blk)]
    p = np.array(pts)
    return [*p.min(0).round(3).tolist(), *p.max(0).round(3).tolist()]


def export_board(pcb, name, models, work):
    """kicad-cli GLB of one board -> meshes grouped by material (mm, three.js axes: y up) + part positions."""
    import trimesh

    text = pcb.read_text()
    fps = footprints(text)
    outline = board_outline(text)
    # temporary copy: absolute library paths, the third-party models from `models`, no stacked-board model
    text = re.sub(r"\$\{KIPRJMOD\}/(?:\.\./)+libs/3dmodels/", str(models).rstrip("/") + "/", text)
    text = text.replace("${KIPRJMOD}/../../libs", str(ROOT / "hardware" / "libs"))
    text = text.replace("${KIPRJMOD}/../libs", str(ROOT / "hardware" / "libs"))
    from make_board_media import block_end
    while (m := re.search(r'\(model "[^"]*readoutboard_top\.step"', text)):
        s = text.rfind("\n", 0, m.start()) + 1
        text = text[:s] + text[block_end(text, m.start()):].lstrip("\n")
    tmp = work / f"{name}.kicad_pcb"
    tmp.write_text(text)
    shutil.copy(pcb.with_suffix(".kicad_pro"), tmp.with_suffix(".kicad_pro"))
    glb = work / f"{name}.glb"
    run([kicad_cli(), "pcb", "export", "glb", "-f", "--subst-models", "--include-tracks", "--include-pads",
         "--include-zones", "--include-silkscreen", "--include-soldermask", "-o", glb, tmp])

    groups = {}
    for mesh in trimesh.load(glb).dump(concatenate=False):
        if not isinstance(mesh, trimesh.Trimesh):      # outlines (Path3D) carry no faces
            continue
        mat = getattr(mesh.visual, "material", None)
        if mat is None:                                 # plain colours (no glTF material)
            rgba, metal, rough = tuple(int(v) for v in mesh.visual.main_color), 0.0, 0.6
        else:
            rgba = tuple(int(v) for v in mat.baseColorFactor)
            metal = mat.metallicFactor if mat.metallicFactor is not None else 0.0
            rough = mat.roughnessFactor if mat.roughnessFactor is not None else 0.6
        groups.setdefault((rgba, round(metal, 2), round(rough, 2)), []).append(mesh)
    blobs, meta = [], {"groups": [], "parts": {}, "outline": outline}
    off = 0
    for (rgba, metal, rough), meshes in sorted(groups.items(), key=lambda kv: -sum(len(m.faces) for m in kv[1])):
        m = trimesh.util.concatenate(meshes)
        pos = (np.asarray(m.vertices) * 1000).astype("<f4")
        nrm = np.asarray(m.vertex_normals).astype("<f4")
        idx = np.asarray(m.faces).astype("<u4")
        g = {"rgba": rgba, "metal": metal, "rough": rough, "nv": len(pos), "nf": len(idx)}
        for key, arr in (("pos", pos), ("nrm", nrm), ("idx", idx)):
            g[key] = off
            blobs.append(arr.tobytes())
            off += arr.nbytes
        meta["groups"].append(g)
    for ref in REFS[name]:
        _, _, x, y, rot = fps[ref]
        meta["parts"][ref] = [x, y, rot]
    (work / f"{name}.bin").write_bytes(b"".join(blobs))
    (work / f"{name}.json").write_text(json.dumps(meta))
    print(f"  {name}: {sum(g['nf'] for g in meta['groups'])} triangles, {len(meta['groups'])} materials")


def export_boards(models, work):
    if not Path(models).is_dir():
        print(f"warning: no 3D models in {models}: parts without a KiCad library model will be missing")
    export_board(PCB60, "readout_60pin", models, work)
    export_board(TOP30, "readout_30pin_top", models, work)
    export_board(BOT30, "readout_30pin_bottom", models, work)
    (work / "stack.json").write_text(json.dumps({"gap": STACK_GAP_MM}))


# ---------------------------------------------------------------- the IMU take

def export_imu(recording, work):
    z = np.load(recording)
    # The recording's own clock: frames are back to back, each scanned over its dt_us.
    t = np.cumsum(z["dt_us"].astype(float)) / 1e6
    t -= t[0]
    t0, t1 = IMU_WINDOW
    lo, hi = t0 - IMU_HISTORY - IMU_XFADE - 0.5, t1 + 0.5
    sel = (t >= lo) & (t <= hi)
    ts = t[sel] - t0
    quat = z["quat"][sel]                     # w, x, y, z: the sensor's own fusion, sensor axes
    quat *= np.sign(quat[:, :1] + 1e-12)      # one hemisphere, so interpolation takes the short way
    accel = z["accel"][sel] / 1000.0          # mg -> g, axes as recorded
    # Vibration: each axis high-passed (2nd-order Butterworth, run forward and back so nothing shifts
    # in time), over the whole take so the window has no edge effects; then its magnitude.
    from scipy.signal import butter, filtfilt
    fs = 1 / np.median(np.diff(t))
    b, a = butter(2, VIB_HIGHPASS_HZ, "highpass", fs=fs)
    vib = np.linalg.norm(filtfilt(b, a, z["accel"] / 1000.0, axis=0), axis=1)[sel]
    tac = np.clip(z["raw"][sel].astype(float) - z["baseline"][None], 0, None)
    # One tactile frame per video frame: the median of the scans that finished in that 1/30 s.
    f0 = -int(round(IMU_XFADE * FPS))
    n = int(round((t1 - t0) * FPS)) - f0 + 1
    frames = []
    for f in range(f0, f0 + n):
        tf = f / FPS
        k = (ts > tf - 1 / FPS) & (ts <= tf)
        frames.append(np.median(tac[k], 0) if k.any() else tac[np.searchsorted(ts, tf)])
    frames = np.round(np.array(frames)).astype(int)
    out = {
        "t": np.round(ts, 5).tolist(), "quat": np.round(quat, 5).tolist(), "accel": np.round(accel, 4).tolist(),
        "vib": np.round(vib, 4).tolist(), "vib_highpass_hz": VIB_HIGHPASS_HZ,
        "xy": np.round(z["xy"], 3).tolist(), "rc": z["grid_rc"].tolist(),
        "frames": frames.tolist(), "frame0": f0, "fps": FPS, "duration": t1 - t0, "history": IMU_HISTORY,
    }
    (work / "imu.json").write_text(json.dumps(out))
    print(f"  imu: {sel.sum()} samples ({ts[0]:.2f} .. {ts[-1]:.2f} s), {n} tactile frames")


# ---------------------------------------------------------------- serving and capture

class Handler(http.server.SimpleHTTPRequestHandler):
    """/scene/ -> scripts/hw_anim, /three/ -> the vendored three.js, /data/ -> the work folder."""
    work = None

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, base in (("/scene/", SCENES), ("/three/", THREE), ("/data/", self.work)):
            if path.startswith(prefix):
                return str(Path(base) / path[len(prefix):])
        return str(SCENES / "missing")

    def log_message(self, *a):
        pass


def serve(work, port=0):
    Handler.work = work
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def capture(name, port, frames_dir, puppeteer, seconds=None, start=0.0, h=H):
    url = f"http://127.0.0.1:{port}/scene/index.html?clip={name}&capture=1&w={W}&h={h}&dpr=2"
    env = dict(os.environ, PUPPETEER_CORE=str(puppeteer), CHROME=CHROME)
    args = ["node", SCENES / "capture.js", url, frames_dir, FPS]
    if seconds:
        args += [seconds, start]
    subprocess.run([str(a) for a in args], check=True, env=env)


def encode(frames_dir, mp4, crf, pre=0):
    """PNG frames (2x) -> H.264 at W x H. The first `pre` frames come from before the clip's start:
    the clip's last `pre` frames are blended into them, so the loop runs on without a jump."""
    files = sorted(frames_dir.glob("f*.png"))
    if pre:
        head, body = files[:pre], files[pre:]
        for j, f in enumerate(body[-pre:]):
            a = (j + 1) / (pre + 1)                   # weight of the frame from before the start
            Image.blend(Image.open(f).convert("RGB"), Image.open(head[j]).convert("RGB"), a).save(f)
        for f in head:
            f.unlink()
        for i, f in enumerate(body):                  # renumber from f0000
            f.rename(frames_dir / f"f{i:04d}.png")
    run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", FPS, "-i", frames_dir / "f%04d.png",
         "-vf", f"scale={W}:{H}:flags=lanczos", "-c:v", "libx264", "-preset", "slow", "-crf", crf,
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", mp4])


def poster(frames_dir, t, jpg):
    files = sorted(frames_dir.glob("f*.png"))
    f = files[min(len(files) - 1, int(round(t * FPS)))]
    Image.open(f).convert("RGB").resize((W, H), Image.LANCZOS).save(jpg, quality=84, optimize=True, progressive=True)


def contact_sheet(mp4, png, n=12, cols=4, width=480):
    """n frames spread over the clip, with their times, tiled."""
    dur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", mp4]))
    with tempfile.TemporaryDirectory() as td:
        tiles = []
        for i in range(n):
            t = dur * i / n
            f = Path(td) / f"{i}.png"
            run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", mp4, "-frames:v", "1", f])
            im = Image.open(f).convert("RGB")
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
            ImageDraw.Draw(im).text((8, 6), f"{t:5.2f} s", fill=(200, 30, 30), font=ImageFont.load_default(18))
            tiles.append(im)
    rows = (n + cols - 1) // cols
    th = tiles[0].height
    sheet = Image.new("RGB", (cols * width + (cols + 1) * 6, rows * th + (rows + 1) * 6), (255, 255, 255))
    for i, im in enumerate(tiles):
        sheet.paste(im, (6 + (i % cols) * (width + 6), 6 + (i // cols) * (th + 6)))
    sheet.save(png)
    print(f"  {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", default=list(CLIPS))
    ap.add_argument("--recording", type=Path, default=RECORDING, help="tactileN.npz of the IMU take")
    ap.add_argument("--models", type=Path, default=ROOT / "hardware" / "libs" / "3dmodels")
    ap.add_argument("--puppeteer", type=Path, default=Path.home() / ".cache/opentac-anim/node_modules/puppeteer-core")
    ap.add_argument("--work", type=Path, help="keep the meshes, data and frames here (default: a temporary folder)")
    ap.add_argument("--sheets", type=Path, help="write a contact sheet per clip into this folder")
    ap.add_argument("--serve", action="store_true", help="prepare the data, then serve the scenes until Ctrl-C")
    ap.add_argument("--port", type=int, default=0, help="port for --serve (default: any free one)")
    args = ap.parse_args()
    unknown = set(args.clips) - set(CLIPS)
    if unknown:
        sys.exit(f"unknown clip(s): {', '.join(sorted(unknown))}; choose from {', '.join(CLIPS)}")

    with tempfile.TemporaryDirectory() as td:
        work = args.work or Path(td)
        work.mkdir(parents=True, exist_ok=True)
        if {"imu", "daisy_chain", "usb"} & set(args.clips):
            export_boards(args.models, work)
        if "imu" in args.clips:
            export_imu(args.recording, work)
        srv, port = serve(work, args.port if args.serve else 0)
        if args.serve:
            print(f"http://127.0.0.1:{port}/scene/index.html?clip={args.clips[0]}  (Ctrl-C to stop)")
            try:
                threading.Event().wait()
            except KeyboardInterrupt:
                return
        OUT.mkdir(parents=True, exist_ok=True)
        for name in args.clips:
            seconds, t_poster, crf = CLIPS[name]
            frames = work / f"frames_{name}"
            shutil.rmtree(frames, ignore_errors=True)
            frames.mkdir()
            if crf is None:                                   # a still
                capture(name, port, frames, args.puppeteer, 1 / FPS, 0.0, STILL_H)
                jpg = OUT / f"{name}.jpg"
                Image.open(frames / "f0000.png").convert("RGB").resize((W, STILL_H), Image.LANCZOS).save(
                    jpg, quality=86, optimize=True, progressive=True)
                print(f"  {jpg}: {W}x{STILL_H}, {jpg.stat().st_size / 1e3:.0f} kB")
                continue
            pre = round(IMU_XFADE * FPS) if name == "imu" else 0
            capture(name, port, frames, args.puppeteer, seconds and seconds + pre / FPS, -pre / FPS)
            mp4 = OUT / f"{name}.mp4"
            encode(frames, mp4, crf, pre)
            n = len(list(frames.glob("f*.png")))
            jpg = OUT / f"{name}_poster.jpg"
            poster(frames, t_poster, jpg)
            print(f"  {mp4}: {n / FPS:.1f} s, {mp4.stat().st_size / 1e6:.2f} MB; "
                  f"{jpg}")
            if args.sheets:
                args.sheets.mkdir(parents=True, exist_ok=True)
                contact_sheet(mp4, args.sheets / f"{name}_sheet.png")
        srv.shutdown()


if __name__ == "__main__":
    main()
