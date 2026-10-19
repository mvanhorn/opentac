"""Render the readout-board pictures and turntable loops for the website.

    uv run python scripts/make_board_media.py              # everything
    uv run python scripts/make_board_media.py readout30    # one board
    uv run python scripts/make_board_media.py --step 10    # quicker draft spin

Re-run it whenever a board in hardware/ changes. Needs kicad-cli (KiCad 10) and
ffmpeg on PATH. Writes into docs/assets/boards/:

    readout60.png, readout60_spin.mp4                    60-pin board
    readout30.png, readout30_exploded.png, readout30_spin.mp4   30-pin pair, stacked
    readout60_poster.jpg, readout30_poster.jpg           first frame of each loop, its
                                                         poster on the website

The 30-pin pair is shown stacked. The top board is exported to STEP, recoloured to
match KiCad's own board colours, and placed on the bottom board's socket J2 in a
temporary copy of the bottom board. The files in hardware/ are not changed.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = Path(os.environ.get("OPENTAC_SITE_ROOT", ROOT.parent / "opentac.github.io")).resolve()
HW = ROOT / "hardware"
OUT = SITE_ROOT / "docs" / "assets" / "boards"
PCB60 = HW / "readout_60pin" / "readout_60pin.kicad_pcb"
TOP30 = HW / "readout_30pin" / "top" / "readout_30pin_top.kicad_pcb"
BOT30 = HW / "readout_30pin" / "bottom" / "readout_30pin_bottom.kicad_pcb"

STACK_GAP_MM = 5.7          # board-to-board gap, as set for the stacked model in the bottom board file
EXPLODE_GAP_MM = 20.0
VIDEO_BG = (128, 130, 134)  # neutral mid grey: reads on light and dark pages
VIDEO_W = 1080
FPS = 20

# KiCad's STEP export colours for the board layers -> the colours its 3D renderer uses,
# so the STEP-placed top board looks like the natively rendered bottom board.
STEP_RECOLOUR = {
    (0.3133, 0.4845, 0.4100): (0.06, 0.17, 0.11),   # solder mask
    (0.8543, 0.8036, 0.0): (0.30, 0.42, 0.22),      # copper under the mask
    (0.6800, 0.7014, 0.5749): (0.24, 0.20, 0.13),   # board body (FR4 edge)
}


def kicad_cli():
    exe = shutil.which("kicad-cli") or "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"
    if not Path(exe).exists():
        sys.exit("kicad-cli not found")
    return exe


# KiCad's "high" render quality adds a floor with a long drop shadow. Use our own
# settings instead (full ray tracing, parts still shadow the board, no floor), in a
# throwaway config folder so the user's own 3D viewer settings don't matter.
RENDER_SETTINGS = {"render": {
    "raytrace_backfloor": False, "raytrace_shadows": True, "raytrace_reflections": True,
    "raytrace_refractions": True, "raytrace_post_processing": True,
    "raytrace_anti_aliasing": True, "raytrace_procedural_textures": True}}
ENV = dict(os.environ)


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def render(pcb, png, rotate, w, h, zoom, pivot_cm=0.0):
    run([kicad_cli(), "pcb", "render", "-o", png, "--quality", "user", "--perspective",
         "--background", "transparent", "--rotate", rotate, "--zoom", zoom,
         "--pivot", f"0,0,{pivot_cm:g}", "-w", w, "-h", h, pcb])


# ---------- tiny S-expression helpers (just enough for footprint positions) ----------

def block_end(text, start):
    """Index just past the ')' that closes the '(' at text[start]."""
    depth, i = 0, start
    while True:
        c = text[i]
        if c == '"':
            i += 1
            while text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1


def footprints(text):
    """{reference: (start, end, x, y, rot)} for every footprint in a .kicad_pcb."""
    out = {}
    for m in re.finditer(r'\n\t\(footprint "', text):
        s = m.start() + 2
        e = block_end(text, s)
        blk = text[s:e]
        ref = re.search(r'\(property "Reference" "([^"]+)"', blk).group(1)
        at = re.search(r'\n\t\t\(at ([-\d.]+) ([-\d.]+)(?: ([-\d.]+))?\)', blk)
        out[ref] = (s, e, float(at.group(1)), float(at.group(2)), float(at.group(3) or 0))
    return out


def mid(fps, a, b):
    return ((fps[a][2] + fps[b][2]) / 2, (fps[a][3] + fps[b][3]) / 2)


# ---------- the 30-pin stack ----------

def stack_boards(tmp, gaps):
    """Write temporary copies of the bottom board with the top board's STEP sitting on
    its sockets, one per gap. Returns {gap: path}."""
    top_txt, bot_txt = TOP30.read_text(), BOT30.read_text()
    top_fp, bot_fp = footprints(top_txt), footprints(bot_txt)
    # header rows on the top board (J1, J3) sit on the socket rows of the bottom board (J2, J3)
    tx, ty = mid(top_fp, "J1", "J3")
    bx, by = mid(bot_fp, "J2", "J3")

    step = tmp / "top.step"
    run([kicad_cli(), "pcb", "export", "step", "-f", "--subst-models",
         "--user-origin", f"{tx:.4f}x{ty:.4f}mm",
         "--include-tracks", "--include-pads", "--include-zones",
         "--include-silkscreen", "--include-soldermask", "-o", step, TOP30])
    found = set()

    def recolour(m):
        rgb = tuple(float(v) for v in m.group(1).split(","))
        for old, new in STEP_RECOLOUR.items():
            if all(abs(a - b) < 2e-3 for a, b in zip(rgb, old)):
                found.add(old)
                return "COLOUR_RGB(''," + ",".join(f"{v:.3f}" for v in new) + ")"
        return m.group(0)
    txt = re.sub(r"COLOUR_RGB\('',([-\d.Ee+,]+)\)", recolour, step.read_text(encoding="latin-1"))
    if len(found) < len(STEP_RECOLOUR):
        print("warning: STEP colours not all found; the top board may look lighter than the bottom")
    step.write_text(txt, encoding="latin-1")

    # model offset is in J2's own frame: undo J2's rotation, KiCad y points down
    _, _, jx, jy, jrot = bot_fp["J2"]
    dx, dy = bx - jx, -(by - jy)
    a = math.radians(-jrot)
    ox, oy = dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a)

    base = bot_txt.replace("${KIPRJMOD}/../../libs", str(HW / "libs"))
    # drop any stacked-board model the design file already carries
    while (m := re.search(r'\(model "[^"]*readoutboard_top\.step"', base)):
        s = base.rfind("\n", 0, m.start()) + 1
        base = base[:s] + base[block_end(base, m.start()):].lstrip("\n")
    out = {}
    for gap in gaps:
        s, e, *_ = footprints(base)["J2"]
        model = (f'\t\t(model "{step}"\n\t\t\t(offset\n\t\t\t\t(xyz {ox:.4f} {oy:.4f} {gap})\n\t\t\t)\n'
                 f'\t\t\t(scale\n\t\t\t\t(xyz 1 1 1)\n\t\t\t)\n\t\t\t(rotate\n\t\t\t\t(xyz 0 0 {jrot:g})\n\t\t\t)\n\t\t)\n\t')
        txt = base[:e - 1].rstrip() + "\n" + model + base[e - 1:]
        p = tmp / f"stack_{gap:g}.kicad_pcb"
        p.write_text(txt)
        shutil.copy(BOT30.with_suffix(".kicad_pro"), p.with_suffix(".kicad_pro"))
        out[gap] = p
    return out


# ---------- framing ----------

def content_box(img, thresh=8):
    return img.getchannel("A").point(lambda a: 255 if a > thresh else 0).getbbox()


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def pad_box(box, size, margin):
    x0, y0, x1, y1 = box
    m = int(margin * max(x1 - x0, y1 - y0))
    if x0 - m < 0 or y0 - m < 0 or x1 + m > size[0] or y1 + m > size[1]:
        print("warning: board touches the frame edge; lower the zoom")
    return (max(0, x0 - m), max(0, y0 - m), min(size[0], x1 + m), min(size[1], y1 + m))


def still(pcb, png, rotate, zoom, tmp, pivot_cm=0.0, long_side=1600):
    raw = tmp / ("raw_" + png.name)
    render(pcb, raw, rotate, 2400, 1800, zoom, pivot_cm)
    img = Image.open(raw).convert("RGBA")
    img = img.crop(pad_box(content_box(img), img.size, 0.04))
    k = long_side / max(img.size)
    img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    img.save(png, optimize=True)
    print(f"  {png}  {img.width}x{img.height}")


def spin(pcb, mp4, tilt, start, step, zoom, tmp, pivot_cm=0.0):
    frames = tmp / ("frames_" + mp4.stem)
    frames.mkdir()
    n = round(360 / step)
    for i in range(n):
        render(pcb, frames / f"{i:03d}.png", f"{tilt},0,{start + i * step:g}", 1600, 1200, zoom, pivot_cm)
        print(f"\r  {mp4.name}: frame {i + 1}/{n}", end="", flush=True)
    print()
    imgs = [Image.open(frames / f"{i:03d}.png").convert("RGBA") for i in range(n)]
    box = pad_box(union([content_box(im) for im in imgs]), imgs[0].size, 0.05)
    w = VIDEO_W
    h = round(w * (box[3] - box[1]) / (box[2] - box[0]) / 2) * 2
    for i, im in enumerate(imgs):
        bg = Image.new("RGBA", im.size, VIDEO_BG + (255,))
        bg.alpha_composite(im)
        bg.crop(box).convert("RGB").resize((w, h), Image.LANCZOS).save(frames / f"c{i:03d}.png")
    poster = mp4.with_name(mp4.stem.replace("_spin", "_poster") + ".jpg")
    Image.open(frames / "c000.png").save(poster, quality=84, optimize=True, progressive=True)
    run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", FPS, "-i", frames / "c%03d.png",
         "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", mp4])
    print(f"  {mp4}  {w}x{h}, {n} frames, {n / FPS:.1f} s, "
          f"{mp4.stat().st_size / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boards", nargs="*", default=["readout60", "readout30"])
    ap.add_argument("--step", type=float, default=5, help="turntable step in degrees (default 5)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "kicad" / "10.0").mkdir(parents=True)
        (tmp / "kicad" / "10.0" / "3d_viewer.json").write_text(json.dumps(RENDER_SETTINGS))
        ENV["KICAD_CONFIG_HOME"] = str(tmp / "kicad")
        if "readout60" in args.boards:
            still(PCB60, OUT / "readout60.png", "-45,0,45", 0.6, tmp)
            spin(PCB60, OUT / "readout60_spin.mp4", -45, 45, args.step, 0.6, tmp)
        if "readout30" in args.boards:
            stacks = stack_boards(tmp, [STACK_GAP_MM, EXPLODE_GAP_MM])
            # pivot: aim the camera at the middle of the stack, not at the bottom board
            still(stacks[STACK_GAP_MM], OUT / "readout30.png", "-55,0,30", 0.6, tmp, pivot_cm=0.6)
            still(stacks[EXPLODE_GAP_MM], OUT / "readout30_exploded.png", "-60,0,30", 0.45, tmp,
                  pivot_cm=1.3)
            spin(stacks[STACK_GAP_MM], OUT / "readout30_spin.mp4", -55, 30, args.step, 0.6, tmp,
                 pivot_cm=0.6)


if __name__ == "__main__":
    main()
