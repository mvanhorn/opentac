"""Stamp the CAD tool's own files with the release version at deploy time.

    python scripts/stamp_cad_assets.py site/cad <version>

GitHub Pages lets browsers cache each file for 10 minutes. Right after a deploy
a browser can therefore combine a new app.js with an older cached module it
imports; when the new code needs something only the new module has, the whole
module graph fails to load and the tool does nothing. Stamping every reference
to the tool's own files with ?v=<version> makes each release a separate set of
URLs, so one page load never mixes two releases. Every reference to a module
gets the same URL, so each module still loads once.

Rewrites, in the given folder only (vendor/ untouched, it never changes):
  *.js       from "./x.js" / import "./x.js" / import("./x.js") / fetch("x.json")
  *.html     src="x.js", href="x.css" (relative, own files only)
"""
import re
import sys
from pathlib import Path

root, ver = Path(sys.argv[1]), sys.argv[2]
q = f"?v={ver}"

js_import = re.compile(r'''((?:\bfrom|\bimport)\s*\(?\s*)(["'])(\./[^"'?]+\.js)\2''')
js_fetch = re.compile(r'''(\bfetch\(\s*)(["'])([\w./-]+\.json)\2''')
html_ref = re.compile(r'''(\b(?:src|href)=)(["'])(?![a-z]+:|/|#)([\w./-]+\.(?:js|css))\2''')

changed = 0
for p in sorted(root.glob("*.js")):
    s = p.read_text()
    t = js_import.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{q}{m.group(2)}", s)
    t = js_fetch.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{q}{m.group(2)}", t)
    if t != s:
        p.write_text(t); changed += 1
for p in sorted(root.glob("*.html")):
    s = p.read_text()
    t = html_ref.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{q}{m.group(2)}", s)
    if t != s:
        p.write_text(t); changed += 1

# Every relative module reference must now carry the version, or a module
# could load twice under two URLs.
left = [f"{p.name}: {m.group(3)}" for p in root.glob("*.js") for m in js_import.finditer(p.read_text())
        if "?v=" not in m.group(0)]
if left:
    sys.exit("unstamped imports: " + ", ".join(left))
print(f"stamped {changed} files with {q}")
