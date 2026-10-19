from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve()

def edit(name, transform):
    p = root / name
    old = p.read_text()
    new = transform(old)
    if new != old:
        p.write_text(new)
        print(name)

old_url = 'https://github.com/opentac/opentac.github.io'
new_url = 'https://github.com/opentac/opentac'
for name in ['README.md', 'CONTRIBUTING.md', 'CITATION.cff', 'pyproject.toml']:
    edit(name, lambda s: s.replace(old_url, new_url))

edit('README.md', lambda s: s.replace(
    "`--depth 1` leaves out the git history, most of which is old versions of the website's videos:\n`.git` is about 180 MB instead of 320 MB.\n\n", '')
    .replace('| `docs/` | the website |\n', '')
    .replace('maintenance: regenerate the example designs and board media, preview the website locally, check the QR decoder page',
             'maintenance: regenerate the example designs and board media, check the website against the package')
    .replace('## Readout boards\n',
             'The website source is maintained separately in [opentac.github.io](https://github.com/opentac/opentac.github.io).\n\n## Readout boards\n'))

edit('CONTRIBUTING.md', lambda s: s.replace(
    '| `docs/` | the website (MkDocs); `mkdocs.yml` is its table of contents |\n', '')
    .replace('To build the website as CI does, with no warnings allowed:\n\n```bash\nuv run --extra docs mkdocs build --strict -d /tmp/opentac-site\nuv run --extra docs python scripts/preview_site.py    # the site, with OpenTac-CAD at /cad/\n```',
'''The website source is in [opentac.github.io](https://github.com/opentac/opentac.github.io).
For website integration tests or asset-generation scripts, clone it beside this repository,
or set `OPENTAC_SITE_ROOT` to its absolute path. CI checks the revision recorded in
`website-revision.txt`; it does not silently follow the website's latest changes.

```bash
git clone https://github.com/opentac/opentac.github.io ../opentac.github.io
uv run --extra docs python ../opentac.github.io/scripts/preview_site.py
```

The QR integration test is skipped when no website checkout is available. CI always
provides one and runs both the QR and transfer consistency checks. The xampling
checker still needs the external lab sources described in its own docstring.'''))

paths = ['scripts/check_qr_page.py', 'scripts/make_board_media.py',
         'scripts/make_doc_figures.py', 'scripts/make_example_designs.py',
         'scripts/make_hw_animations.py', 'scripts/learn/build_data.py',
         'scripts/learn/check_transfer.py', 'scripts/learn/check_xampling.py']
for name in paths:
    def relocate(s):
        if not re.search(r'^import os$', s, re.M):
            match = re.search(r'^import \w+', s, re.M)
            s = s[:match.start()] + 'import os\n' + s[match.start():]
        match = re.search(r'^(ROOT|REPO) = Path\(__file__\).resolve\(\).parents\[\d\]$', s, re.M)
        var = match.group(1)
        extra = '\nSITE_ROOT = Path(os.environ.get("OPENTAC_SITE_ROOT", ' + var + '.parent / "opentac.github.io")).resolve()'
        s = s[:match.end()] + extra + s[match.end():]
        s = s.replace(var + ' / "docs"', 'SITE_ROOT / "docs"')
        s = s.replace('.relative_to(' + var + ')', '')
        return s
    edit(name, relocate)

edit('scripts/make_doc_screens.cjs', lambda s: s.replace(
    'const OUT = path.join(__dirname, "..", "docs", "assets", "screens");',
    'const SITE_ROOT = process.env.OPENTAC_SITE_ROOT || path.resolve(__dirname, "..", "..", "opentac.github.io");\nconst OUT = path.join(SITE_ROOT, "docs", "assets", "screens");')
    .replace('python scripts/preview_site.py', 'python ../opentac.github.io/scripts/preview_site.py'))

edit('tests/test_qr_page.py', lambda s: s.replace('import shutil\n', 'import os\nimport shutil\n')
    .replace('ROOT = Path(__file__).resolve().parents[1]',
             'ROOT = Path(__file__).resolve().parents[1]\nSITE_ROOT = Path(os.environ.get("OPENTAC_SITE_ROOT", ROOT.parent / "opentac.github.io")).resolve()')
    .replace('def test_the_qr_page_writes_the_same_manifest_as_the_cli():\n',
'''def test_the_qr_page_writes_the_same_manifest_as_the_cli():
    if not (SITE_ROOT / "docs" / "javascripts" / "qr.js").is_file():
        if "OPENTAC_SITE_ROOT" in os.environ:
            pytest.fail("OPENTAC_SITE_ROOT does not contain the website QR decoder")
        pytest.skip("needs the website checkout; set OPENTAC_SITE_ROOT")
'''))

for name in ['scripts/preview_site.py', 'scripts/stamp_cad_assets.py']:
    (root / name).unlink()
(root / 'website-revision.txt').write_text('dba4f9017ceb1c25389b2a56f7cd4de19db424d0\n')
