"""Open RLE patterns in the HTML viewer (viewer/index.html).

    python -m gol.view pattern.rle [--out file.html] [--no-open]

writes a self-contained HTML file with the pattern embedded and opens it in
the default browser.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

VIEWER = Path(__file__).resolve().parent.parent / "viewer" / "index.html"
MARKER = "<!-- EMBED -->"


def _js_string(s: str) -> str:
    """JSON-encode s so it is safe inside an inline <script> element."""
    return json.dumps(s).replace("</", "<\\/").replace("<!--", "<\\!--")


def make_html(rle_text: str, name: str, out_path) -> Path:
    """Write a copy of the viewer with `rle_text` embedded; returns the output path."""
    html = VIEWER.read_text(encoding="utf-8")
    script = (f"<script>window.EMBEDDED_RLE = {_js_string(rle_text)}; "
              f"window.EMBEDDED_NAME = {_js_string(name)};</script>")
    if MARKER in html:
        html = html.replace(MARKER, script, 1)
    else:  # fall back to just before the first script
        i = html.index("<script")
        html = html[:i] + script + "\n" + html[i:]
    out = Path(out_path)
    out.write_text(html, encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gol.view", description=__doc__.split("\n\n")[0])
    ap.add_argument("rle", help="RLE pattern file")
    ap.add_argument("--out", help="output HTML file (default: <rle name>.html next to the RLE)")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)

    src = Path(args.rle)
    out = Path(args.out) if args.out else src.with_suffix(".html")
    make_html(src.read_text(encoding="utf-8"), src.stem, out)
    print(f"wrote {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
