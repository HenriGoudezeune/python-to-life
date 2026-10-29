"""Editor app: write a Python function, compile it to Life and watch it run.

    python -m gol.editor [--port 8765] [--no-open]

starts a local web server (127.0.0.1 only) with the editor page
(editor/index.html) and the viewer (viewer/index.html) side by side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .compiler import CompileError, compile_function
from .compiler.__main__ import python_reference
from .compiler.frontend import parse_function

ROOT = Path(__file__).resolve().parent.parent
PAGES = {"/": ROOT / "editor" / "index.html", "/viewer": ROOT / "viewer" / "index.html"}
EXAMPLES = ROOT / "examples"

_cache: dict = {}            # (sha1 of source, function name) -> Program
_lock = threading.Lock()


def _program(src: str, func: str | None):
    fname = func or parse_function(src).name
    key = (hashlib.sha1(src.encode()).hexdigest(), fname)
    with _lock:
        prog = _cache.get(key)
    if prog is None:
        prog = compile_function(src, fname)
        with _lock:
            if len(_cache) > 32:
                _cache.clear()
            _cache[key] = prog
    return fname, prog


def _values(prog, raw: dict) -> dict:
    vals = {}
    for n, w in prog.inputs:
        v = int(str(raw.get(n, 0) or 0), 0)
        if not 0 <= v < (1 << w):
            raise ValueError(f"input {n} = {v} does not fit in {w} bit(s)")
        vals[n] = v
    return vals


def api_compile(req: dict) -> dict:
    t = time.time()
    fname, prog = _program(req["src"], req.get("func"))
    vals = _values(prog, req.get("inputs", {}))
    t_end = prog.layout.t_end
    # where each output bit's glider is at t_end, so the browser can read the result itself
    probes = [{"name": name, "bit": bit, "const": None if const is None else int(const),
               "cells": None if traj is None else traj.cells_at(t_end).tolist()}
              for name, bit, traj, const in prog.layout.outputs]
    return {"func": fname, "inputs": prog.inputs, "outputs": prog.outputs,
            "values": vals, "stats": prog.stats(), "t_end": t_end, "probes": probes,
            "python": python_reference(req["src"], fname, prog, vals),
            "rle": prog.rle(**vals), "seconds": round(time.time() - t, 2)}


def api_run(req: dict) -> dict:
    fname, prog = _program(req["src"], req.get("func"))
    vals = _values(prog, req.get("inputs", {}))
    t = time.time()
    r = prog.run(**vals)
    got = prog.result(r.outputs)
    want = python_reference(req["src"], fname, prog, vals)
    return {"life": got, "python": want, "ok": got == want and r.clean, "clean": r.clean,
            "problems": [str(p) for p in r.problems], "generations": r.generations,
            "seconds": round(time.time() - t, 1)}


def api_examples(_req=None) -> dict:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(EXAMPLES.glob("*.py"))}


API = {"/api/compile": api_compile, "/api/run": api_run}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in PAGES:
            self._send(200, PAGES[path].read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/examples":
            self._json(200, api_examples())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        fn = API.get(self.path)
        if fn is None:
            return self._send(404, b"not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._json(200, fn(req))
        except (CompileError, SyntaxError, ValueError) as e:
            msg = f"line {e.lineno}: {e.msg}" if isinstance(e, SyntaxError) else str(e)
            self._json(400, {"error": msg})
        except Exception as e:  # report, keep serving
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gol.editor", description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"editor running at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
