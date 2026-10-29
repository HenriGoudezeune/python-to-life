"""Command line: compile a Python function to a Game of Life pattern.

    python -m gol.compiler examples/add4.py --inputs a=5,b=9 --run --html

Without --inputs only the circuit statistics are printed.
"""
from __future__ import annotations

import argparse
import time
import webbrowser
from pathlib import Path

from . import compile_function, drc
from .frontend import parse_function


def python_reference(src: str, fname: str, prog, values: dict):
    """Run the original Python function and mask the result to the output widths."""
    ns: dict = {f"u{i}": int for i in range(1, 65)}
    exec(compile(src, "<program>", "exec"), ns)
    res = ns[fname](*[values[n] for n, _ in prog.inputs])
    if isinstance(res, tuple):
        return tuple(int(v) & ((1 << w) - 1) for v, (_, w) in zip(res, prog.outputs))
    return int(res) & ((1 << prog.outputs[0][1]) - 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gol.compiler", description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="Python file with the function to compile")
    ap.add_argument("--func", help="function name (default: the last function in the file)")
    ap.add_argument("--inputs", help="input values, e.g. a=5,b=9")
    ap.add_argument("-o", "--out", help="output RLE file (default: out/<func>_<inputs>.rle)")
    ap.add_argument("--html", action="store_true", help="also write a viewer HTML page and open it")
    ap.add_argument("--run", action="store_true", help="simulate the pattern and compare with Python")
    ap.add_argument("--drc", action="store_true", help="run the layout design-rule check")
    args = ap.parse_args(argv)

    src = Path(args.source).read_text(encoding="utf-8")
    fname = args.func or parse_function(src).name
    t = time.time()
    prog = compile_function(src, fname)
    print(f"compiled {fname} in {time.time() - t:.1f}s: {prog.stats()}")
    if args.drc:
        probs = drc.check(prog.layout)
        print("DRC:", "clean" if not probs else f"{len(probs)} problems")
        for p in probs:
            print("  ", p)
    if not args.inputs:
        return
    values = {}
    for part in args.inputs.split(","):
        k, v = part.split("=")
        values[k.strip()] = int(v, 0)
    tag = "_".join(f"{k}{v}" for k, v in values.items())
    out = Path(args.out) if args.out else Path("out") / f"{fname}_{tag}.rle"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(prog.rle(**values), encoding="utf-8")
    print(f"wrote {out} (outputs readable at generation {prog.layout.t_end})")
    if args.run:
        t = time.time()
        r = prog.run(**values)
        got = prog.result(r.outputs)
        want = python_reference(src, fname, prog, values)
        status = "OK" if got == want and r.clean else "MISMATCH"
        print(f"Game of Life: {got}   Python: {want}   [{status}] "
              f"({r.generations} generations, {time.time() - t:.1f}s)")
        for p in r.problems:
            print("  ", p)
    if args.html:
        from ..view import make_html
        h = make_html(out.read_text(encoding="utf-8"), out.stem, out.with_suffix(".html"))
        print(f"wrote {h}")
        webbrowser.open(h.resolve().as_uri())


if __name__ == "__main__":
    main()
