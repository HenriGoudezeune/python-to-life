"""Read and write the RLE pattern format (B3/S23 only)."""
from __future__ import annotations

import re

import numpy as np


def parse(text: str) -> tuple[np.ndarray, dict]:
    """Parse RLE text. Returns ((N,2) array of (x, y), meta) with meta holding
    header fields and '#' comment lines. Coordinates come from '#P'/'#R' lines
    if present, else start at (0, 0)."""
    meta: dict = {"comments": []}
    ox = oy = 0
    body = []
    header_seen = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            tag = s[:2]
            if tag in ("#P", "#R"):
                ox, oy = (int(v) for v in s[2:].split()[:2])
            else:
                meta["comments"].append(s)
            continue
        if not header_seen and s.startswith("x"):
            header_seen = True
            for k, v in re.findall(r"(\w+)\s*=\s*([^,]+)", s):
                meta[k] = v.strip()
            rule = meta.get("rule", "B3/S23").upper().replace(" ", "")
            if rule not in ("B3/S23", "23/3"):
                raise ValueError(f"unsupported rule {rule}")
            continue
        body.append(s)
    data = "".join(body)
    xs, ys = [], []
    x = y = 0
    for count, tag in re.findall(r"(\d*)([bo$!]|[A-Za-z])", data):
        n = int(count) if count else 1
        if tag == "!":
            break
        if tag == "$":
            y += n
            x = 0
        elif tag == "b" or tag == ".":
            x += n
        else:  # 'o' or any other live-state letter
            xs.extend(range(x, x + n))
            ys.extend([y] * n)
            x += n
    cells = np.stack([np.array(xs, np.int64) + ox, np.array(ys, np.int64) + oy], axis=1) if xs \
        else np.zeros((0, 2), np.int64)
    return cells, meta


def load(path: str) -> tuple[np.ndarray, dict]:
    with open(path, encoding="utf-8") as f:
        return parse(f.read())


def dumps(cells, comments=(), keep_origin: bool = True, width: int = 70) -> str:
    """Encode cells as RLE. With keep_origin a '#R x y' line records the
    top-left corner so that load(dumps(c)) gives back the same coordinates."""
    cells = np.unique(np.asarray(cells, np.int64).reshape(-1, 2), axis=0)
    lines = [f"#C {c}" if not c.startswith("#") else c for c in comments]
    if not len(cells):
        return "\n".join(lines + ["x = 0, y = 0, rule = B3/S23", "!"]) + "\n"
    x0, y0 = cells.min(0)
    x1, y1 = cells.max(0)
    if keep_origin:
        lines.append(f"#R {x0} {y0}")
    lines.append(f"x = {x1 - x0 + 1}, y = {y1 - y0 + 1}, rule = B3/S23")
    order = np.lexsort((cells[:, 0], cells[:, 1]))
    c = cells[order] - [x0, y0]
    toks: list[str] = []

    def emit(n, t):
        toks.append((str(n) if n > 1 else "") + t)

    cy = 0
    i = 0
    while i < len(c):
        y = c[i, 1]
        if y > cy:
            emit(y - cy, "$")
            cy = y
        cx = 0
        while i < len(c) and c[i, 1] == y:
            x = c[i, 0]
            if x > cx:
                emit(x - cx, "b")
            run = 1
            while i + run < len(c) and c[i + run, 1] == y and c[i + run, 0] == x + run:
                run += 1
            emit(run, "o")
            cx = x + run
            i += run
    toks.append("!")
    out, cur = [], ""
    for t in toks:
        if len(cur) + len(t) > width:
            out.append(cur)
            cur = ""
        cur += t
    out.append(cur)
    return "\n".join(lines + out) + "\n"


def save(path: str, cells, comments=()):
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps(cells, comments))
