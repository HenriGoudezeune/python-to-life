"""Brute-force search for clean glider reactions, verified by simulation.

Every candidate collision is simulated (many at once on stacked bit boards)
and its result is decomposed into outgoing gliders plus leftover debris.
A reaction is *clean* when the leftover is empty (or, for still-life targets,
when it is exactly the untouched target, i.e. nothing happened).

Run `python -m gol.search` to (re)build data/reactions.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import gliders as G
from . import patterns as P
from .life import pack, step, unpack

BOARD = 320          # board size per candidate
ORIGIN = BOARD // 2  # candidate coordinates are relative to the board centre
GENS = 300


def simulate_many(configs: list[np.ndarray], gens: int = GENS) -> list[np.ndarray]:
    boards = np.stack([pack(c + ORIGIN, BOARD, BOARD) for c in configs])
    for _ in range(gens):
        boards = step(boards)
    return [unpack(b) - ORIGIN for b in boards]


def outcome(cells: np.ndarray, t: int):
    trajs, rest = G.identify(cells, t)
    return trajs, rest


def _result(inputs, trajs, rest, t):
    return {
        "inputs": [list(map(_j, (x.dir, x.lane, x.tref))) for x in inputs],
        "outputs": [list(map(_j, (x.dir, x.lane, x.tref))) for x in trajs],
        "debris": rest.tolist(),
        "debris_still": G.is_still(rest) if len(rest) else True,
        "t": t,
    }


def _j(v):
    return v.item() if hasattr(v, "item") else v


def glider_pairs():
    """All distinct collisions of a SE glider with a SW (90 deg) or NW (head-on) glider."""
    a = G.Traj("SE", 0, 0)
    cands = []
    for p in range(4):
        for dy in range(-10, 11):
            for bx in (24, 25):  # SW glider coming from the right
                b = G.traj_of("SW", p, bx, dy, 0)
                cands.append(("90", a, b))
        for k in range(-10, 11):
            for s in (40, 41, 42, 43):  # NW glider coming from below-right
                if (s - k) % 2:
                    continue
                bx, by = (s + k) // 2, (s - k) // 2
                b = G.traj_of("NW", p, bx, by, 0)
                cands.append(("180", a, b))
    configs = [np.concatenate([a.cells_at(0), b.cells_at(0)]) for _, a, b in cands]
    results = simulate_many(configs)
    out = []
    seen = set()
    for (kind, a, b), cells in zip(cands, results):
        trajs, rest = outcome(cells, GENS)
        if trajs == sorted([a, b]):
            continue  # no interaction
        r = _result([a, b], trajs, rest, GENS)
        r["kind"] = kind
        key = (kind, tuple(map(tuple, r["outputs"])), len(rest), b)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def glider_still_life():
    """A SE glider hitting each still life (all orientations) on every lane/phase."""
    out = []
    for name, sl in P.STILL_LIFES.items():
        shapes = {}
        for sym in P.SYMMETRIES:
            c = P.normalize(P.transform(sl, sym))
            shapes[frozenset(map(tuple, c.tolist()))] = c
        for orient, (_, c) in enumerate(shapes.items()):
            c = c - c.max(0) // 2  # roughly centred at 0
            lanes = c[:, 0] - c[:, 1]
            cands = []
            for lane in range(int(lanes.min()) - 6, int(lanes.max()) + 7):
                for s in (-40, -39):
                    for p in range(4):
                        # place glider anchor on x - y = lane, x + y ~ s
                        if (s + lane) % 2:
                            continue
                        x, y = (s + lane) // 2, (s - lane) // 2
                        cands.append(G.traj_of("SE", p, x, y, 0))
            configs = [np.concatenate([c, g.cells_at(0)]) for g in cands]
            results = simulate_many(configs)
            for g, cells in zip(cands, results):
                trajs, rest = outcome(cells, GENS)
                same_sl = len(rest) == len(c) and set(map(tuple, rest.tolist())) == set(map(tuple, c.tolist()))
                if same_sl and trajs == [g]:
                    continue  # missed
                r = _result([g], trajs, rest, GENS)
                r["kind"] = "sl"
                r["still_life"] = name
                r["sl_cells"] = c.tolist()
                r["sl_survives"] = bool(same_sl)
                out.append(r)
    return out


def main(path="data/reactions.json"):
    pairs = glider_pairs()
    sls = glider_still_life()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"glider_pairs": pairs, "glider_still_life": sls}, f)
    summarize(pairs, sls)


def summarize(pairs, sls):
    def desc(r):
        o = ",".join(f"{d}" for d, _, _ in r["outputs"]) or "-"
        deb = "clean" if not r["debris"] else ("still" if r["debris_still"] else "messy")
        return o, deb

    from collections import Counter
    cnt = Counter()
    for r in pairs:
        o, deb = desc(r)
        cnt[(r["kind"], o, deb)] += 1
    print("glider+glider outcomes:")
    for k, v in sorted(cnt.items()):
        print("  ", k, v)
    cnt = Counter()
    for r in sls:
        o, deb = desc(r)
        if r["sl_survives"]:
            deb = "SL-survives"
        cnt[(r["still_life"], o, deb)] += 1
    print("glider+still life (clean/interesting):")
    for k, v in sorted(cnt.items()):
        if k[2] in ("clean", "SL-survives") :
            print("  ", k, v)


if __name__ == "__main__":
    main()
