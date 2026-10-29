"""Reaction library built from the simulated catalog (data/reactions.json).

A Reaction is a set of input glider trajectories, output trajectories and
static cells (still lifes) that are consumed or left behind. Reactions can be
transformed by the 8 square symmetries and translated in space and time, and
instantiated so that a canonical input matches a given trajectory.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import gliders as G
from . import patterns as P

CATALOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reactions.json")


def transform_traj(t: G.Traj, sym) -> G.Traj:
    cells = P.transform(t.cells_at(0), sym)
    trajs, rest = G.identify(cells, 0)
    assert len(trajs) == 1 and not len(rest)
    return trajs[0]


@dataclass
class Reaction:
    name: str
    inputs: list
    outputs: list
    statics: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))
    statics_survive: bool = False

    def transformed(self, sym) -> "Reaction":
        return Reaction(self.name, [transform_traj(t, sym) for t in self.inputs],
                        [transform_traj(t, sym) for t in self.outputs],
                        P.transform(self.statics, sym) if len(self.statics) else self.statics,
                        self.statics_survive)

    def shifted(self, dx: int, dy: int, dt: int) -> "Reaction":
        return Reaction(self.name, [t.shifted(dx, dy, dt) for t in self.inputs],
                        [t.shifted(dx, dy, dt) for t in self.outputs],
                        self.statics + np.array([dx, dy]) if len(self.statics) else self.statics,
                        self.statics_survive)

    def dirs(self):
        return tuple(t.dir for t in self.inputs), tuple(t.dir for t in self.outputs)


def theta(t: G.Traj) -> int:
    """Timing invariant: t - 4y for downward gliders, t + 4y for upward ones
    (evaluated at the phase-0 anchor)."""
    sx, sy = P.DIRS[t.dir]
    return t.tref + 4 * sx * sy * t.lane


def _load():
    with open(CATALOG) as f:
        d = json.load(f)
    pairs, singles = [], []
    for r in d["glider_pairs"]:
        if r["debris"]:
            continue
        ins = [G.Traj(*x) for x in r["inputs"]]
        outs = [G.Traj(*x) for x in r["outputs"]]
        pairs.append(Reaction("K" if not outs else "KB" if r["kind"] == "90" else "HB", ins, outs))
    for r in d["glider_still_life"]:
        if r["debris"] and not r["sl_survives"]:
            continue
        ins = [G.Traj(*x) for x in r["inputs"]]
        outs = [G.Traj(*x) for x in r["outputs"]]
        sl = np.array(r["sl_cells"], np.int64)
        if r["sl_survives"]:
            pairs  # eater: glider absorbed, still life intact
            singles.append(Reaction(f"eat:{r['still_life']}", ins, [], sl, True))
        elif outs:
            singles.append(Reaction(f"reflect:{r['still_life']}", ins, outs, sl, False))
    return pairs, singles


PAIRS, SINGLES = _load()


def all_syms(rs):
    out = []
    for r in rs:
        for s in P.SYMMETRIES:
            out.append(r.transformed(s))
    return out


def reflectors(in_dir: str, out_dir: str, kinds=("boat",)) -> list[Reaction]:
    res = []
    for r in all_syms([s for s in SINGLES if s.name.startswith("reflect:") and s.name.split(":")[1] in kinds]):
        if r.dirs() == ((in_dir,), (out_dir,)):
            res.append(r)
    return _dedupe(res)


def eaters(in_dir: str, kinds=("eater1",)) -> list[Reaction]:
    res = []
    for r in all_syms([s for s in SINGLES if s.name.startswith("eat:") and s.name.split(":")[1] in kinds]):
        if r.dirs() == ((in_dir,), ()):
            res.append(r)
    return _dedupe(res)


def _dedupe(rs):
    """Drop reactions that are time/space translates of each other."""
    seen, out = set(), []
    for r in rs:
        # normalise: shift so that input glider is on lane 0 at tref 0 by moving along x
        t = r.inputs[0]
        sx, sy = P.DIRS[t.dir]
        # translate by (0, dy) to zero the lane, then dt to zero tref
        dy = sx * t.lane
        r2 = r.shifted(0, dy, 0)
        r2 = r2.shifted(0, 0, -r2.inputs[0].tref)
        key = (tuple(r2.outputs), tuple(map(tuple, P.normalize(r2.statics).tolist())) if len(r2.statics) else (),
               tuple(map(tuple, (r2.statics.min(0),))) if len(r2.statics) else ())
        if key in seen:
            continue
        seen.add(key)
        out.append(r2)
    return out


def place_single(r: Reaction, incoming: G.Traj, dx: int) -> Reaction:
    """Translate single-input reaction r so its input equals `incoming`.
    `dx` (x shift) chooses the position along the incoming lane."""
    i = r.inputs[0]
    assert i.dir == incoming.dir
    sx, sy = P.DIRS[i.dir]
    dy = sx * (i.lane + sy * dx - incoming.lane)
    dt = incoming.tref - i.tref + 4 * sx * dx
    out = r.shifted(dx, dy, dt)
    assert out.inputs[0] == incoming, (out.inputs[0], incoming)
    return out


def place_pair(r: Reaction, a: G.Traj, b: G.Traj) -> Reaction | None:
    """Translate two-input reaction r so its inputs equal (a, b); None if the
    relative timing of a and b does not fit r."""
    ia, ib = r.inputs
    if (ia.dir, ib.dir) != (a.dir, b.dir):
        if (ib.dir, ia.dir) == (a.dir, b.dir):
            ia, ib = ib, ia
        else:
            return None
    # solve lanes: a.lane = ia.lane + sya*dx - sxa*dy ; b.lane = ib.lane + syb*dx - sxb*dy
    sxa, sya = P.DIRS[ia.dir]
    sxb, syb = P.DIRS[ib.dir]
    m = np.array([[sya, -sxa], [syb, -sxb]], float)
    rhs = np.array([a.lane - ia.lane, b.lane - ib.lane], float)
    if abs(np.linalg.det(m)) < 1e-9:
        return None
    dx, dy = np.linalg.solve(m, rhs)
    if abs(dx - round(dx)) > 1e-9 or abs(dy - round(dy)) > 1e-9:
        return None
    dx, dy = int(round(dx)), int(round(dy))
    dt = a.tref - ia.tref + 4 * sxa * dx
    out = r.shifted(dx, dy, dt)
    if sorted(out.inputs) != sorted([a, b]):
        return None
    return out
