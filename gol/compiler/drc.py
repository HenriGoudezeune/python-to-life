"""Design-rule check: find unintended near-meetings in a layout.

Every glider path piece (segment) moves in a straight line at c/4, so the
closest approach of two segments over their common time window can be found
exactly. Pairs that the builder did not mark as reacting must stay at least
`GAP` cells apart (Chebyshev distance between glider centres); paths must also
keep clear of still lifes they are not meant to hit. This is a fast static
check; the real proof is still the simulation.
"""
from __future__ import annotations

import numpy as np

from .. import patterns as P

GAP = 7.0          # glider centres closer than this may interact
STATIC_GAP = 5.0   # extra clearance around a still life's bounding box


def _motion(traj, xa, xb):
    sx, sy = P.DIRS[traj.dir]
    ta, tb = traj.time_at(xa), traj.time_at(xb)
    if ta > tb:
        ta, tb = tb, ta
    # centre at time t: (1 + sx (t - tref)/4, -sx*lane + 1 + sy (t - tref)/4)
    p0 = np.array([1.0, -sx * traj.lane + 1.0])
    v = np.array([sx, sy]) / 4.0
    return ta, tb, p0 - v * traj.tref, v


def _min_cheb(a0, va, b0, vb, t0, t1):
    """min over t in [t0, t1] of Chebyshev |(a0 + va t) - (b0 + vb t)| and its argmin."""
    r0, rv = a0 - b0, va - vb
    cands = [t0, t1]
    for i in range(2):
        if rv[i]:
            cands.append(-r0[i] / rv[i])
    if rv[0] != rv[1]:
        cands.append(-(r0[0] - r0[1]) / (rv[0] - rv[1]))
    if rv[0] != -rv[1]:
        cands.append(-(r0[0] + r0[1]) / (rv[0] + rv[1]))
    best = None
    for t in cands:
        if t0 <= t <= t1:
            d = np.abs(r0 + rv * t).max()
            if best is None or d < best[0]:
                best = (d, t)
    return best


def check(layout) -> list[str]:
    segs = layout.segs
    mot = [_motion(t, xa, xb) for t, xa, xb, _ in segs]
    problems = []
    for i in range(len(segs)):
        ta, tb, a0, va = mot[i]
        for j in range(i + 1, len(segs)):
            if (i, j) in layout.ok_pairs or (j, i) in layout.ok_pairs:
                continue
            tc, td, b0, vb = mot[j]
            t0, t1 = max(ta, tc), min(tb, td)
            if t0 > t1:
                continue
            d, t = _min_cheb(a0, va, b0, vb, t0, t1)
            if d < GAP:
                p = a0 + va * t
                problems.append(f"gliders {segs[i][3]} / {segs[j][3]} meet at gen {t:.0f} "
                                f"near ({p[0]:.0f},{p[1]:.0f}), distance {d:.1f}")
    for si, (cells, label) in enumerate(layout.statics):
        lo, hi = cells.min(0), cells.max(0)
        c = (lo + hi) / 2
        half = (hi - lo) / 2
        for i, (traj, xa, xb, lab) in enumerate(segs):
            if (i, si) in layout.ok_static:
                continue
            ta, tb, a0, va = mot[i]
            # distance from the path to the box centre, per axis minus half size
            best = None
            for t in (ta, tb, *[(c[k] - a0[k]) / va[k] for k in range(2)]):
                if ta <= t <= tb:
                    d = (np.abs(a0 + va * t - c) - half).max()
                    best = d if best is None else min(best, d)
            # Chebyshev to a box is piecewise linear; also test where the two axes balance
            for sgn in (1, -1):
                den = va[0] - sgn * va[1]
                if den:
                    t = (c[0] - a0[0] - sgn * (c[1] - a0[1])) / den
                    if ta <= t <= tb:
                        d = (np.abs(a0 + va * t - c) - half).max()
                        best = min(best, d) if best is not None else d
            if best is not None and best < STATIC_GAP:
                problems.append(f"glider {lab} passes still life {label} (clearance {best:.1f})")
    return problems
