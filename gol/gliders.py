"""Glider trajectories: identify gliders in a cell set and place them exactly.

A trajectory is (dir, lane, tref):
  * dir  in SE/SW/NE/NW, unit step (sx, sy) per 4 generations;
  * lane = sy*x - sx*y of the phase-0 bounding-box corner (constant along the path);
  * tref = generation at which the glider is in phase 0 with its phase-0
    bounding-box corner at x = 0 (may be negative / in the past).
`cells_at(traj, t)` gives the exact cells at generation t, so a trajectory is a
complete description of one glider for all time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import patterns as P
from .life import pack, step, unpack

# (dir, phase) -> (normalized shape as frozenset, offset of bbox-min relative to glider(d,p,0,0) anchor)
_SHAPES: dict[frozenset, tuple[str, int, tuple[int, int]]] = {}
for _d in P.DIRS:
    for _p in range(4):
        _g = P.glider(_d, _p, 0, 0)
        _m = _g.min(0)
        _SHAPES[frozenset(map(tuple, (_g - _m).tolist()))] = (_d, _p, (int(_m[0]), int(_m[1])))


@dataclass(frozen=True, order=True)
class Traj:
    dir: str
    lane: int
    tref: int

    def __str__(self):
        return f"{self.dir}(lane={self.lane}, t={self.tref})"

    def shifted(self, dx: int = 0, dy: int = 0, dt: int = 0) -> "Traj":
        """Same glider translated by (dx, dy) in space and dt in time."""
        sx, sy = P.DIRS[self.dir]
        return Traj(self.dir, self.lane + sy * dx - sx * dy, self.tref + dt - 4 * sx * dx)

    def anchor(self, t: int) -> tuple[int, int, int]:
        """(phase, x0, y0) such that the glider at time t is glider(dir, phase, x0, y0)."""
        sx, sy = P.DIRS[self.dir]
        k, p = divmod(t - self.tref, 4)
        x0 = sx * k
        y0 = sx * sy * x0 - sx * self.lane
        return p, x0, y0

    def pos(self, t: int) -> tuple[int, int]:
        """Approximate centre of the glider at time t."""
        p, x0, y0 = self.anchor(t)
        sx, sy = P.DIRS[self.dir]
        return x0 + 1 + sx * p / 4, y0 + 1 + sy * p / 4

    def time_at(self, x: float) -> float:
        """Generation at which the glider centre has x-coordinate x."""
        sx, _ = P.DIRS[self.dir]
        return self.tref + 4 * sx * (x - 1)

    def cells_at(self, t: int) -> np.ndarray:
        p, x0, y0 = self.anchor(t)
        return P.glider(self.dir, p, x0, y0)


def traj_of(direction: str, phase: int, x: int, y: int, t: int) -> Traj:
    """Trajectory of glider(direction, phase, x, y) seen at generation t."""
    sx, sy = P.DIRS[direction]
    t0 = t - phase  # at t0 it was glider(direction, 0, x, y)
    return Traj(direction, sy * x - sx * y, t0 - 4 * sx * x)


def components(cells: np.ndarray, gap: int = 1) -> list[np.ndarray]:
    """Group cells into 8-connected clusters (after dilating by `gap`-1 cells)."""
    if not len(cells):
        return []
    mn = cells.min(0)
    c = cells - mn
    img = np.zeros(c.max(0)[::-1] + 1, bool)
    img[c[:, 1], c[:, 0]] = True
    if gap > 1:
        img2 = ndimage.binary_dilation(img, np.ones((gap, gap), bool))
    else:
        img2 = img
    lab, n = ndimage.label(img2, structure=np.ones((3, 3), bool))
    ids = lab[c[:, 1], c[:, 0]]
    return [cells[ids == i] for i in range(1, n + 1)]


def identify(cells: np.ndarray, t: int) -> tuple[list[Traj], np.ndarray]:
    """Split cells (seen at generation t) into isolated gliders and the rest."""
    trajs, rest = [], []
    for comp in components(cells):
        key = frozenset(map(tuple, (comp - comp.min(0)).tolist()))
        hit = _SHAPES.get(key)
        if hit is None:
            rest.append(comp)
            continue
        d, p, (ox, oy) = hit
        mx, my = comp.min(0)
        trajs.append(traj_of(d, p, int(mx - ox), int(my - oy), t))
    rest_cells = np.concatenate(rest) if rest else np.zeros((0, 2), np.int64)
    return sorted(trajs), rest_cells


def is_still(cells: np.ndarray, period: int = 1) -> bool:
    if not len(cells):
        return True
    mn = cells.min(0) - 2
    h, w = (cells.max(0) - mn + 3)[::-1]
    a = pack(cells, h, w, mn[0], mn[1])
    b = a
    for _ in range(period):
        b = step(b)
    return bool(np.array_equal(a, b))
