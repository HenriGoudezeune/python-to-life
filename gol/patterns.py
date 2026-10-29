"""Small pattern library and geometric helpers.

All patterns are (N,2) int arrays of (x, y) with y pointing down.
"""
from __future__ import annotations

import numpy as np

from . import rle


def from_rows(*rows: str) -> np.ndarray:
    pts = [(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch in "O*o#"]
    return np.array(pts, np.int64).reshape(-1, 2)


def from_rle(text: str) -> np.ndarray:
    return rle.parse(text)[0]


# Glider moving south-east (+x, +y), phase 0.
GLIDER_SE = from_rows(".O.", "..O", "OOO")
BLOCK = from_rows("OO", "OO")
BEEHIVE = from_rows(".OO.", "O..O", ".OO.")
BOAT = from_rows("OO.", "O.O", ".O.")
SHIP = from_rows("OO.", "O.O", ".OO")
LOAF = from_rows(".OO.", "O..O", ".O.O", "..O.")
TUB = from_rows(".O.", "O.O", ".O.")
POND = from_rows(".OO.", "O..O", "O..O", ".OO.")
EATER1 = from_rows("OO..", "O.O.", "..O.", "..OO")
GOSPER_GUN = from_rle(
    "x = 36, y = 9, rule = B3/S23\n"
    "24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4b"
    "obo$10bo5bo7bo$11bo3bo$12b2o!"
)

STILL_LIFES = {
    "block": BLOCK,
    "beehive": BEEHIVE,
    "boat": BOAT,
    "ship": SHIP,
    "loaf": LOAF,
    "tub": TUB,
    "pond": POND,
    "eater1": EATER1,
}

# The 8 symmetries of the square as (a, b, c, d): (x, y) -> (a*x + b*y, c*x + d*y)
SYMMETRIES = [
    (1, 0, 0, 1), (-1, 0, 0, 1), (1, 0, 0, -1), (-1, 0, 0, -1),
    (0, 1, 1, 0), (0, -1, 1, 0), (0, 1, -1, 0), (0, -1, -1, 0),
]

# Directions as unit (dx, dy) per 4 generations
DIRS = {"SE": (1, 1), "SW": (-1, 1), "NE": (1, -1), "NW": (-1, -1)}


def transform(cells: np.ndarray, sym, dx: int = 0, dy: int = 0) -> np.ndarray:
    a, b, c, d = sym
    x, y = cells[:, 0], cells[:, 1]
    return np.stack([a * x + b * y + dx, c * x + d * y + dy], axis=1)


def normalize(cells: np.ndarray) -> np.ndarray:
    """Shift to min corner (0,0) and sort, for shape comparison."""
    if not len(cells):
        return cells
    c = cells - cells.min(0)
    return c[np.lexsort((c[:, 0], c[:, 1]))]


def shift(cells: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return cells + np.array([dx, dy], np.int64)


def glider(direction: str = "SE", phase: int = 0, x: int = 0, y: int = 0) -> np.ndarray:
    """Glider heading `direction`, `phase` generations (0-3) past the canonical
    phase-0 shape, with the canonical phase-0 bounding box at (x, y).

    A glider with phase p at (x, y) is exactly the phase-0 glider at (x, y)
    run for p generations, so `glider(d, p, x, y)` after 4-p more generations
    equals `glider(d, 0, x + dx, y + dy)`.
    """
    sx, sy = DIRS[direction]
    sym = (sx, 0, 0, sy)
    g = transform(GLIDER_SE, sym)
    g = g - g.min(0)
    if phase:
        from .life import pack, run, unpack
        a = pack(g + 4, 12, 12)
        g = unpack(run(a, phase)) - 4
    return shift(g, x, y)


def union(*parts) -> np.ndarray:
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.zeros((0, 2), np.int64)
    c = np.concatenate(parts)
    return np.unique(c, axis=0)
