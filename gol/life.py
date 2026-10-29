"""Fast Game of Life simulation on bit-packed numpy arrays.

A board is stored as uint64 words: row y, word j, bit i holds cell x = 64*j + i.
`step` works on arrays of shape (..., H, W) so many small boards can be
simulated at once (used by the reaction search). `Universe` wraps one board
with an origin offset and grows it when cells approach the edge.
"""
from __future__ import annotations

import numpy as np

U64 = np.uint64
ONE = U64(1)
S63 = U64(63)


def _shift_x(a: np.ndarray, d: int) -> np.ndarray:
    """Return board whose cell x holds a's cell x-d (d = +1 or -1)."""
    out = np.empty_like(a)
    if d == 1:
        out[...] = a << ONE
        out[..., 1:] |= a[..., :-1] >> S63
    else:
        out[...] = a >> ONE
        out[..., :-1] |= a[..., 1:] << S63
    return out


def _shift_y(a: np.ndarray, d: int) -> np.ndarray:
    out = np.zeros_like(a)
    if d == 1:
        out[..., 1:, :] = a[..., :-1, :]
    else:
        out[..., :-1, :] = a[..., 1:, :]
    return out


def step(a: np.ndarray) -> np.ndarray:
    """One generation (B3/S23). Cells beyond the board edge count as dead."""
    l = _shift_x(a, 1)
    r = _shift_x(a, -1)
    # per row: sum of (l, c, r) as t1 + 2*t2, and of (l, r) as p1 + 2*p2
    t1 = l ^ a ^ r
    t2 = (l & a) | (r & (l ^ a))
    p1 = l ^ r
    p2 = l & r
    u1, u2 = _shift_y(t1, 1), _shift_y(t2, 1)
    d1, d2 = _shift_y(t1, -1), _shift_y(t2, -1)
    # ones column: u1 + d1 + p1 = a0 + 2*a1
    a0 = u1 ^ d1 ^ p1
    a1 = (u1 & d1) | (p1 & (u1 ^ d1))
    # twos column: u2 + d2 + p2 + a1; we need "exactly one" of these
    s, c = u2 ^ d2, u2 & d2
    s2, c2 = p2 ^ a1, p2 & a1
    exactly1 = (s ^ s2) & ~(c | c2)
    return exactly1 & (a0 | a)


def run(a: np.ndarray, gens: int) -> np.ndarray:
    for _ in range(gens):
        a = step(a)
    return a


def pack(cells: np.ndarray, h: int, w: int, ox: int = 0, oy: int = 0) -> np.ndarray:
    """Pack (N,2) array of (x, y) into a board of h rows and >= w columns."""
    wn = (w + 63) // 64
    a = np.zeros((h, wn), dtype=U64)
    if len(cells):
        xs = cells[:, 0] - ox
        ys = cells[:, 1] - oy
        if xs.min() < 0 or ys.min() < 0 or xs.max() >= wn * 64 or ys.max() >= h:
            raise ValueError("cells outside board")
        np.bitwise_or.at(a, (ys, xs // 64), ONE << (xs % 64).astype(U64))
    return a


def unpack(a: np.ndarray, ox: int = 0, oy: int = 0) -> np.ndarray:
    """Inverse of pack for a single 2-D board: (N,2) array of (x, y), sorted by y then x."""
    bits = np.unpackbits(a.view(np.uint8).reshape(a.shape[0], -1), axis=1, bitorder="little")
    ys, xs = np.nonzero(bits)
    return np.stack([xs + ox, ys + oy], axis=1).astype(np.int64)


def popcount(a: np.ndarray) -> int:
    return int(np.unpackbits(a.view(np.uint8)).sum())


class Universe:
    """Unbounded board for sparse patterns.

    The board is cut into tiles of TH rows x 64 columns (one uint64 word).
    Only tiles that may change are recomputed: a tile is active in the next
    generation if it changed, or a neighbour changed along the shared edge.
    The board grows automatically when activity reaches its outer tile ring.
    """

    TH = 32

    def __init__(self, cells=None, pad: int = 128):
        cells = np.zeros((0, 2), np.int64) if cells is None else np.asarray(cells, np.int64).reshape(-1, 2)
        self.gen = 0
        self._build(cells, pad)

    def _build(self, cells, pad):
        th = self.TH
        if len(cells):
            x0, y0 = cells.min(0) - pad
            x1, y1 = cells.max(0) + pad
        else:
            x0 = y0 = -pad
            x1 = y1 = pad
        x0 = int(x0) // 64 * 64
        y0 = int(y0) // th * th
        nty = (int(y1) - y0) // th + 1
        ntx = (int(x1) - x0) // 64 + 1
        self.ox, self.oy = x0, y0
        self.board = pack(cells, nty * th, ntx * 64, x0, y0)
        occ = self.board.reshape(nty, th, ntx).any(axis=1)
        act = occ.copy()
        act[1:, :] |= occ[:-1, :]
        act[:-1, :] |= occ[1:, :]
        a2 = act.copy()
        act[:, 1:] |= a2[:, :-1]
        act[:, :-1] |= a2[:, 1:]
        self.active = act

    @property
    def cells(self) -> np.ndarray:
        return unpack(self.board, self.ox, self.oy)

    @property
    def population(self) -> int:
        return popcount(self.board)

    def _maybe_grow(self):
        a = self.active
        if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
            nty, ntx = a.shape
            pad = max(128, (nty * self.TH + ntx * 64) // 4)
            self._build(self.cells, pad)

    def step(self, n: int = 1):
        th = self.TH
        off = np.arange(-1, th + 1)
        for _ in range(n):
            self._maybe_grow()
            ty, tx = np.nonzero(self.active)
            self.gen += 1
            if not len(ty):
                continue
            rows = ty[:, None] * th + off[None, :]
            cols = tx[:, None] + np.array([-1, 0, 1])[None, :]
            blk = self.board[rows[:, :, None], cols[:, None, :]]
            new = step(blk)[:, 1:-1, 1]
            old = blk[:, 1:-1, 1]
            diff = new ^ old
            self.board[rows[:, 1:-1], tx[:, None]] = new
            ch = diff.any(axis=1)
            top = diff[:, 0] != 0
            bot = diff[:, -1] != 0
            anyd = np.bitwise_or.reduce(diff, axis=1)
            left = (anyd & ONE) != 0
            right = (anyd >> S63) != 0
            act = np.zeros_like(self.active)
            for m, dy, dx in ((ch, 0, 0), (top, -1, 0), (bot, 1, 0), (left, 0, -1), (right, 0, 1),
                              (top & left, -1, -1), (top & right, -1, 1), (bot & left, 1, -1), (bot & right, 1, 1)):
                if m.any():
                    act[ty[m] + dy, tx[m] + dx] = True
            self.active = act
        return self
