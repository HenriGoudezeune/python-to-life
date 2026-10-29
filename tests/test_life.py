"""Simulator, RLE and glider bookkeeping."""
import numpy as np
import pytest

from gol import gliders as G
from gol import life, rle
from gol import patterns as P


def naive_step(s: set) -> set:
    cnt: dict = {}
    for x, y in s:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    cnt[(x + dx, y + dy)] = cnt.get((x + dx, y + dy), 0) + 1
    return {c for c, n in cnt.items() if n == 3 or (n == 2 and c in s)}


def as_set(cells):
    return set(map(tuple, np.asarray(cells).tolist()))


@pytest.mark.parametrize("seed", range(4))
def test_universe_matches_naive(seed):
    rng = np.random.default_rng(seed)
    cells = rng.integers(-40, 140, (1500, 2))   # crosses tile borders and negative coords
    s = as_set(cells)
    u = life.Universe(cells)
    for _ in range(60):
        s = naive_step(s)
        u.step(1)
    assert as_set(u.cells) == s


def test_dense_step_matches_naive():
    rng = np.random.default_rng(7)
    cells = rng.integers(10, 110, (2000, 2))
    a = life.pack(cells, 128, 128)
    s = as_set(cells)
    for _ in range(30):
        a = life.step(a)
        s = naive_step(s)
    assert as_set(life.unpack(a)) == s


def test_glider_moves_one_cell_per_four_generations():
    for d, (sx, sy) in P.DIRS.items():
        g = P.glider(d)
        u = life.Universe(g)
        u.step(40)
        assert as_set(u.cells) == as_set(g + [10 * sx, 10 * sy])


def test_gosper_gun_period_30_and_emits_gliders():
    u = life.Universe(P.GOSPER_GUN)

    def core(c):
        return {p for p in as_set(c) if 0 <= p[0] < 36 and 0 <= p[1] < 9}

    c0 = core(u.cells)
    u.step(30)
    assert core(u.cells) == c0
    u.step(90)
    assert u.population == 36 + 4 * 5  # 4 gliders emitted after 120 generations


def test_still_lifes_are_still():
    for name, sl in P.STILL_LIFES.items():
        assert G.is_still(sl), name


def test_rle_roundtrip():
    rng = np.random.default_rng(1)
    c = rng.integers(-30, 200, (3000, 2))
    back, _ = rle.parse(rle.dumps(c))
    assert as_set(back) == as_set(c)
    cells, meta = rle.parse("#C test\nx = 3, y = 3, rule = B3/S23\nbo$2bo$3o!\n")
    assert as_set(cells) == as_set(P.GLIDER_SE)
    assert meta["comments"] == ["#C test"]


def test_trajectory_identify_and_place():
    rng = np.random.default_rng(3)
    for _ in range(50):
        d = list(P.DIRS)[rng.integers(4)]
        tr = G.Traj(d, int(rng.integers(-30, 30)), int(rng.integers(-50, 50)))
        t = int(rng.integers(0, 40))
        u = life.Universe(tr.cells_at(t))
        u.step(9)
        found, rest = G.identify(u.cells, t + 9)
        assert found == [tr] and not len(rest)
