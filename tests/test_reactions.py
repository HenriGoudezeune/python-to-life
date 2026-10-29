"""Every reaction the compiler relies on is re-simulated here."""
import numpy as np
import pytest

from gol import gliders as G
from gol import reactions as R
from gol.compiler import layout as LY
from gol.life import Universe


def simulate(inputs, statics=np.zeros((0, 2), np.int64), t0=-200, gens=600):
    cells = np.concatenate([t.cells_at(t0) for t in inputs] + [statics])
    u = Universe(cells)
    u.gen = t0
    u.step(gens)
    return G.identify(u.cells, u.gen)


def same_cells(a, b):
    return set(map(tuple, a.tolist())) == set(map(tuple, b.tolist()))


@pytest.mark.parametrize("key", sorted(LY.KILL))
def test_kill_reactions_annihilate(key):
    gl, rest = simulate(LY.KILL[key].inputs)
    assert gl == [] and len(rest) == 0


def test_kickback():
    gl, rest = simulate(LY.KB.inputs)
    assert gl == LY.KB.outputs and not len(rest)
    assert LY.KB_DTHETA == 17


@pytest.mark.parametrize("pair", [("SW", "SE"), ("SE", "NE"), ("NE", "SE"), ("SE", "SW")])
@pytest.mark.parametrize("kind", ["boat", "eater1"])
def test_one_shot_reflectors(pair, kind):
    for r in R.reflectors(*pair, (kind,)):
        gl, rest = simulate(r.inputs, r.statics)
        assert gl == r.outputs and not len(rest)


@pytest.mark.parametrize("d", ["SE", "SW", "NE", "NW"])
def test_eaters_survive(d):
    for r in R.eaters(d):
        gl, rest = simulate(r.inputs, r.statics)
        assert gl == [] and same_cells(rest, r.statics)


def test_fanout_second_constant():
    """FAN core: input + C (kickback) + G2 kills the kickback glider;
    without input, C and G2 pass each other untouched."""
    a = next(t for t in LY.KB.inputs if t.dir == "SE")
    c = next(t for t in LY.KB.inputs if t.dir == "SW")
    gl, rest = simulate([a, c, LY.FAN_G2])
    assert gl == [] and not len(rest)
    gl, rest = simulate([c, LY.FAN_G2])
    assert gl == sorted([c, LY.FAN_G2]) and not len(rest)


@pytest.mark.parametrize("dth", [19, 20, 25, 40, -19, -20, -25, -40])
@pytest.mark.parametrize("par", [0, 1])
def test_crossing_with_large_timing_offset_passes(dth, par):
    """A row and a column whose theta differs by >= 19 never interact."""
    row = G.Traj("SE", 0, 0)
    col = G.Traj("SW", 20 + par, dth + 4 * (20 + par))
    assert R.theta(col) - R.theta(row) == dth
    gl, rest = simulate([row, col])
    assert gl == sorted([row, col]) and not len(rest)
