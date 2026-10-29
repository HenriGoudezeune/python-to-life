"""Whole programs: the Game of Life result must equal what Python computes."""
import itertools
import random
from pathlib import Path

import pytest

from gol.compiler import compile_function, drc
from gol.compiler.__main__ import python_reference

EX = Path(__file__).resolve().parent.parent / "examples"


def check(src, fname, cases=None):
    prog = compile_function(src, fname)
    assert drc.check(prog.layout) == []
    if cases is None:
        cases = itertools.product(*[range(1 << w) for _, w in prog.inputs])
    for vals in cases:
        kw = dict(zip([n for n, _ in prog.inputs], vals))
        r = prog.run(**kw)
        assert r.clean, (kw, r.problems)
        assert prog.result(r.outputs) == python_reference(src, fname, prog, kw), kw


def test_majority_exhaustive():
    check((EX / "logic.py").read_text(), "majority")


def test_xor3_exhaustive():
    check("def xor3(a: bool, b: bool, c: bool) -> bool:\n    return a ^ b ^ c\n", "xor3")


@pytest.mark.slow
def test_add2_exhaustive():
    check("def add2(a: u2, b: u2) -> u3:\n    return a + b\n", "add2")


@pytest.mark.slow
def test_popcount_exhaustive():
    check((EX / "popcount.py").read_text(), "popcount")


@pytest.mark.slow
def test_max2_sample():
    check((EX / "max2.py").read_text(), "max2", [(0, 0), (1, 2), (2, 1), (3, 3), (3, 0), (0, 3)])


@pytest.mark.slow
def test_add4_sample():
    rnd = random.Random(4)
    cases = [(0, 0), (15, 15), (15, 1), (8, 8)] + [(rnd.randrange(16), rnd.randrange(16)) for _ in range(4)]
    check((EX / "add4.py").read_text(), "add4", cases)
