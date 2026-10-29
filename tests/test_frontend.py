"""Front-half compiler tests: Python subset -> AIG -> NOR/FAN netlist.

No Game of Life simulation here. Every program is compiled with ``to_aig`` and
``to_netlist``; both are evaluated and compared against running the very same
source as ordinary Python (with ``u1..u64`` bound to ``int``), masked to the
output width(s). All input combinations are checked when the inputs have at
most 10 bits in total, otherwise 200 random combinations with a fixed seed.
"""
from __future__ import annotations

import ast
import itertools
import random

import pytest

from gol.compiler.frontend import CompileError, to_aig
from gol.compiler.techmap import to_netlist

EXHAUSTIVE_BITS = 10
SAMPLES = 200


# ---------------------------------------------------------------- harness

def _python_fn(src: str):
    ns = {f"u{i}": int for i in range(1, 65)}
    exec(compile(src, "<program>", "exec"), ns)
    name = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)][-1].name
    return ns[name]


def _input_vectors(inputs, seed=1234):
    total = sum(w for _, w in inputs)
    names = [n for n, _ in inputs]
    if total <= EXHAUSTIVE_BITS:
        for combo in itertools.product(*[range(1 << w) for _, w in inputs]):
            yield dict(zip(names, combo))
    else:
        rng = random.Random(seed)
        # always include the corners
        yield {n: 0 for n, _ in inputs}
        yield {n: (1 << w) - 1 for n, w in inputs}
        for _ in range(SAMPLES):
            yield {n: rng.getrandbits(w) for n, w in inputs}


def _expected(fn, vals, outputs):
    res = fn(**vals)
    if len(outputs) == 1 and outputs[0][0] == "out":
        res = (res,)
    assert isinstance(res, tuple) and len(res) == len(outputs), \
        f"python returned {res!r}, compiler has outputs {[n for n, _ in outputs]}"
    return {name: int(r) & ((1 << len(lits)) - 1) for (name, lits), r in zip(outputs, res)}


def check(src: str):
    g, inputs, outputs = to_aig(src)
    net = to_netlist(g, inputs, outputs)
    net.check()
    fn = _python_fn(src)
    for vals in _input_vectors(inputs):
        want = _expected(fn, vals, outputs)
        got_aig = {name: sum(g.eval(l, vals) << i for i, l in enumerate(lits))
                   for name, lits in outputs}
        assert got_aig == want, f"AIG mismatch for {vals}: got {got_aig}, want {want}"
        got_net = net.evaluate(vals)
        # outputs of width 0 never appear in the netlist; treat them as 0
        got_net = {name: got_net.get(name, 0) for name, _ in outputs}
        assert got_net == want, f"netlist mismatch for {vals}: got {got_net}, want {want}"


# ---------------------------------------------------------------- programs

PROGRAMS = {
    # --- arithmetic -------------------------------------------------------
    "add": """
def f(a: u4, b: u4):
    return a + b
""",
    "add3_truncated": """
def f(a: u3, b: u3, c: u3) -> u4:
    return a + b + c
""",
    "add16_random": """
def f(a: u16, b: u16) -> u16:
    return a + b
""",
    "add_const_growth": """
def f(a: u4):
    return a + 13 + 2 * 3
""",
    "sub_wrap_annotation": """
def f(a: u4, b: u4) -> u4:
    return a - b
""",
    "sub_wrap_masked": """
def f(a: u4, b: u4):
    return (a - b) & 15
""",
    "sub_const_left": """
def f(a: u3) -> u3:
    return 1 - a
""",
    "unary_minus": """
def f(a: u4) -> u4:
    return -a
""",
    "invert": """
def f(a: u5) -> u5:
    return ~a
""",
    "mul_small": """
def f(a: u4, b: u4):
    return a * b
""",
    "mul_const_plus": """
def f(a: u4, b: u3):
    return a * 3 + b * 5
""",
    "mul8_random": """
def f(a: u8, b: u8):
    return a * b
""",
    "floordiv_mod_pow2": """
def f(a: u7):
    return a // 4, a % 8, a // 1, a % 1, a // 128
""",
    "collatz_step": """
def f(a: u6):
    return a // 2 if a % 2 == 0 else 3 * a + 1
""",
    # --- shifts -----------------------------------------------------------
    "shift_const": """
def f(a: u6):
    return (a << 3) | (a >> 2) | (a >> 9)
""",
    "shift_left_var": """
def f(a: u4, s: u3):
    return a << s
""",
    "shift_right_var": """
def f(a: u7, s: u3):
    return a >> s
""",
    "rotate_left_var": """
def f(a: u6, s: u2) -> u6:
    return ((a << s) | (a >> (6 - s))) & 63
""",
    # --- comparisons ------------------------------------------------------
    "compare_all": """
def f(a: u3, b: u3):
    return a < b, a <= b, a > b, a >= b, a == b, a != b
""",
    "compare_mixed_width": """
def f(a: u2, b: u5):
    return a < b, b <= a, a == b, b != a, a >= 3, b > 17
""",
    "compare_chained": """
def f(a: u3, b: u3, c: u3):
    return a < b < c, a <= b == c, a != b != c, 1 < a <= b < 6
""",
    # --- boolean logic ----------------------------------------------------
    "bool_params": """
def f(x: bool, y: bool, z: bool) -> bool:
    return (x or y) and not z
""",
    "bool_logic_vectors": """
def f(a: u2, b: u2, c: u3):
    return bool((a and b) or not c), not a, bool(c), (a > 1 and b < 2) or c == 3
""",
    "builtins_abs_int_bool": """
def f(a: u3, b: u3):
    return int(a > b) + int(bool(b)) + abs(a)
""",
    "xor_and_or_widths": """
def f(a: u3, b: u5):
    return a & b, a | b, a ^ b, ~(a & b) & 7
""",
    # --- muxes / control flow ---------------------------------------------
    "mux_if_else": """
def f(a: u4, b: u4, s: bool):
    if s:
        r = a
    else:
        r = b + 1
    return r
""",
    "mux_ternary": """
def f(a: u4, b: u4, c: u2):
    return a if c else b
""",
    "elif_chain": """
def f(a: u4) -> u3:
    if a < 3:
        r = 0
    elif a < 7:
        r = 1
    elif a == 7:
        r = 2
    else:
        r = 3
    return r + 1
""",
    "nested_ifs": """
def f(a: u3, b: u3, c: bool):
    r = 0
    if a > b:
        if c:
            r = a - b
        else:
            r = a
    else:
        if a == b:
            r = 7
        elif c:
            r = b
    return r
""",
    "early_return_in_if": """
def f(a: u4, b: u4) -> u5:
    r = a
    if a == 0:
        return b
    r = r + b
    if r > 20:
        return 20
    r += 1
    return r
""",
    "early_return_tuple": """
def f(a: u3, b: u3):
    if a > b:
        return a - b, True
    return b - a, False
""",
    "var_defined_in_one_branch": """
def f(a: u4, b: u4) -> u4:
    if a > b:
        d = a - b
        return d
    else:
        e = b - a
    return e
""",
    "var_defined_in_one_branch_guarded_use": """
def f(a: u4, c: bool) -> u4:
    if c:
        t = a + 3
    if c:
        return t
    return a
""",
    # --- loops ------------------------------------------------------------
    "first_set_bit_early_return_in_loop": """
def f(a: u8) -> u4:
    for i in range(8):
        if (a >> i) & 1:
            return i
    return 8
""",
    "prefix_sum_exceeds": """
def f(a: u8, b: u4) -> u4:
    s = 0
    for i in range(4):
        s += (a >> (2 * i)) & 3
        if s > b:
            return i
    return 15
""",
    "loop_accumulate_mul": """
def f(a: u4):
    acc = 0
    for i in range(5):
        acc += a * i
    return acc
""",
    "loop_range_start_stop_step": """
def f(a: u8):
    acc = 0
    for i in range(1, 8, 2):
        acc = acc + ((a >> i) & 1)
    return acc
""",
    "popcount8": """
def f(a: u8):
    c = 0
    for i in range(8):
        c += (a >> i) & 1
    return c
""",
    "popcount16_random": """
def f(a: u16):
    c = 0
    for i in range(16):
        c += (a >> i) & 1
    return c
""",
    "parity8": """
def f(a: u8) -> bool:
    p = 0
    for i in range(8):
        p ^= (a >> i) & 1
    return p
""",
    "nested_loops": """
def f(a: u3, b: u3):
    acc = 0
    for i in range(3):
        for j in range(3):
            acc += (a >> i) & (b >> j) & 1
    return acc
""",
    "bit_reverse": """
def f(a: u6):
    r = 0
    for i in range(6):
        r = (r << 1) | ((a >> i) & 1)
    return r
""",
    # --- min / max --------------------------------------------------------
    "min_max_two": """
def f(a: u4, b: u4):
    return min(a, b), max(a, b)
""",
    "min_max_three_mixed": """
def f(a: u3, b: u4, c: u2):
    return min(a, b, c), max(a, b, c), max(a, 5), min(b, 9)
""",
    # --- assignment forms -------------------------------------------------
    "tuple_return_annotated_subscript": """
def f(a: u4, b: u4) -> tuple[u5, u4]:
    return a + b, a - b
""",
    "tuple_return_annotated_tuple": """
def f(a: u4, b: u3) -> (u3, u8):
    return a * b, a * b
""",
    "tuple_unpack_swap": """
def f(a: u4, b: u4):
    a, b = b, a
    x, (y, z) = a + 1, (b, a ^ b)
    return a, b, x, y, z
""",
    "sort3_by_swaps": """
def f(a: u3, b: u3, c: u3):
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return a, b, c
""",
    "bubble_sort_loops": """
def f(a: u3, b: u3, c: u3, d: u3) -> tuple[u3, u3, u3, u3]:
    for _ in range(3):
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if c > d:
            c, d = d, c
    return a, b, c, d
""",
    "augmented_assignment": """
def f(a: u4, b: u4, c: u3) -> u6:
    x = a
    x += b
    x *= 3
    x -= c
    x &= 31
    x |= 1
    x ^= a
    x >>= 1
    x <<= 2
    x %= 64
    x //= 2
    return x
""",
    "ann_assign_truncates": """
def f(a: u4, b: u4):
    s: u3 = a + b
    t: u6 = a
    return s, t + 0
""",
    "reassign_parameter": """
def f(a: u4) -> u4:
    a = a + 1
    a = a * 2
    return a
""",
    "constant_only": """
def f(a: u2):
    return 5 + 3 * 2
""",
}


@pytest.mark.parametrize("src", list(PROGRAMS.values()), ids=list(PROGRAMS))
def test_program_matches_python(src):
    check(src)


def test_enough_programs():
    assert len(PROGRAMS) >= 25


# ---------------------------------------------------------------- interface

def test_output_names_and_widths():
    _, inputs, outputs = to_aig("def f(a: u4, b: bool) -> u6:\n    return a\n")
    assert inputs == [("a", 4), ("b", 1)]
    assert [n for n, _ in outputs] == ["out"]
    assert len(outputs[0][1]) == 6

    _, _, outputs = to_aig("def f(a: u4) -> tuple[u2, u7]:\n    return a, a\n")
    assert [(n, len(v)) for n, v in outputs] == [("out0", 2), ("out1", 7)]


def test_add_grows_by_one_bit_mul_adds_widths():
    _, _, outs = to_aig("def f(a: u4, b: u3):\n    return a + b, a * b, a - b, ~b\n")
    assert [len(v) for _, v in outs] == [5, 7, 4, 3]


# ---------------------------------------------------------------- errors

ERRORS = {
    "while_loop": """
def f(a: u4):
    while a:
        a = a - 1
    return a
""",
    "missing_annotation": """
def f(a, b: u4):
    return b
""",
    "bad_annotation": """
def f(a: int):
    return a
""",
    "division_by_3": """
def f(a: u4):
    return a // 3
""",
    "mod_by_6": """
def f(a: u4):
    return a % 6
""",
    "division_by_variable": """
def f(a: u4, b: u2):
    return a // b
""",
    "true_division": """
def f(a: u4):
    return a / 2
""",
    "unknown_name": """
def f(a: u4):
    return a + zz
""",
    "float_constant": """
def f(a: u4):
    return a + 1.5
""",
    "string_constant": """
def f(a: u4):
    return "x"
""",
    "break_in_loop": """
def f(a: u4):
    for i in range(3):
        break
    return a
""",
    "range_not_constant": """
def f(a: u4):
    s = 0
    for i in range(a):
        s += 1
    return s
""",
    "for_over_non_range": """
def f(a: u4):
    for i in [1, 2]:
        a = a + i
    return a
""",
    "never_returns": """
def f(a: u4):
    b = a
""",
    "return_shape_mismatch": """
def f(a: u4):
    if a:
        return a, a
    return a
""",
    "unsupported_call": """
def f(a: u4):
    return len(a)
""",
    "power_operator": """
def f(a: u4):
    return a ** 2
""",
}


@pytest.mark.parametrize("src", list(ERRORS.values()), ids=list(ERRORS))
def test_unsupported_raises_compile_error(src):
    with pytest.raises(CompileError):
        to_aig(src)
