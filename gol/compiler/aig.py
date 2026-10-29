"""And-inverter graph with structural hashing, plus bit-vector arithmetic.

A literal is an int: 2*node + neg. Node 0 is constant false, so literal 0 is
False and literal 1 is True. Other nodes are primary inputs or 2-input ANDs.
"""
from __future__ import annotations

FALSE, TRUE = 0, 1


def neg(l: int) -> int:
    return l ^ 1


class AIG:
    def __init__(self):
        self.nodes: list = [("const",)]        # node id -> ("const",) | ("in", name, bit) | ("and", a, b)
        self._hash: dict = {}
        self.inputs: dict = {}                # (name, bit) -> literal

    def input(self, name: str, bit: int) -> int:
        key = (name, bit)
        if key not in self.inputs:
            self.nodes.append(("in", name, bit))
            self.inputs[key] = 2 * (len(self.nodes) - 1)
        return self.inputs[key]

    def AND(self, a: int, b: int) -> int:
        if a == FALSE or b == FALSE or a == neg(b):
            return FALSE
        if a == TRUE:
            return b
        if b == TRUE or a == b:
            return a
        if a > b:
            a, b = b, a
        key = (a, b)
        if key not in self._hash:
            self.nodes.append(("and", a, b))
            self._hash[key] = 2 * (len(self.nodes) - 1)
        return self._hash[key]

    def OR(self, a, b):
        return neg(self.AND(neg(a), neg(b)))

    def XOR(self, a, b):
        if a == FALSE:
            return b
        if b == FALSE:
            return a
        if a == TRUE:
            return neg(b)
        if b == TRUE:
            return neg(a)
        if a == b:
            return FALSE
        if a == neg(b):
            return TRUE
        return self.OR(self.AND(a, neg(b)), self.AND(neg(a), b))

    def MUX(self, s, a, b):
        """s ? a : b"""
        if s == TRUE or a == b:
            return a
        if s == FALSE:
            return b
        return self.OR(self.AND(s, a), self.AND(neg(s), b))

    def MAJ(self, a, b, c):
        return self.OR(self.AND(a, b), self.AND(c, self.OR(a, b)))

    def eval(self, lit: int, values: dict) -> int:
        """Evaluate literal for input values {name: int} (for self-checks)."""
        memo: dict = {}

        def node(n):
            if n in memo:
                return memo[n]
            k = self.nodes[n]
            if k[0] == "const":
                v = 0
            elif k[0] == "in":
                v = (values[k[1]] >> k[2]) & 1
            else:
                v = lit_(k[1]) & lit_(k[2])
            memo[n] = v
            return v

        def lit_(l):
            return node(l >> 1) ^ (l & 1)

        return lit_(lit)


# ---------------- bit vectors (LSB first, unsigned) ----------------

def const_vec(value: int, width: int | None = None) -> list:
    if width is None:
        width = max(1, value.bit_length())
    return [TRUE if (value >> i) & 1 else FALSE for i in range(width)]


def extend(v: list, width: int) -> list:
    return (v + [FALSE] * width)[:width]


def add(g: AIG, a: list, b: list, cin: int = FALSE, width: int | None = None) -> list:
    if width is None:
        width = max(len(a), len(b)) + 1
    a, b = extend(a, width), extend(b, width)
    out, c = [], cin
    for x, y in zip(a, b):
        out.append(g.XOR(g.XOR(x, y), c))
        c = g.MAJ(x, y, c)
    return out


def sub(g: AIG, a: list, b: list, width: int | None = None) -> list:
    """a - b modulo 2**width (default: max width)."""
    if width is None:
        width = max(len(a), len(b))
    return add(g, extend(a, width), [neg(x) for x in extend(b, width)], TRUE, width)


def less_than(g: AIG, a: list, b: list) -> int:
    """unsigned a < b: borrow out of a - b."""
    w = max(len(a), len(b))
    a, b = extend(a, w), extend(b, w)
    c = TRUE  # carry of a + ~b + 1; a < b  <=>  no carry out
    for x, y in zip(a, b):
        c = g.MAJ(x, neg(y), c)
    return neg(c)


def equal(g: AIG, a: list, b: list) -> int:
    w = max(len(a), len(b))
    r = TRUE
    for x, y in zip(extend(a, w), extend(b, w)):
        r = g.AND(r, neg(g.XOR(x, y)))
    return r


def mul(g: AIG, a: list, b: list, width: int) -> list:
    acc = [FALSE] * width
    for i, bit in enumerate(b):
        if i >= width or bit == FALSE:
            continue
        part = [g.AND(bit, x) for x in ([FALSE] * i + a)[:width]]
        acc = add(g, acc, part, width=width)
    return acc


def mux_vec(g: AIG, s: int, a: list, b: list) -> list:
    w = max(len(a), len(b))
    return [g.MUX(s, x, y) for x, y in zip(extend(a, w), extend(b, w))]


def shift_left(g: AIG, a: list, amount: list, width: int) -> list:
    v = extend(a, width)
    for i, s in enumerate(amount):
        k = 1 << i
        shifted = ([FALSE] * k + v)[:width] if k < width else [FALSE] * width
        v = mux_vec(g, s, shifted, v)
    return v


def shift_right(g: AIG, a: list, amount: list) -> list:
    v = list(a)
    for i, s in enumerate(amount):
        k = 1 << i
        shifted = v[k:] + [FALSE] * min(k, len(v))
        v = mux_vec(g, s, shifted, v)
    return v
