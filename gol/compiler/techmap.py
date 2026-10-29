"""AIG -> NOR/FAN netlist.

Every AND node becomes a NOR of the complemented children (AND(a,b) = NOR(~a,~b));
chains of single-use ANDs are merged into one wider NOR. Primary inputs are
free in either polarity. A gate output that is needed k times (in some mix of
polarities) gets a cheapest tree of FAN gates (each FAN gives two inverted
copies) and NOT gates (1-input NOR).
"""
from __future__ import annotations

from functools import lru_cache

from .aig import AIG
from .netlist import Const, Lit, Netlist

FANIN_MAX = 4


@lru_cache(maxsize=None)
def _cost(p: int, q: int):
    """(cost, plan) to supply p copies of w and q copies of ~w from wire w."""
    if (p, q) == (1, 0):
        return 0, ("use",)
    if (p, q) == (0, 1):
        return 1, ("not",)
    best = (_fan_cost(p, q), ("fan",))
    if p + q >= 2:
        c = 1 + _fan_cost(q, p)
        if c < best[0]:
            best = (c, ("not_fan",))
    return best


@lru_cache(maxsize=None)
def _fan_cost(p: int, q: int):
    # FAN gives two copies of ~w: they must supply q (same polarity) and p (inverted)
    best = None
    for q1 in range(q + 1):
        for p1 in range(p + 1):
            q2, p2 = q - q1, p - p1
            if q1 + p1 == 0 or q2 + p2 == 0:
                continue
            c = 1 + _cost(q1, p1)[0] + _cost(q2, p2)[0]
            if best is None or c < best[0]:
                best = (c, (q1, p1))
    return best[0] if best else 10 ** 9


def _split(p, q):
    best = None
    for q1 in range(q + 1):
        for p1 in range(p + 1):
            q2, p2 = q - q1, p - p1
            if q1 + p1 == 0 or q2 + p2 == 0:
                continue
            c = _cost(q1, p1)[0] + _cost(q2, p2)[0]
            if best is None or c < best[0]:
                best = (c, (q1, p1), (q2, p2))
    return best[1], best[2]


def distribute(net: Netlist, w, p: int, q: int):
    """Return (list of p copies of w, list of q copies of ~w)."""
    _, plan = _cost(p, q)
    if plan[0] == "use":
        return [w], []
    if plan[0] == "not":
        return [], [net.nor(w)]
    if plan[0] == "not_fan":
        nw = net.nor(w)
        negs, poss = distribute(net, nw, q, p)
        return poss, negs
    a, b = net.fan(w)
    (q1, p1), (q2, p2) = _split(p, q)
    s1, i1 = distribute(net, a, q1, p1)
    s2, i2 = distribute(net, b, q2, p2)
    return i1 + i2, s1 + s2


def to_netlist(g: AIG, inputs, outputs) -> Netlist:
    net = Netlist(inputs=list(inputs))
    nodes = g.nodes

    def is_and(lit):
        return nodes[lit >> 1][0] == "and"

    # fanout counts over AND-node literals (with polarity) from ANDs and outputs
    refs: dict = {}

    def ref(lit):
        refs[lit] = refs.get(lit, 0) + 1

    reach, stack = set(), [l for _, v in outputs for l in v if is_and(l)]
    while stack:
        n = stack.pop() >> 1
        if n in reach:
            continue
        reach.add(n)
        for c in nodes[n][1:]:
            ref(c)
            if is_and(c):
                stack.append(c)
    for _, v in outputs:
        for l in v:
            ref(l)

    # NOR node per AND node, merging single-use positive AND children
    nor_ins: dict = {}

    def gather(n, acc):
        for c in nodes[n][1:]:
            if is_and(c) and not (c & 1) and refs.get(c, 0) == 1 and refs.get(c ^ 1, 0) == 0 \
                    and len(acc) + 2 <= FANIN_MAX + 1:
                gather(c >> 1, acc)
            else:
                acc.append(c ^ 1)       # NOR input = complemented child
        return acc

    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        ins = gather(n, [])
        nor_ins[n] = ins
        for l in ins:
            if is_and(l):
                visit(l >> 1)
        order.append(n)

    for _, v in outputs:
        for l in v:
            if is_and(l):
                visit(l >> 1)

    # uses per NOR node: positive / negative
    uses: dict = {}
    for n in order:
        for l in nor_ins[n]:
            if is_and(l):
                uses.setdefault(l >> 1, [0, 0])[l & 1] += 1
    for _, v in outputs:
        for l in v:
            if is_and(l):
                uses.setdefault(l >> 1, [0, 0])[l & 1] += 1

    pools: dict = {}

    def signal(l):
        k = nodes[l >> 1]
        if k[0] == "const":
            return Const(l & 1)
        if k[0] == "in":
            return Lit(k[1], k[2], not (l & 1))
        return pools[l >> 1][l & 1].pop()

    for n in order:
        ins = [signal(l) for l in nor_ins[n]]
        assert all(not isinstance(s, Const) for s in ins)
        w = net.nor(*ins)
        p, q = uses[n]
        pos, negs = distribute(net, w, p, q)
        pools[n] = (pos, negs)
    for name, v in outputs:
        for i, l in enumerate(v):
            net.outputs.append((name, i, signal(l)))
    net.check()
    return net
