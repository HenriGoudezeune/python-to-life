"""Each gate is compiled on its own and simulated for every input combination."""
import itertools

import pytest

from gol.compiler import drc
from gol.compiler import layout as LY
from gol.compiler import verify
from gol.compiler.netlist import Lit, Netlist


def net_not():
    n = Netlist(inputs=[("a", 1)])
    n.outputs = [("y", 0, n.nor(Lit("a", 0)))]
    return n


def net_nor(k):
    names = "abc"[:k]
    n = Netlist(inputs=[(c, 1) for c in names])
    n.outputs = [("y", 0, n.nor(*[Lit(c, 0) for c in names]))]
    return n


def net_fan():
    n = Netlist(inputs=[("a", 1)])
    w1, w2 = n.fan(Lit("a", 0))
    n.outputs = [("y", 0, w1), ("y", 1, w2)]
    return n


def net_chain():
    """Gate outputs feeding gates: NOR -> FAN -> two NORs."""
    n = Netlist(inputs=[("a", 1), ("b", 1)])
    w = n.nor(Lit("a", 0), Lit("b", 0))
    w1, w2 = n.fan(w)
    n.outputs = [("y", 0, n.nor(w1, Lit("b", 0, False))), ("y", 1, n.nor(w2))]
    return n


@pytest.mark.parametrize("make", [net_not, lambda: net_nor(2), lambda: net_nor(3), net_fan, net_chain],
                         ids=["NOT", "NOR2", "NOR3", "FAN", "chain"])
def test_gate_exhaustive(make):
    net = make()
    lay = LY.build(net)
    assert drc.check(lay) == []
    names = [n for n, _ in net.inputs]
    for vals in itertools.product([0, 1], repeat=len(names)):
        v = dict(zip(names, vals))
        r = verify.run(lay, v)
        assert r.ok, (v, r.outputs, r.expected, r.problems)
