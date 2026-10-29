"""Compile a small Python function into a Game of Life pattern.

    from gol.compiler import compile_function
    prog = compile_function(src_or_function)
    cells = prog.cells(a=3, b=5)          # initial pattern for these inputs
    res = prog.run(a=3, b=5)              # simulate and read the outputs
"""
from __future__ import annotations

from dataclasses import dataclass

from . import layout as _layout
from . import techmap, verify
from .frontend import CompileError, to_aig
from .netlist import Netlist

__all__ = ["compile_function", "Program", "CompileError"]


@dataclass
class Program:
    name: str
    inputs: list            # [(name, width)]
    outputs: list           # [(name, width)]
    netlist: Netlist
    layout: _layout.Layout

    def _values(self, kw):
        missing = [n for n, _ in self.inputs if n not in kw]
        if missing:
            raise ValueError(f"missing inputs: {missing}")
        return {n: int(kw[n]) & ((1 << w) - 1) for n, w in self.inputs}

    def cells(self, **kw):
        return self.layout.cells(self._values(kw))

    def rle(self, **kw) -> str:
        v = self._values(kw)
        return self.layout.rle(v, f"{self.name}: compiled by gol.compiler")

    def run(self, strict: bool = True, **kw) -> verify.RunResult:
        return verify.run(self.layout, self._values(kw), strict)

    def result(self, outs: dict):
        """Outputs dict -> value in the same shape as the Python function returns."""
        if len(self.outputs) == 1 and self.outputs[0][0] == "out":
            return outs.get("out", 0)
        return tuple(outs.get(n, 0) for n, _ in self.outputs)

    def stats(self) -> dict:
        s = self.netlist.stats()
        cells = self.layout.cells({n: 0 for n, _ in self.inputs})
        span = cells.max(0) - cells.min(0) + 1 if len(cells) else (0, 0)
        s.update(gliders=len(self.layout.gliders), statics=len(self.layout.statics),
                 width=int(span[0]), height=int(span[1]), generations=self.layout.t_end)
        return s


def compile_function(src_or_fn, name: str | None = None) -> Program:
    g, inputs, outputs = to_aig(src_or_fn, name)
    net = techmap.to_netlist(g, inputs, outputs)
    lay = _layout.build(net)
    fname = name or getattr(src_or_fn, "__name__", "program")
    return Program(fname, inputs, [(n, len(v)) for n, v in outputs], net, lay)
