"""Run a compiled layout in the simulator and read its outputs."""
from __future__ import annotations

from dataclasses import dataclass

from .. import gliders as G
from ..life import Universe
from .layout import Layout


@dataclass
class RunResult:
    outputs: dict          # what the Life pattern computed
    expected: dict         # what the netlist / Python computed
    clean: bool            # only the expected output gliders are still flying, no junk left
    problems: list
    generations: int

    @property
    def ok(self) -> bool:
        return self.clean and self.outputs == self.expected


def run(layout: Layout, values: dict, strict: bool = True) -> RunResult:
    u = Universe(layout.cells(values))
    u.step(layout.t_end)
    cells = u.cells
    outs = layout.read_outputs(cells, u.gen)
    exp = layout.netlist.evaluate(values)
    problems = []
    if strict:
        trajs, rest = G.identify(cells, u.gen)
        want = layout.expected_gliders(values)
        if trajs != want:
            extra = sorted(set(trajs) - set(want))
            missing = sorted(set(want) - set(trajs))
            problems.append(f"gliders: extra={[str(t) for t in extra]} missing={[str(t) for t in missing]}")
        if len(rest) and not G.is_still(rest):
            problems.append("leftover debris is not still")
    return RunResult(outs, exp, not problems, problems, u.gen)
