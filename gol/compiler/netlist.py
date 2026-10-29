"""Gate-level netlist that the layout engine turns into a Life pattern.

Only two gate kinds exist, because they map one-to-one onto glider reactions:
  NOR(a, b, ...) -> 1 output   (a constant glider crossing the input rows)
  FAN(a)         -> 2 outputs, both equal to NOT a   (kickback + NOT)

Signals:
  Lit(name, bit, pol)  primary input bit (pol=False means the complemented bit);
                       a literal may be used any number of times (one glider per use)
  Const(v)             constant 0/1 (only allowed as a circuit output)
  Wire(id)             output of a gate; must be used exactly once
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lit:
    name: str
    bit: int
    pol: bool = True

    def __str__(self):
        return f"{'' if self.pol else '~'}{self.name}[{self.bit}]"


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class Wire:
    id: int

    def __str__(self):
        return f"w{self.id}"


@dataclass
class Gate:
    kind: str            # "NOR" or "FAN"
    ins: list
    outs: list


@dataclass
class Netlist:
    inputs: list = field(default_factory=list)    # [(name, width)]
    gates: list = field(default_factory=list)     # topological order
    outputs: list = field(default_factory=list)   # [(name, bit, signal)]
    _next: int = 0

    def wire(self) -> Wire:
        self._next += 1
        return Wire(self._next)

    def nor(self, *ins) -> Wire:
        w = self.wire()
        self.gates.append(Gate("NOR", list(ins), [w]))
        return w

    def fan(self, a) -> tuple[Wire, Wire]:
        w1, w2 = self.wire(), self.wire()
        self.gates.append(Gate("FAN", [a], [w1, w2]))
        return w1, w2

    def check(self):
        uses: dict = {}
        defined = set()
        for g in self.gates:
            for s in g.ins:
                if isinstance(s, Wire):
                    assert s.id in defined, f"{s} used before defined"
                    uses[s.id] = uses.get(s.id, 0) + 1
                else:
                    assert isinstance(s, Lit), f"gate input must be Lit or Wire, got {s}"
            defined.update(w.id for w in g.outs)
        for _, _, s in self.outputs:
            if isinstance(s, Wire):
                uses[s.id] = uses.get(s.id, 0) + 1
        bad = {w: n for w, n in uses.items() if n != 1}
        assert not bad, f"wires must be used exactly once: {bad}"
        unused = defined - set(uses)
        assert not unused, f"unused wires: {unused}"

    def evaluate(self, values: dict) -> dict:
        """Reference evaluation: values maps input name -> int. Returns {name: int}."""
        env: dict = {}

        def val(s):
            if isinstance(s, Lit):
                b = (values[s.name] >> s.bit) & 1
                return b if s.pol else 1 - b
            if isinstance(s, Const):
                return s.value
            return env[s.id]

        for g in self.gates:
            if g.kind == "NOR":
                v = 0 if any(val(s) for s in g.ins) else 1
                env[g.outs[0].id] = v
            else:
                v = 1 - val(g.ins[0])
                for w in g.outs:
                    env[w.id] = v
        out: dict = {}
        for name, bit, s in self.outputs:
            out[name] = out.get(name, 0) | (val(s) << bit)
        return out

    def stats(self) -> dict:
        return {"NOR": sum(g.kind == "NOR" for g in self.gates),
                "FAN": sum(g.kind == "FAN" for g in self.gates)}
