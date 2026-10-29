"""Turn a Netlist into a Game of Life pattern of one-shot glider logic.

Geometry (see CLAUDE.md for the full story):
  * A *row* is a south-east glider on lane r = x - y; a *column* is a
    south-west glider on lane c = x + y. Rows carry signals (glider present = 1).
  * Gate k owns column c_k. A constant glider C_k comes down that column and
    crosses every row. Rows it must interact with have exactly the right
    timing; all other rows are >= 19 generations off and pass through.
  * timing is measured by theta = t - 4y (constant along a downward path).
    Gates get increasing theta slots theta_k (spacing D).
  * NOR: C_k is annihilated by any input row that carries a glider, so it
    survives iff all inputs are 0. Rows after the first get an eater for the
    case that C_k was already gone.
  * FAN: C_k meets the input row in a kickback reaction; the kickback glider is
    caught by a boat and becomes a copy row, which a second constant C'_k
    inverts. Both surviving constants carry NOT input.
  * A surviving constant is turned into a new row at the bottom of the bus by
    a 3-reflector delay loop (down, right, up, right). The length of the upward
    leg sets the theta of the new row exactly to what its consumer needs.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .. import gliders as G
from .. import patterns as P
from .. import reactions as R
from .netlist import Const, Lit, Netlist, Wire

# ---- layout constants (lane units: rows r = x - y, columns c = x + y) ----
SR = 64        # spacing between rows
E = 48         # offset of the delay loop's upward leg from its column
EAT = 8        # eater sits this many cells (in x) past the crossing
W = 120        # spacing between gate columns
WF = 88        # FAN: offset of the second column
D = 136        # theta spacing between gate slots
RISE_MIN = 13  # minimal length (cells) of the upward leg of a delay loop


def _kill_table():
    t = {}
    for r in R.PAIRS:
        if r.name != "K":
            continue
        a, b = r.inputs
        if {a.dir, b.dir} != {"SE", "SW"}:
            continue
        se, sw = (a, b) if a.dir == "SE" else (b, a)
        t[(R.theta(sw) - R.theta(se), (se.lane + sw.lane) % 2)] = r
    return t


KILL = _kill_table()   # (theta_col - theta_row, parity) -> reaction
KB = next(r for r in R.PAIRS if r.name == "KB" and r.outputs[0].dir == "NE"
          and {t.dir for t in r.inputs} == {"SE", "SW"})
_KB_ROW = next(t for t in KB.inputs if t.dir == "SE")
_KB_COL = next(t for t in KB.inputs if t.dir == "SW")
KB_DTHETA = R.theta(_KB_COL) - R.theta(_KB_ROW)          # 17
assert (_KB_ROW.lane + _KB_COL.lane) % 2 == 0
# Constant SE glider that annihilates the kickback glider of KB (relative to the
# canonical KB inputs). Found and verified by tools/search_fanout.py, which
# simulates both cases (input present / absent) with all gliders coming from
# far upstream. If the input is absent it passes and carries NOT input.
FAN_G2 = G.Traj("SE", 24, -3)
BOAT_SE_SW = R.reflectors("SE", "SW")[0]
BOAT_SW_SE = R.reflectors("SW", "SE")[0]
G2_FEED = 24   # FAN_G2 is fed in as a column that a boat turns SE this far left of c_k
_b = R.place_single(BOAT_SE_SW, FAN_G2, 0)
BOAT_SE_SW_DTH = R.theta(_b.outputs[0]) - R.theta(FAN_G2)   # theta gain of that turn


def _loop_variants():
    """Delay-loop reflector triples (SW->SE, SE->NE, NE->SE) with their
    timing residue rho = (theta_out - theta_C) mod 8."""
    out = []
    kinds = ("boat", "eater1")
    C = G.Traj("SW", 0, 0)
    for k1, k2, k3 in itertools.product(kinds, repeat=3):
        r1 = R.reflectors("SW", "SE", (k1,))[0]
        r2 = R.reflectors("SE", "NE", (k2,))[0]
        r3 = R.reflectors("NE", "SE", (k3,))[0]
        a = R.place_single(r1, C, 0)
        b = R.place_single(r2, a.outputs[0], 0)
        c = R.place_single(r3, b.outputs[0], 0)
        rho = (R.theta(c.outputs[0]) - R.theta(C)) % 8
        out.append(((k1, k2, k3), (r1, r2, r3), rho))
    return out


LOOPS = _loop_variants()


@dataclass
class Row:
    traj: G.Traj
    lane: int
    consumer: object          # ("gate", k) or ("out", i)
    cond: object = None       # Lit for primary rows, None otherwise
    start_u: float = 0.0
    seg: int = -1


@dataclass
class Layout:
    statics: list = field(default_factory=list)      # [(cells, label)]
    gliders: list = field(default_factory=list)      # [(traj, cond)]  cond None = always
    outputs: list = field(default_factory=list)      # [(name, bit, traj or None, const)]
    labels: list = field(default_factory=list)       # [(x, y, text)]
    t_end: int = 0
    netlist: Netlist | None = None
    segs: list = field(default_factory=list)          # [(traj, x_from, x_to, label)] glider path pieces
    ok_pairs: set = field(default_factory=set)        # {(seg, seg)} meant to react
    ok_static: set = field(default_factory=set)       # {(seg, static index)} meant to react

    def cells(self, values: dict) -> np.ndarray:
        parts = [c for c, _ in self.statics]
        for traj, cond in self.gliders:
            if cond is None or _lit_value(cond, values):
                parts.append(traj.cells_at(0))
        return P.union(*parts)

    def rle(self, values: dict, title: str = "") -> str:
        from .. import rle
        com = [f"#C {title}"] if title else []
        com.append("#C inputs: " + ", ".join(f"{k}={v}" for k, v in values.items()))
        com.append(f"#C read the outputs at generation {self.t_end}")
        com += [f"#C LABEL {x} {y} {t}" for x, y, t in self.labels]
        return rle.dumps(self.cells(values), com)

    def expected_gliders(self, values: dict) -> list:
        exp = self.netlist.evaluate(values)
        out = []
        for name, bit, traj, const in self.outputs:
            if (exp[name] >> bit) & 1 and traj is not None:
                out.append(traj)
        return sorted(out)

    def read_outputs(self, cells: np.ndarray, t: int) -> dict:
        live = set(map(tuple, cells.tolist()))
        res: dict = {}
        for name, bit, traj, const in self.outputs:
            if traj is None:
                v = const
            else:
                v = int(all(tuple(c) in live for c in traj.cells_at(t).tolist()))
            res[name] = res.get(name, 0) | (v << bit)
        return res


def _lit_value(l: Lit, values: dict) -> bool:
    b = (values[l.name] >> l.bit) & 1
    return bool(b) if l.pol else not b


def _sw(lane, theta):
    return G.Traj("SW", lane, theta + 4 * lane)


def _se(lane, theta):
    return G.Traj("SE", lane, theta - 4 * lane)


def _kill_choice(parity: int, residue: int | None = None, even_only=False):
    """Pick a clean kill offset (theta_col - theta_row) for a lane parity.
    With `residue`, the offset must be congruent to it mod 8."""
    opts = sorted((k for (k, p) in KILL if p == parity), key=lambda k: (abs(abs(k) - 8), k))
    for k in opts:
        if residue is None or k % 8 == residue % 8:
            return k
    return None


class Builder:
    def __init__(self, net: Netlist):
        net.check()
        self.net = net
        self.L = Layout(netlist=net)
        self.rows: dict = {}      # key -> Row (alive)

    # ---------- pass 1: columns and theta slots ----------
    def schedule(self):
        net = self.net
        prod = {}                 # wire id -> (col lane, theta of producing constant)
        self.sched = []
        c, th = 0, 400
        for g in net.gates:
            s: dict = {}
            src = g.ins[0]
            if g.kind == "FAN":
                if isinstance(src, Wire):
                    pc, pth = prod[src.id]
                    if (c - pc) % 2:
                        c += 1
                    if (th - pth - KB_DTHETA) % 2:   # loop residues are even
                        th += 1
                s["c"], s["th"] = c, th
                prod[g.outs[0].id] = (c, th)
                self.sched.append(s)
                # second output: FAN_G2 survives, is turned down by a boat at column c2
                s["th2"] = th - KB_DTHETA + (R.theta(FAN_G2) - R.theta(_KB_ROW)) + BOAT_SE_SW_DTH
                s["c2"] = c + WF + (WF % 2)          # same parity as c (and as the input row)
                prod[g.outs[1].id] = (s["c2"], s["th2"])
                c = s["c2"] + W
                th = max(th, s["th2"]) + D
            else:
                s["c"], s["th"] = c, th
                prod[g.outs[0].id] = (c, th)
                self.sched.append(s)
                c += W
                th += D
        self.c_end = c
        self.th_final = th
        self.prod = prod

    # ---------- helpers ----------
    def _input_target(self, k: int, idx: int, src):
        """theta and lane parity that input idx of gate k must have."""
        g = self.net.gates[k]
        s = self.sched[k]
        if g.kind == "FAN":
            return s["th"] - KB_DTHETA, s["c"] % 2   # parity 0 => lane = c (mod 2)
        if isinstance(src, Lit):
            dk = _kill_choice(0)
            return s["th"] - dk, s["c"] % 2
        pc, pth = self.prod[src.id]
        parity = (pc + s["c"]) % 2
        # choose a kill offset whose residue a loop variant can produce
        for dk in sorted((kk for (kk, p) in KILL if p == parity), key=lambda kk: (abs(abs(kk) - 8), kk)):
            target = s["th"] - dk
            if any((target - pth) % 8 == rho for *_, rho in LOOPS):
                return target, pc % 2
        raise RuntimeError("no kill timing fits")

    def _seg(self, traj, xa, xb, label) -> int:
        self.L.segs.append((traj, float(xa), float(xb), label))
        return len(self.L.segs) - 1

    def _end_seg(self, i, xb):
        t, xa, _, lab = self.L.segs[i]
        self.L.segs[i] = (t, xa, float(xb), lab)

    def _static(self, cells, label, seg=None) -> int:
        self.L.statics.append((cells, label))
        i = len(self.L.statics) - 1
        for sg in ([seg] if isinstance(seg, int) else seg or []):
            self.L.ok_static.add((sg, i))
        return i

    def _lowest_lane(self):
        return min((r.lane for r in self.rows.values()), default=0)

    def _loop(self, col: G.Traj, col_seg: int, lane: int, theta: int, label: str):
        """Delay loop from constant `col` (SW) to a new row with given lane and theta."""
        best = None
        for names, (r1, r2, r3), rho in LOOPS:
            if (theta - R.theta(col)) % 8 != rho:
                continue

            def build(d1, d2, d3):
                a = R.place_single(r1, col, d1)
                b = R.place_single(r2, a.outputs[0], d2)
                cc = R.place_single(r3, b.outputs[0], d3)
                return a, b, cc

            a, b, cc = build(0, 0, 0)
            T = cc.outputs[0]
            L0, TH0 = T.lane, R.theta(T)
            delta = theta - TH0
            assert delta % 8 == 0
            aa = delta // 8                     # = d3 - d2
            # lane = L0 + 2 d1 - 2 d2 + 2 d3 = L0 + 2 d1 + 2 aa
            assert (lane - L0) % 2 == 0
            d1 = (lane - L0) // 2 - aa
            # choose d2 so that the up leg sits at u = col.lane + E
            def up_u(d2):
                _, b2, _ = build(d1, d2, d2 + aa)
                return -b2.outputs[0].lane
            u0, u1 = up_u(0), up_u(1)
            d2 = round((col.lane + E - u0) / (u1 - u0)) if u1 != u0 else 0
            a, b, cc = build(d1, d2, d2 + aa)
            T = cc.outputs[0]
            assert T.lane == lane and R.theta(T) == theta, (T, lane, theta)
            rise = b.statics[:, 1].mean() - cc.statics[:, 1].mean()
            if rise < RISE_MIN:
                continue
            cand = (names.count("eater1"), (a, b, cc), T)
            if best is None or cand[0] < best[0]:
                best = cand
        if best is None:
            raise RuntimeError(f"delay loop impossible for {label} (theta {theta} vs {R.theta(col)})")
        _, (a, b, cc), T = best
        xs = [p.statics[:, 0].mean() for p in (a, b, cc)]
        self._end_seg(col_seg, xs[0])
        s1 = self._seg(a.outputs[0], xs[0], xs[1], f"{label}:loop-right")
        s2 = self._seg(b.outputs[0], xs[1], xs[2], f"{label}:loop-up")
        s3 = self._seg(T, xs[2], xs[2], f"{label}:row")
        self.L.ok_pairs.update({(col_seg, s1), (s1, s2), (s2, s3)})
        self._static(a.statics, f"{label}:loop1", [col_seg, s1])
        self._static(b.statics, f"{label}:loop2", [s1, s2])
        self._static(cc.statics, f"{label}:loop3", [s2, s3])
        return T, -b.outputs[0].lane, s3

    def _eater(self, row: Row, col_lane: int, label: str):
        ea = R.eaters("SE")[0]
        x_cross = (row.lane + col_lane) / 2
        r0 = R.place_single(ea, row.traj, 0)
        dx = int(round(x_cross + EAT - r0.statics[:, 0].mean()))
        r1 = R.place_single(ea, row.traj, dx)
        self._end_seg(row.seg, r1.statics[:, 0].mean())
        self._static(r1.statics, label, row.seg)

    # ---------- pass 2: geometry ----------
    def build(self) -> Layout:
        self.schedule()
        net, L = self.net, self.L
        consumer = {}
        for k, g in enumerate(net.gates):
            for i, s in enumerate(g.ins):
                if isinstance(s, Wire):
                    consumer[s.id] = ("gate", k, i)
        for i, (_, _, s) in enumerate(net.outputs):
            if isinstance(s, Wire):
                consumer[s.id] = ("out", i)

        # primary rows: one per literal use, plus literal outputs; put the ones
        # consumed first at the bottom so their lanes free up early
        prim = []
        for k, g in enumerate(net.gates):
            for i, s in enumerate(g.ins):
                if isinstance(s, Lit):
                    prim.append((k, i, s))
        lit_outs = [(i, s) for i, (_, _, s) in enumerate(net.outputs) if isinstance(s, Lit)]
        lane = 0
        top = []
        for i, s in lit_outs:
            top.append((("out", i), s))
        for k, i, s in sorted(prim, key=lambda t: -t[0]):
            top.append((("gate", k, i), s))
        self.th_out = self.th_final + D
        for key, lit in top:
            if key[0] == "out":
                th, par = self.th_out, 0
            else:
                th, par = self._input_target(key[1], key[2], lit)
            if (lane - par) % 2:
                lane -= 1
            traj = _se(lane, th)
            sg = self._seg(traj, traj.pos(0)[0], traj.pos(0)[0], f"in {lit}")
            self.rows[key] = Row(traj, lane, key, lit, seg=sg)
            L.gliders.append((traj, lit))
            p = traj.pos(0)
            L.labels.append((int(p[0]), int(p[1]) - 3, f"in {lit}"))
            lane -= SR

        for k, g in enumerate(net.gates):
            s = self.sched[k]
            col = _sw(s["c"], s["th"])
            L.gliders.append((col, None))
            col_seg = self._seg(col, col.pos(0)[0], col.pos(0)[0], f"{g.kind}#{k}:C")
            low = self._lowest_lane()
            ins = [self.rows.pop(("gate", k, i)) for i in range(len(g.ins))]
            # rows are crossed from the top (highest lane) down
            ins.sort(key=lambda r: -r.lane)
            for r in ins:
                assert r.traj.lane % 2 == r.lane % 2
            xc, yc = (ins[0].lane + s["c"]) / 2, (s["c"] - ins[0].lane) / 2
            L.labels.append((int(xc) + 4, int(yc), f"{g.kind}#{k}"))
            if g.kind == "NOR":
                for j, r in enumerate(ins):
                    rx = R.place_pair(KILL[(R.theta(col) - R.theta(r.traj), (r.lane + s["c"]) % 2)], r.traj, col)
                    assert rx is not None, "kill timing mismatch"
                    xc = (r.lane + s["c"]) / 2
                    self._end_seg(r.seg, xc)
                    L.ok_pairs.add((r.seg, col_seg))
                    if j > 0:
                        self._eater(r, s["c"], f"NOR#{k}:eater")
                outs = [(col, g.outs[0], col_seg)]
            else:
                r = ins[0]
                kb = R.place_pair(KB, r.traj, col)
                assert kb is not None, "kickback timing mismatch"
                xc = (r.lane + s["c"]) / 2
                self._end_seg(r.seg, xc)
                L.ok_pairs.add((r.seg, col_seg))
                kick = kb.outputs[0]
                # second constant, shifted like the kickback, then a boat turns it down
                g2 = FAN_G2.shifted(*_match_shift(KB, kb))
                # feed g2 from a column just left of c_k (a glider coming from far
                # up-left along its lane would cross older delay loops)
                feed = _place_by_output(BOAT_SW_SE, g2, s["c"] - G2_FEED)
                L.gliders.append((feed.inputs[0], None))
                fx = feed.statics[:, 0].mean()
                fseg = self._seg(feed.inputs[0], feed.inputs[0].pos(0)[0], fx, f"FAN#{k}:feed")
                g2seg = self._seg(g2, fx, fx, f"FAN#{k}:G2")
                self._static(feed.statics, f"FAN#{k}:feed", [fseg, g2seg])
                L.ok_pairs.add((fseg, g2seg))
                # kickback glider runs from the collision to where G2 kills it
                kx = (kick.lane * -1 + g2.lane) / 2       # NE lane = -x-y, SE lane = x-y
                kseg = self._seg(kick, xc, kx, f"FAN#{k}:kick")
                L.ok_pairs.update({(kseg, g2seg), (kseg, col_seg), (kseg, r.seg)})
                b0 = R.place_single(BOAT_SE_SW, g2, 0)
                b1 = R.place_single(BOAT_SE_SW, g2, 1)
                step = b1.outputs[0].lane - b0.outputs[0].lane
                dx = (s["c2"] - b0.outputs[0].lane) // step
                boat = R.place_single(BOAT_SE_SW, g2, dx)
                col2 = boat.outputs[0]
                assert col2.lane == s["c2"] and R.theta(col2) == s["th2"], (col2, s)
                bx = boat.statics[:, 0].mean()
                self._end_seg(g2seg, bx)
                c2seg = self._seg(col2, bx, bx, f"FAN#{k}:C2")
                self._static(boat.statics, f"FAN#{k}:turn", [g2seg, c2seg])
                L.ok_pairs.add((g2seg, c2seg))
                outs = [(col, g.outs[0], col_seg), (col2, g.outs[1], c2seg)]
            for cl, w, cseg in outs:
                key = consumer[w.id]
                if key[0] == "gate":
                    th, par = self._input_target(key[1], key[2], w)
                else:
                    th = self.th_out
                    th += (R.theta(cl) - th) % 2   # keep residues even
                    par = cl.lane % 2
                lane = min(low, self._lowest_lane()) - SR
                if (lane - par) % 2:
                    lane -= 1
                T, u_up, rseg = self._loop(cl, cseg, lane, th, f"{g.kind}#{k}")
                self.rows[key] = Row(T, lane, key, None, u_up, rseg)
        # outputs
        c_read = self.c_end + W
        t_end = 0
        for i, (name, bit, s) in enumerate(net.outputs):
            if isinstance(s, Const):
                L.outputs.append((name, bit, None, s.value))
                continue
            row = self.rows.pop(("out", i))
            x = (row.lane + c_read) / 2
            self._end_seg(row.seg, x)
            t = int(np.ceil(row.traj.time_at(x)))
            t_end = max(t_end, t)
            L.outputs.append((name, bit, row.traj, None))
        L.t_end = t_end + 4
        for name, bit, traj, const in L.outputs:
            if traj is not None:
                px, py = traj.pos(L.t_end)
                L.labels.append((int(px) + 4, int(py), f"out {name}[{bit}]"))
        return L


def _place_by_output(r: R.Reaction, outgoing: G.Traj, u: float) -> R.Reaction:
    """Translate single-input reaction r so that its output equals `outgoing`,
    with the static part near x + y = u."""
    o = r.outputs[0]
    sx, sy = P.DIRS[o.dir]
    assert sx * sy == 1, "u varies along the path only for SE/NW outputs"

    def dy_of(dx):
        return sx * (o.lane + sy * dx - outgoing.lane)

    base = r.statics[:, 0].mean() + r.statics[:, 1].mean() + dy_of(0)
    dx = int(round((u - base) / 2))       # u grows by 2 per step along the lane
    dy = dy_of(dx)
    dt = outgoing.tref - o.tref + 4 * sx * dx
    out = r.shifted(dx, dy, dt)
    assert out.outputs[0] == outgoing
    return out


def _match_shift(canon: R.Reaction, inst: R.Reaction):
    """(dx, dy, dt) with canon.shifted(dx, dy, dt) == inst (both same orientation)."""
    a0 = canon.inputs[0]
    a1 = next(t for t in inst.inputs if t.dir == a0.dir)
    b0 = canon.inputs[1]
    b1 = next(t for t in inst.inputs if t.dir == b0.dir)
    sxa, sya = P.DIRS[a0.dir]
    sxb, syb = P.DIRS[b0.dir]
    m = np.array([[sya, -sxa], [syb, -sxb]], float)
    dx, dy = np.linalg.solve(m, [a1.lane - a0.lane, b1.lane - b0.lane])
    dx, dy = int(round(dx)), int(round(dy))
    dt = a1.tref - a0.tref + 4 * sxa * dx
    assert canon.shifted(dx, dy, dt).inputs == inst.inputs or sorted(canon.shifted(dx, dy, dt).inputs) == sorted(inst.inputs)
    return dx, dy, dt


def build(net: Netlist) -> Layout:
    return Builder(net).build()
