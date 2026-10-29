"""Python subset -> and-inverter graph.

Supported (see README for details):
  def f(a: u4, b: u4, c: bool) -> u5:  parameters need a uN/bool annotation
  ints are unsigned bit vectors; + grows by one bit, * adds widths,
  - and ~ wrap within the operand width, & | ^ << >> // % (by powers of two),
  comparisons, and/or/not, x if c else y, min/max/abs/int/bool,
  assignments (also a, b = ...; +=, ...), if/elif/else, for i in range(...)
  with constant bounds (unrolled), return anywhere (also early returns).
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap

from . import aig as A
from .aig import AIG, FALSE, TRUE, neg

MAXW = 64


class CompileError(Exception):
    pass


def _width_of_annotation(node) -> int | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        name = node.value
    elif isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    else:
        raise CompileError(f"unsupported annotation {ast.dump(node)}")
    if name == "bool":
        return 1
    m = re.fullmatch(r"u(\d+)", name)
    if not m:
        raise CompileError(f"annotation must be uN or bool, got {name}")
    return int(m.group(1))


class Frontend:
    def __init__(self, fn: ast.FunctionDef):
        self.fn = fn
        self.g = AIG()
        self.env: dict = {}
        self.ret = None
        self.returned = FALSE
        self.inputs: list = []

    # ---------- helpers ----------
    def err(self, node, msg):
        raise CompileError(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def truthy(self, v: list) -> int:
        r = FALSE
        for b in v:
            r = self.g.OR(r, b)
        return r

    def assign(self, name: str, val: list, guard: int):
        if guard == TRUE or name not in self.env:
            self.env[name] = val if guard == TRUE else A.mux_vec(self.g, guard, val, [FALSE])
        else:
            self.env[name] = A.mux_vec(self.g, guard, val, self.env[name])

    @staticmethod
    def trim(v: list) -> list:
        """Drop constant-zero high bits (keep at least one bit)."""
        v = list(v[:MAXW])
        while len(v) > 1 and v[-1] == FALSE:
            v.pop()
        return v

    def const_int(self, node) -> int:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.const_int(node.operand)
        v = self.expr(node)
        if any(b not in (FALSE, TRUE) for b in v):
            self.err(node, "expected a compile-time constant")
        return sum((b == TRUE) << i for i, b in enumerate(v))

    # ---------- statements ----------
    def compile(self):
        args = self.fn.args
        if args.vararg or args.kwarg or args.kwonlyargs:
            self.err(self.fn, "only plain positional parameters are supported")
        for a in args.args:
            w = _width_of_annotation(a.annotation)
            if w is None:
                self.err(a, f"parameter {a.arg} needs an annotation like u4 or bool")
            self.inputs.append((a.arg, w))
            self.env[a.arg] = [self.g.input(a.arg, i) for i in range(w)]
        self.block(self.fn.body, TRUE)
        if self.ret is None:
            self.err(self.fn, "function never returns a value")
        return self.outputs()

    def outputs(self):
        ann = self.fn.returns
        vals = self.ret
        if isinstance(vals, tuple):
            widths = [None] * len(vals)
            if isinstance(ann, ast.Tuple):
                widths = [_width_of_annotation(e) for e in ann.elts]
            elif isinstance(ann, ast.Subscript):   # tuple[u4, u1]
                elts = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
                widths = [_width_of_annotation(e) for e in elts]
            if len(widths) != len(vals):
                raise CompileError(f"return annotation has {len(widths)} elements, function returns {len(vals)}")
            outs = []
            for i, (v, w) in enumerate(zip(vals, widths)):
                outs.append((f"out{i}", A.extend(v, w) if w else v))
            return outs
        w = _width_of_annotation(ann) if ann is not None else None
        return [("out", A.extend(vals, w) if w else vals)]

    def block(self, stmts, guard):
        for st in stmts:
            eff = self.g.AND(guard, neg(self.returned))
            if eff == FALSE:
                return
            self.stmt(st, eff)

    def stmt(self, st, guard):
        g = self.g
        if isinstance(st, (ast.Pass, ast.Expr)) and not isinstance(getattr(st, "value", None), ast.Call):
            return
        if isinstance(st, ast.Assign):
            val = self.value(st.value)
            for t in st.targets:
                self.bind(t, val, guard)
        elif isinstance(st, ast.AnnAssign):
            val = self.expr(st.value)
            w = _width_of_annotation(st.annotation)
            self.bind(st.target, A.extend(val, w), guard)
        elif isinstance(st, ast.AugAssign):
            if not isinstance(st.target, ast.Name):
                self.err(st, "augmented assignment only to plain names")
            val = self.binop(st.op, self.expr(st.target), self.expr(st.value), st)
            self.assign(st.target.id, val, guard)
        elif isinstance(st, ast.If):
            c = self.truthy(self.expr(st.test))
            self.block(st.body, g.AND(guard, c))
            self.block(st.orelse, g.AND(guard, neg(c)))
        elif isinstance(st, ast.For):
            if st.orelse:
                self.err(st, "for/else is not supported")
            if not (isinstance(st.iter, ast.Call) and isinstance(st.iter.func, ast.Name)
                    and st.iter.func.id == "range"):
                self.err(st, "only 'for x in range(...)' with constant bounds is supported")
            rng = range(*[self.const_int(a) for a in st.iter.args])
            if len(rng) > 4096:
                self.err(st, "loop too long to unroll")
            if not isinstance(st.target, ast.Name):
                self.err(st, "loop variable must be a plain name")
            var = st.target.id
            old = self.env.get(var)
            for i in rng:
                # the loop variable is a compile-time constant in every iteration
                self.env[var] = A.const_vec(i)
                self.block(st.body, guard)
            if len(rng) and guard != TRUE and old is not None:
                self.env[var] = A.mux_vec(self.g, guard, self.env[var], old)
        elif isinstance(st, ast.Return):
            val = self.value(st.value) if st.value is not None else None
            if val is None:
                self.err(st, "return needs a value")
            if self.ret is None:
                self.ret = val
                if guard != TRUE:
                    self.ret = self._mux_ret(guard, val, None)
            else:
                self.ret = self._mux_ret(guard, val, self.ret)
            self.returned = g.OR(self.returned, guard)
        else:
            self.err(st, f"unsupported statement {type(st).__name__}")

    def _mux_ret(self, s, a, b):
        if isinstance(a, tuple) or isinstance(b, tuple):
            if b is None:
                b = tuple([FALSE] for _ in a)
            if not (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)):
                raise CompileError("all returns must have the same shape")
            return tuple(A.mux_vec(self.g, s, x, y) for x, y in zip(a, b))
        return A.mux_vec(self.g, s, a, b if b is not None else [FALSE])

    def bind(self, target, val, guard):
        if isinstance(target, ast.Name):
            if isinstance(val, tuple):
                self.err(target, "cannot bind a tuple to one name")
            self.assign(target.id, val, guard)
        elif isinstance(target, ast.Tuple):
            if not isinstance(val, tuple) or len(val) != len(target.elts):
                self.err(target, "tuple unpacking size mismatch")
            for t, v in zip(target.elts, val):
                self.bind(t, v, guard)
        else:
            self.err(target, "unsupported assignment target")

    # ---------- expressions ----------
    def value(self, node):
        """Expression that may be a tuple."""
        if isinstance(node, ast.Tuple):
            return tuple(self.expr(e) for e in node.elts)
        return self.expr(node)

    def expr(self, node) -> list:
        g = self.g
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return [TRUE if node.value else FALSE]
            if isinstance(node.value, int) and node.value >= 0:
                return A.const_vec(node.value)
            self.err(node, f"unsupported constant {node.value!r} (only non-negative ints and bools)")
        if isinstance(node, ast.Name):
            if node.id in ("True", "False"):
                return [TRUE if node.id == "True" else FALSE]
            if node.id not in self.env:
                self.err(node, f"unknown name {node.id}")
            return self.env[node.id]
        if isinstance(node, ast.BinOp):
            return self.binop(node.op, self.expr(node.left), self.expr(node.right), node)
        if isinstance(node, ast.UnaryOp):
            v = self.expr(node.operand)
            if isinstance(node.op, ast.Invert):
                return [neg(b) for b in v]
            if isinstance(node.op, ast.Not):
                return [neg(self.truthy(v))]
            if isinstance(node.op, ast.USub):
                if isinstance(node.operand, ast.Constant):
                    self.err(node, "negative constants are not supported (ints are unsigned)")
                return A.sub(g, [FALSE], v, len(v))
            if isinstance(node.op, ast.UAdd):
                return v
        if isinstance(node, ast.BoolOp):
            r = TRUE if isinstance(node.op, ast.And) else FALSE
            for e in node.values:
                t = self.truthy(self.expr(e))
                r = g.AND(r, t) if isinstance(node.op, ast.And) else g.OR(r, t)
            return [r]
        if isinstance(node, ast.Compare):
            r = TRUE
            left = self.expr(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = self.expr(right_node)
                r = g.AND(r, self.compare(op, left, right, node))
                left = right
            return [r]
        if isinstance(node, ast.IfExp):
            c = self.truthy(self.expr(node.test))
            return A.mux_vec(g, c, self.expr(node.body), self.expr(node.orelse))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            f = node.func.id
            args = [self.expr(a) for a in node.args]
            if f in ("min", "max") and len(args) >= 2:
                r = args[0]
                for a in args[1:]:
                    lt = A.less_than(g, a, r)
                    r = A.mux_vec(g, lt, a, r) if f == "min" else A.mux_vec(g, lt, r, a)
                return r
            if f in ("abs", "int") and len(args) == 1:
                return args[0]
            if f == "bool" and len(args) == 1:
                return [self.truthy(args[0])]
            self.err(node, f"unsupported call {f}")
        self.err(node, f"unsupported expression {type(node).__name__}")

    def compare(self, op, a, b, node) -> int:
        g = self.g
        if isinstance(op, ast.Eq):
            return A.equal(g, a, b)
        if isinstance(op, ast.NotEq):
            return neg(A.equal(g, a, b))
        if isinstance(op, ast.Lt):
            return A.less_than(g, a, b)
        if isinstance(op, ast.GtE):
            return neg(A.less_than(g, a, b))
        if isinstance(op, ast.Gt):
            return A.less_than(g, b, a)
        if isinstance(op, ast.LtE):
            return neg(A.less_than(g, b, a))
        self.err(node, f"unsupported comparison {type(op).__name__}")

    def binop(self, op, a, b, node) -> list:
        g = self.g
        if isinstance(op, ast.Add):
            return self.trim(A.add(g, a, b, width=min(MAXW, max(len(a), len(b)) + 1)))
        if isinstance(op, ast.Sub):
            return A.sub(g, a, b)
        if isinstance(op, ast.Mult):
            return self.trim(A.mul(g, a, b, min(MAXW, len(a) + len(b))))
        if isinstance(op, ast.BitAnd):
            w = min(len(a), len(b))
            return [g.AND(x, y) for x, y in zip(a[:w], b[:w])]
        if isinstance(op, ast.BitOr):
            w = max(len(a), len(b))
            return [g.OR(x, y) for x, y in zip(A.extend(a, w), A.extend(b, w))]
        if isinstance(op, ast.BitXor):
            w = max(len(a), len(b))
            return [g.XOR(x, y) for x, y in zip(A.extend(a, w), A.extend(b, w))]
        const_b = all(x in (FALSE, TRUE) for x in b)
        kb = sum((x == TRUE) << i for i, x in enumerate(b)) if const_b else None
        if isinstance(op, ast.LShift):
            if const_b:
                return self.trim([FALSE] * kb + a)
            w = min(MAXW, len(a) + (1 << len(b)) - 1)
            return self.trim(A.shift_left(g, a, b, w))
        if isinstance(op, ast.RShift):
            if const_b:
                return a[kb:] or [FALSE]
            return A.shift_right(g, a, b)
        if isinstance(op, (ast.FloorDiv, ast.Mod)):
            if not const_b or kb == 0 or kb & (kb - 1):
                self.err(node, "// and % only by constant powers of two")
            k = kb.bit_length() - 1
            if isinstance(op, ast.FloorDiv):
                return a[k:] or [FALSE]
            return a[:k] or [FALSE]
        self.err(node, f"unsupported operator {type(op).__name__}")


def parse_function(src_or_fn, name: str | None = None) -> ast.FunctionDef:
    if callable(src_or_fn):
        src = textwrap.dedent(inspect.getsource(src_or_fn))
        name = name or src_or_fn.__name__
    else:
        src = src_or_fn
    tree = ast.parse(src)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if name:
        fns = [f for f in fns if f.name == name]
    if not fns:
        raise CompileError(f"function {name or ''} not found")
    return fns[-1] if name is None else fns[0]


def to_aig(src_or_fn, name: str | None = None):
    """Returns (aig, inputs [(name, width)], outputs [(name, [literals])])."""
    fn = parse_function(src_or_fn, name)
    fe = Frontend(fn)
    outs = fe.compile()
    return fe.g, fe.inputs, outs
