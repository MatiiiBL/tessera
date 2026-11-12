"""Integer arithmetic on index expressions: simplification and bounds.

The simplifier rewrites integer expressions into a canonical linear form,
``c0 + c1*t1 + c2*t2 + ...``, where each term ``t`` is a variable or a
non-linear sub-expression (``floordiv``, ``floormod``, ``min``, a load...).
Like terms combine, so ``(i*4 + j) - i*4`` becomes ``j``. ``floordiv`` and
``floormod`` by positive constants pull out the multiples of the divisor
(``(i*8 + j) // 4 -> i*2 + j // 4``) and fold away when the bounds of the
rest are known (``j // 4 -> 0`` for ``j`` in ``[0, 4)``); ``x // c * c + x % c``
recombines into ``x``. Comparisons are normalized to ``L < c``, ``L >= c``,
``L == c`` or ``L != c`` and decided when the bounds allow it, which is how
predicates of imperfect splits disappear when they always hold.

Floating-point expressions are only constant-folded (rounded to their type)
and stripped of ``x + 0.0``, ``x * 1.0`` and ``x / 1.0``; they are never
reassociated, since that would change rounding.

Bounds come in two kinds. ``Analyzer.bound`` gives a constant interval of an
expression from the intervals of its variables. ``Analyzer.relax`` gives
symbolic bounds: some variables range over their intervals while the others
stay as symbols, which is what region inference needs (the rows of a tile
are ``[i0*4, i0*4 + 3]`` for the outer loop variable ``i0``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from tessera.dtype import INDEX, DType
from tessera.tir.expr import (
    AddressOf,
    Binary,
    BufferLoad,
    Call,
    Cast,
    Compare,
    CompareOp,
    Expr,
    FloatImm,
    IntImm,
    Logical,
    Not,
    Select,
    Var,
)
from tessera.tir.functor import uses_any
from tessera.tir.stmt import Range

TRUE = IntImm(1, DType.bool_)
FALSE = IntImm(0, DType.bool_)

_DTYPE_BOUNDS = {
    DType.bool_: (0, 1),
    DType.uint8: (0, 255),
    DType.int8: (-128, 127),
    DType.int32: (-(2**31), 2**31 - 1),
}


@dataclass(frozen=True, slots=True)
class Interval:
    """Integers ``lo..hi`` inclusive; ``None`` means unbounded on that side."""

    lo: int | None
    hi: int | None

    @classmethod
    def point(cls, v: int) -> Interval:
        return cls(v, v)

    @classmethod
    def everything(cls) -> Interval:
        return cls(None, None)

    @classmethod
    def of_range(cls, r: Range) -> Interval:
        """The values of a loop or iteration domain; unbounded unless constant."""
        if isinstance(r.min, IntImm) and isinstance(r.extent, IntImm):
            return cls(r.min.value, r.min.value + r.extent.value - 1)
        return cls.everything()

    @property
    def is_point(self) -> bool:
        return self.lo is not None and self.lo == self.hi

    @property
    def bounded(self) -> bool:
        return self.lo is not None and self.hi is not None

    def __add__(self, other: Interval) -> Interval:
        lo = None if self.lo is None or other.lo is None else self.lo + other.lo
        hi = None if self.hi is None or other.hi is None else self.hi + other.hi
        return Interval(lo, hi)

    def scale(self, k: int) -> Interval:
        if k == 0:
            return Interval(0, 0)
        lo = None if self.lo is None else self.lo * k
        hi = None if self.hi is None else self.hi * k
        return Interval(lo, hi) if k > 0 else Interval(hi, lo)

    def union(self, other: Interval) -> Interval:
        lo = None if self.lo is None or other.lo is None else min(self.lo, other.lo)
        hi = None if self.hi is None or other.hi is None else max(self.hi, other.hi)
        return Interval(lo, hi)

    def __mul__(self, other: Interval) -> Interval:
        if not (self.bounded and other.bounded):
            if self.is_point and self.lo is not None:
                return other.scale(self.lo)
            if other.is_point and other.lo is not None:
                return self.scale(other.lo)
            return Interval.everything()
        assert self.lo is not None
        assert self.hi is not None
        assert other.lo is not None
        assert other.hi is not None
        products = [a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(min(products), max(products))


class _Lin:
    """``const + sum(coef * term)``; terms keep their order of first appearance."""

    __slots__ = ("const", "terms")

    def __init__(self, terms: dict[Expr, int] | None = None, const: int = 0) -> None:
        self.terms = terms if terms is not None else {}
        self.const = const

    @classmethod
    def atom(cls, e: Expr, k: int = 1) -> _Lin:
        return cls({e: k})

    @property
    def is_const(self) -> bool:
        return not self.terms

    def copy(self) -> _Lin:
        return _Lin(dict(self.terms), self.const)

    def add(self, other: _Lin, k: int = 1) -> _Lin:
        out = self.copy()
        for t, c in other.terms.items():
            v = out.terms.get(t, 0) + c * k
            if v:
                out.terms[t] = v
            else:
                out.terms.pop(t, None)
        out.const += other.const * k
        return out

    def scale(self, k: int) -> _Lin:
        if k == 0:
            return _Lin()
        return _Lin({t: c * k for t, c in self.terms.items()}, self.const * k)


def _is_int(dtype: DType) -> bool:
    return dtype.is_int or dtype == DType.bool_


class Analyzer:
    """Simplifies and bounds expressions given the ranges of their variables."""

    def __init__(self, ranges: Mapping[Var, Interval | Range] | None = None) -> None:
        self.ranges: dict[Var, Interval] = {}
        for v, r in (ranges or {}).items():
            self.bind(v, r)

    def bind(self, var: Var, r: Interval | Range) -> None:
        """Declare the values ``var`` takes (a loop range or an interval)."""
        if isinstance(r, Range):
            if isinstance(r.min, IntImm) and isinstance(r.extent, IntImm):
                r = Interval.of_range(r)
            else:
                lo = self.bound(r.min)
                ext = self.bound(r.extent)
                hi = None if lo.hi is None or ext.hi is None else lo.hi + ext.hi - 1
                r = Interval(lo.lo, hi)
        self.ranges[var] = r

    def unbind(self, var: Var) -> None:
        self.ranges.pop(var, None)

    def copy(self) -> Analyzer:
        out = Analyzer()
        out.ranges = dict(self.ranges)
        return out

    # -- simplification -----------------------------------------------------

    def simplify(self, e: Expr) -> Expr:
        """An equivalent, canonical expression."""
        if isinstance(e, IntImm | FloatImm | Var):
            return e
        if isinstance(e, Compare):
            return self._compare(e)
        if isinstance(e, Logical):
            return self._logical(e)
        if isinstance(e, Not):
            return self._not(e)
        if _is_int(e.dtype) and not isinstance(e, AddressOf):
            if e.dtype == DType.bool_:
                return self._bool_other(e)
            return self._from_lin(self._to_lin(e), e.dtype)
        return self._float(e)

    def can_prove(self, cond: Expr) -> bool:
        """True if ``cond`` holds for every value of its variables."""
        return self.simplify(cond) == TRUE

    def _bool_other(self, e: Expr) -> Expr:
        if isinstance(e, Select):
            c = self.simplify(e.cond)
            if isinstance(c, IntImm):
                return self.simplify(e.true_value if c.value else e.false_value)
            t, f = self.simplify(e.true_value), self.simplify(e.false_value)
            return t if t == f else Select(c, t, f)
        if isinstance(e, Cast):
            v = self.simplify(e.value)
            if isinstance(v, IntImm | FloatImm):
                return TRUE if v.value != 0 else FALSE
            if v.dtype == DType.bool_:
                return v
            return Cast(e.type, v)
        return self._rebuild(e)

    def _rebuild(self, e: Expr) -> Expr:
        """Simplify the children of a node the simplifier does not otherwise rewrite."""
        if isinstance(e, BufferLoad):
            idx = tuple(self.simplify(i) for i in e.indices)
            return BufferLoad(e.buffer, idx)
        if isinstance(e, AddressOf):
            return AddressOf(e.buffer, tuple(self.simplify(i) for i in e.indices))
        if isinstance(e, Call):
            return Call(e.name, tuple(self.simplify(a) for a in e.args), e.type)
        if isinstance(e, Cast):
            return Cast(e.type, self.simplify(e.value))
        if isinstance(e, Select):
            return Select(
                self.simplify(e.cond), self.simplify(e.true_value), self.simplify(e.false_value)
            )
        if isinstance(e, Binary):
            return Binary(e.op, self.simplify(e.a), self.simplify(e.b))
        return e

    # integers ---------------------------------------------------------------

    def _to_lin(self, e: Expr) -> _Lin:
        if isinstance(e, IntImm):
            return _Lin(const=e.value)
        if isinstance(e, Var):
            return _Lin.atom(e)
        if isinstance(e, Binary):
            return self._binary_lin(e)
        if isinstance(e, Select):
            c = self.simplify(e.cond)
            if isinstance(c, IntImm):
                return self._to_lin(e.true_value if c.value else e.false_value)
            t, f = self.simplify(e.true_value), self.simplify(e.false_value)
            if t == f:
                return self._to_lin(t)
            return _Lin.atom(Select(c, t, f))
        if isinstance(e, Cast):
            v = self.simplify(e.value)
            if isinstance(v, IntImm):
                return _Lin(const=v.value)
            if isinstance(v, FloatImm) and math.isfinite(v.value):
                return _Lin(const=int(v.value))
            return _Lin.atom(Cast(e.type, v))
        return _Lin.atom(self._rebuild(e))

    def _binary_lin(self, e: Binary) -> _Lin:
        op = e.op
        if op == "add":
            return self._to_lin(e.a).add(self._to_lin(e.b))
        if op == "sub":
            return self._to_lin(e.a).add(self._to_lin(e.b), -1)
        la, lb = self._to_lin(e.a), self._to_lin(e.b)
        dtype = e.dtype
        if op == "mul":
            if la.is_const:
                return lb.scale(la.const)
            if lb.is_const:
                return la.scale(lb.const)
            return _Lin.atom(Binary("mul", self._from_lin(la, dtype), self._from_lin(lb, dtype)))
        if op in ("floordiv", "floormod"):
            if lb.is_const and lb.const > 0:
                return (
                    self._floordiv(la, lb.const)
                    if op == "floordiv"
                    else self._floormod(la, lb.const, dtype)
                )
            if la.is_const and lb.is_const and lb.const != 0:
                v = la.const // lb.const if op == "floordiv" else la.const % lb.const
                return _Lin(const=v)
            return _Lin.atom(Binary(op, self._from_lin(la, dtype), self._from_lin(lb, dtype)))
        if op in ("min", "max"):
            if la.is_const and lb.is_const:
                return _Lin(
                    const=min(la.const, lb.const) if op == "min" else max(la.const, lb.const)
                )
            diff = self._bound_lin(la.add(lb, -1))
            if diff.hi is not None and diff.hi <= 0:
                return la if op == "min" else lb
            if diff.lo is not None and diff.lo >= 0:
                return lb if op == "min" else la
            return _Lin.atom(Binary(op, self._from_lin(la, dtype), self._from_lin(lb, dtype)))
        if op == "div" and la.is_const and lb.is_const and lb.const != 0:
            q = abs(la.const) // abs(lb.const)
            return _Lin(const=q if (la.const >= 0) == (lb.const > 0) else -q)
        if op == "pow" and la.is_const and lb.is_const and lb.const >= 0:
            return _Lin(const=la.const**lb.const)
        return _Lin.atom(Binary(op, self._from_lin(la, dtype), self._from_lin(lb, dtype)))

    def _floordiv(self, la: _Lin, c: int) -> _Lin:
        if c == 1:
            return la
        q, r = _Lin(), _Lin()
        for t, k in la.terms.items():
            if k % c == 0:
                q.terms[t] = k // c
            else:
                r.terms[t] = k
        q.const, r.const = la.const // c, la.const % c
        b = self._bound_lin(r)
        if b.lo is not None and b.hi is not None and b.lo // c == b.hi // c:
            q.const += b.lo // c
            return q
        g = math.gcd(c, r.const, *r.terms.values())
        if g > 1:
            r = _Lin({t: k // g for t, k in r.terms.items()}, r.const // g)
            c //= g
        if len(r.terms) == 1 and r.const == 0:
            ((t, k),) = r.terms.items()
            if (
                k == 1
                and isinstance(t, Binary)
                and t.op == "floordiv"
                and isinstance(t.b, IntImm)
                and t.b.value > 0
            ):
                return q.add(self._floordiv(self._to_lin(t.a), t.b.value * c))
        dtype = _lin_dtype(r)
        return q.add(_Lin.atom(Binary("floordiv", self._from_lin(r, dtype), IntImm(c, dtype))))

    def _floormod(self, la: _Lin, c: int, dtype: DType) -> _Lin:
        if c == 1:
            return _Lin()
        r = _Lin()
        for t, k in la.terms.items():
            k2 = k if 0 < k < c else k % c
            if k2:
                r.terms[t] = k2
        r.const = la.const % c
        b = self._bound_lin(r)
        if b.lo is not None and b.hi is not None and b.lo // c == b.hi // c:
            r.const -= (b.lo // c) * c
            return r
        g = math.gcd(c, r.const, *r.terms.values())
        if g > 1:
            inner = _Lin({t: k // g for t, k in r.terms.items()}, r.const // g)
            return self._floormod(inner, c // g, dtype).scale(g)
        if len(r.terms) == 1 and r.const == 0:
            ((t, k),) = r.terms.items()
            if (
                k == 1
                and isinstance(t, Binary)
                and t.op == "floormod"
                and isinstance(t.b, IntImm)
                and t.b.value > 0
                and t.b.value % c == 0
            ):
                return self._floormod(self._to_lin(t.a), c, dtype)
        dt = _lin_dtype(r, dtype)
        return _Lin.atom(Binary("floormod", self._from_lin(r, dt), IntImm(c, dt)))

    def _recombine(self, lin: _Lin) -> _Lin:
        """Rewrites ``k*c*(x // c) + k*(x % c)`` into ``k*x`` and
        ``k*x - k*c*(x // c)`` into ``k*(x % c)``."""
        changed = True
        while changed:
            changed = False
            for t, k in list(lin.terms.items()):
                if isinstance(t, Binary) and t.op == "floordiv" and isinstance(t.b, IntImm):
                    c = t.b.value
                    whole = lin.terms.get(t.a)
                    if c > 0 and whole is not None and k == -whole * c:
                        rest = lin.copy()
                        del rest.terms[t]
                        del rest.terms[t.a]
                        lin = rest.add(_Lin.atom(Binary("floormod", t.a, t.b)), whole)
                        changed = True
                        break
                if not (isinstance(t, Binary) and t.op == "floormod" and isinstance(t.b, IntImm)):
                    continue
                c = t.b.value
                partner = Binary("floordiv", t.a, t.b)
                if lin.terms.get(partner) == k * c:
                    rest = lin.copy()
                    del rest.terms[t]
                    del rest.terms[partner]
                    lin = rest.add(self._to_lin(t.a), k)
                    changed = True
                    break
        return lin

    def _from_lin(self, lin: _Lin, dtype: DType = INDEX) -> Expr:
        lin = self._recombine(lin)
        out: Expr | None = None
        for t, k in lin.terms.items():
            if out is None:
                out = t if k == 1 else Binary("mul", t, IntImm(k, dtype))
            elif k == 1:
                out = Binary("add", out, t)
            elif k == -1:
                out = Binary("sub", out, t)
            elif k > 0:
                out = Binary("add", out, Binary("mul", t, IntImm(k, dtype)))
            else:
                out = Binary("sub", out, Binary("mul", t, IntImm(-k, dtype)))
        if out is None:
            return IntImm(lin.const, dtype)
        if lin.const > 0:
            return Binary("add", out, IntImm(lin.const, dtype))
        if lin.const < 0:
            return Binary("sub", out, IntImm(-lin.const, dtype))
        return out

    # booleans ---------------------------------------------------------------

    def _compare(self, e: Compare) -> Expr:
        if not _is_int(e.a.dtype) or e.a.dtype == DType.bool_:
            a, b = self.simplify(e.a), self.simplify(e.b)
            if isinstance(a, IntImm | FloatImm) and isinstance(b, IntImm | FloatImm):
                return TRUE if _fold_compare(e.op, a.value, b.value) else FALSE
            return Compare(e.op, a, b)
        la, lb = self._to_lin(e.a), self._to_lin(e.b)
        op = e.op
        if op == "gt":
            op, d = "lt", lb.add(la, -1)
        elif op == "ge":
            op, d = "le", lb.add(la, -1)
        else:
            d = la.add(lb, -1)
        if op == "le":
            op, d = "lt", d.add(_Lin(const=-1))
        return self._canonical_compare(op, d, e.a.dtype)

    def _canonical_compare(self, op: CompareOp, d: _Lin, dtype: DType) -> Expr:
        """``d op 0`` with ``op`` in lt, eq, ne, as a normalized comparison."""
        b = self._bound_lin(d)
        if op == "lt":
            if b.hi is not None and b.hi < 0:
                return TRUE
            if b.lo is not None and b.lo >= 0:
                return FALSE
        elif op in ("eq", "ne"):
            if b.lo == 0 and b.hi == 0:
                return TRUE if op == "eq" else FALSE
            if (b.hi is not None and b.hi < 0) or (b.lo is not None and b.lo > 0):
                return FALSE if op == "eq" else TRUE
        lhs = _Lin(dict(d.terms))
        m = -d.const
        negate = next(iter(lhs.terms.values())) < 0
        if negate:
            lhs, m = lhs.scale(-1), -m
        g = math.gcd(*lhs.terms.values())
        if op == "lt":
            # L < m (or -L < -m, i.e. L > m', i.e. L >= m' + 1)
            if negate:
                out_op: CompareOp = "ge"
                m = m + 1
                bound = -((-m) // g)
            else:
                out_op = "lt"
                bound = -((-m) // g)
        else:
            out_op = op
            if m % g:
                return FALSE if op == "eq" else TRUE
            bound = m // g
        if g > 1:
            lhs = _Lin({t: k // g for t, k in lhs.terms.items()})
        return Compare(out_op, self._from_lin(lhs, dtype), IntImm(bound, dtype))

    def _logical(self, e: Logical) -> Expr:
        terms: list[Expr] = []
        for part in _flatten(e, e.op):
            s = self.simplify(part)
            for p in _flatten(s, e.op) if isinstance(s, Logical) else [s]:
                if isinstance(p, IntImm):
                    if bool(p.value) == (e.op == "or"):
                        return TRUE if e.op == "or" else FALSE
                    continue
                if p not in terms:
                    terms.append(p)
        if e.op == "and":
            terms = _merge_bounds(terms)
        if not terms:
            return TRUE if e.op == "and" else FALSE
        out = terms[0]
        for t in terms[1:]:
            out = Logical(e.op, out, t)
        return out

    def _not(self, e: Not) -> Expr:
        a = self.simplify(e.a)
        if isinstance(a, IntImm):
            return FALSE if a.value else TRUE
        if isinstance(a, Not):
            return a.a
        if isinstance(a, Compare) and _is_int(a.a.dtype) and a.a.dtype != DType.bool_:
            flipped: dict[str, CompareOp] = {
                "lt": "ge",
                "ge": "lt",
                "le": "gt",
                "gt": "le",
                "eq": "ne",
                "ne": "eq",
            }
            return self.simplify(Compare(flipped[a.op], a.a, a.b))
        return Not(a)

    # floats -----------------------------------------------------------------

    def _float(self, e: Expr) -> Expr:
        if isinstance(e, Binary):
            a, b = self.simplify(e.a), self.simplify(e.b)
            if isinstance(a, FloatImm) and isinstance(b, FloatImm):
                folded = _fold_float(e.op, a.value, b.value)
                if folded is not None:
                    return FloatImm(_round(folded, e.dtype), e.dtype)
            if e.op == "add":
                if _is_float_const(b, 0.0):
                    return a
                if _is_float_const(a, 0.0):
                    return b
            elif e.op == "sub" and _is_float_const(b, 0.0):
                return a
            elif e.op == "mul":
                if _is_float_const(b, 1.0):
                    return a
                if _is_float_const(a, 1.0):
                    return b
            elif e.op == "div" and _is_float_const(b, 1.0):
                return a
            return Binary(e.op, a, b)
        if isinstance(e, Select):
            c = self.simplify(e.cond)
            if isinstance(c, IntImm):
                return self.simplify(e.true_value if c.value else e.false_value)
            t, f = self.simplify(e.true_value), self.simplify(e.false_value)
            return t if t == f else Select(c, t, f)
        if isinstance(e, Cast):
            v = self.simplify(e.value)
            if v.dtype == e.type:
                return v
            if isinstance(v, IntImm | FloatImm) and e.type.is_float:
                return FloatImm(_round(float(v.value), e.type), e.type)
            return Cast(e.type, v)
        return self._rebuild(e)

    # -- bounds -------------------------------------------------------------

    def bound(self, e: Expr) -> Interval:
        """A constant interval containing every value of ``e``."""
        if not _is_int(e.dtype) or isinstance(e, AddressOf):
            return Interval.everything()
        if isinstance(e, Compare | Logical | Not):
            return Interval(0, 1)
        if e.dtype == DType.bool_:
            return Interval(0, 1)
        return self._bound_lin(self._to_lin(e))

    def _bound_lin(self, lin: _Lin) -> Interval:
        out = Interval.point(lin.const)
        for t, k in lin.terms.items():
            out = out + self._bound_atom(t).scale(k)
            if out.lo is None and out.hi is None:
                break
        return out

    def _bound_atom(self, t: Expr) -> Interval:
        if isinstance(t, IntImm):
            return Interval.point(t.value)
        if isinstance(t, Var):
            found = self.ranges.get(t)
            if found is not None:
                return found
            lo_hi = _DTYPE_BOUNDS.get(t.type)
            return Interval(*lo_hi) if lo_hi else Interval.everything()
        if isinstance(t, Binary):
            a = self.bound(t.a)
            if t.op in ("floordiv", "floormod") and isinstance(t.b, IntImm) and t.b.value > 0:
                c = t.b.value
                if t.op == "floordiv":
                    lo = None if a.lo is None else a.lo // c
                    hi = None if a.hi is None else a.hi // c
                    return Interval(lo, hi)
                if a.lo is not None and a.hi is not None and a.lo // c == a.hi // c:
                    return Interval(a.lo % c, a.hi % c)
                return Interval(0, c - 1)
            b = self.bound(t.b)
            if t.op == "mul":
                return a * b
            if t.op == "min":
                lo = None if a.lo is None or b.lo is None else min(a.lo, b.lo)
                hi = b.hi if a.hi is None else a.hi if b.hi is None else min(a.hi, b.hi)
                return Interval(lo, hi)
            if t.op == "max":
                lo = b.lo if a.lo is None else a.lo if b.lo is None else max(a.lo, b.lo)
                hi = None if a.hi is None or b.hi is None else max(a.hi, b.hi)
                return Interval(lo, hi)
            if t.op == "add":
                return a + b
            if t.op == "sub":
                return a + b.scale(-1)
            return Interval.everything()
        if isinstance(t, Select):
            return self.bound(t.true_value).union(self.bound(t.false_value))
        if isinstance(t, Cast) and _is_int(t.value.dtype) and _widens(t.value.dtype, t.type):
            return self.bound(t.value)
        lo_hi = _DTYPE_BOUNDS.get(t.dtype)
        return Interval(*lo_hi) if lo_hi else Interval.everything()

    def relax(self, e: Expr, relaxed: Mapping[Var, Interval]) -> tuple[Expr | None, Expr | None]:
        """Symbolic lower and upper bounds of ``e`` when the variables in ``relaxed``
        take every value of their intervals; other variables stay symbols.
        ``None`` means unbounded."""
        lin = self._to_lin(e)
        lo: _Lin | None = _Lin(const=lin.const)
        hi: _Lin | None = _Lin(const=lin.const)
        for t, k in lin.terms.items():
            if not uses_any(t, frozenset(relaxed)):
                term = _Lin.atom(t, k)
                lo = None if lo is None else lo.add(term)
                hi = None if hi is None else hi.add(term)
                continue
            tlo, thi = self._relax_atom(t, relaxed)
            if k < 0:
                tlo, thi = thi, tlo
            lo = None if lo is None or tlo is None else lo.add(self._to_lin(tlo), k)
            hi = None if hi is None or thi is None else hi.add(self._to_lin(thi), k)
        dtype = e.dtype
        return (
            None if lo is None else self._from_lin(lo, dtype),
            None if hi is None else self._from_lin(hi, dtype),
        )

    def _relax_atom(
        self, t: Expr, relaxed: Mapping[Var, Interval]
    ) -> tuple[Expr | None, Expr | None]:
        dtype = t.dtype
        if isinstance(t, Var):
            r = relaxed[t]
            return (
                None if r.lo is None else IntImm(r.lo, dtype),
                None if r.hi is None else IntImm(r.hi, dtype),
            )
        if isinstance(t, Binary) and isinstance(t.b, IntImm) and t.b.value > 0:
            c = IntImm(t.b.value, dtype)
            if t.op == "floordiv":
                lo, hi = self.relax(t.a, relaxed)
                return (
                    None if lo is None else self.simplify(Binary("floordiv", lo, c)),
                    None if hi is None else self.simplify(Binary("floordiv", hi, c)),
                )
            if t.op == "floormod":
                lo, hi = self.relax(t.a, relaxed)
                if lo is not None and hi is not None:
                    same = self.simplify(
                        Binary("sub", Binary("floordiv", hi, c), Binary("floordiv", lo, c))
                    )
                    if same == IntImm(0, dtype):
                        return (
                            self.simplify(Binary("floormod", lo, c)),
                            self.simplify(Binary("floormod", hi, c)),
                        )
                return IntImm(0, dtype), IntImm(t.b.value - 1, dtype)
        if isinstance(t, Binary) and t.op in ("min", "max"):
            alo, ahi = self.relax(t.a, relaxed)
            blo, bhi = self.relax(t.b, relaxed)
            lo = None if alo is None or blo is None else self.simplify(Binary(t.op, alo, blo))
            hi = None if ahi is None or bhi is None else self.simplify(Binary(t.op, ahi, bhi))
            if t.op == "min":
                lo = None if alo is None or blo is None else lo
                hi = hi if hi is not None else (ahi if bhi is None else bhi)
            else:
                hi = None if ahi is None or bhi is None else hi
                lo = lo if lo is not None else (alo if blo is None else blo)
            return lo, hi
        # Anything else: a constant bound with the relaxed variables over their ranges.
        inner = self.copy()
        for v, r in relaxed.items():
            inner.ranges[v] = r
        b = inner.bound(t)
        return (
            None if b.lo is None else IntImm(b.lo, dtype),
            None if b.hi is None else IntImm(b.hi, dtype),
        )


def _widens(src: DType, dst: DType) -> bool:
    order = [DType.bool_, DType.uint8, DType.int8, DType.int32, DType.int64]
    if src == dst:
        return True
    if src == DType.uint8 and dst == DType.int8:
        return False
    return src in order and dst in order and order.index(src) <= order.index(dst)


def _lin_dtype(lin: _Lin, default: DType = INDEX) -> DType:
    for t in lin.terms:
        return t.dtype
    return default


def _flatten(e: Expr, op: str) -> list[Expr]:
    if isinstance(e, Logical) and e.op == op:
        return _flatten(e.a, op) + _flatten(e.b, op)
    return [e]


def _merge_bounds(terms: list[Expr]) -> list[Expr]:
    """Keep the tightest of several ``L < c`` (or ``L >= c``) with the same ``L``."""
    best: dict[tuple[str, Expr], int] = {}
    order: list[tuple[str, Expr] | Expr] = []
    for t in terms:
        if isinstance(t, Compare) and t.op in ("lt", "ge") and isinstance(t.b, IntImm):
            key = (t.op, t.a)
            if key in best:
                best[key] = min(best[key], t.b.value) if t.op == "lt" else max(best[key], t.b.value)
            else:
                best[key] = t.b.value
                order.append(key)
        else:
            order.append(t)
    out: list[Expr] = []
    for item in order:
        if isinstance(item, tuple):
            op, lhs = item
            out.append(Compare("lt" if op == "lt" else "ge", lhs, IntImm(best[item], lhs.dtype)))
        else:
            out.append(item)
    return out


def _fold_compare(op: str, a: float, b: float) -> bool:
    if op == "lt":
        return a < b
    if op == "le":
        return a <= b
    if op == "eq":
        return a == b
    if op == "ne":
        return a != b
    if op == "gt":
        return a > b
    return a >= b


def _fold_float(op: str, a: float, b: float) -> float | None:
    with np.errstate(all="ignore"):
        x, y = np.float64(a), np.float64(b)
        if op == "add":
            return float(x + y)
        if op == "sub":
            return float(x - y)
        if op == "mul":
            return float(x * y)
        if op == "div":
            return float(x / y)
        if op == "min":
            return float(np.minimum(x, y))
        if op == "max":
            return float(np.maximum(x, y))
        if op == "pow":
            return float(np.power(x, y))
    return None


def _round(v: float, dtype: DType) -> float:
    with np.errstate(all="ignore"):
        if dtype == DType.float16:
            return float(np.float16(v))
        if dtype == DType.float32:
            return float(np.float32(v))
    return v


def _is_float_const(e: Expr, v: float) -> bool:
    return isinstance(e, FloatImm) and e.value == v


def simplify(e: Expr, ranges: Mapping[Var, Interval | Range] | None = None) -> Expr:
    """``e`` simplified, given the ranges of its variables (loop ranges, block domains)."""
    return Analyzer(ranges).simplify(e)


def bound(e: Expr, ranges: Mapping[Var, Interval | Range] | None = None) -> Interval:
    """A constant interval containing every value of ``e``."""
    return Analyzer(ranges).bound(e)
