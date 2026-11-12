"""Well-formedness checks of PrimFuncs (``tir.verify``)."""

from __future__ import annotations

import itertools
from math import prod
from typing import NoReturn

from tessera.dtype import DType
from tessera.tir.arith import Analyzer, Interval
from tessera.tir.expr import (
    AddressOf,
    Binary,
    BufferLoad,
    Call,
    Cast,
    Compare,
    Expr,
    IntImm,
    Logical,
    Not,
    Select,
    Var,
)
from tessera.tir.functor import (
    ExprMutator,
    ExprVisitor,
    children,
    free_vars,
    substitute,
    walk_exprs,
)
from tessera.tir.printer import script
from tessera.tir.stmt import (
    THREAD_TAGS,
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    Evaluate,
    For,
    ForKind,
    IfThenElse,
    IterKind,
    LetStmt,
    PrimFunc,
    SeqStmt,
    Stmt,
)


class VerifyError(ValueError):
    """A PrimFunc is not well formed."""


def verify(func: PrimFunc, *, threads: bool = True) -> None:
    """Check that ``func`` is well formed; raise ``VerifyError`` naming the problem.

    Checked: parameters are distinct global buffers; the body is a root block;
    every variable is defined before use and defined once; the body of a block
    with iteration variables uses only those (blocks are isolated from the
    loops around them, which see them only through the bindings); bindings
    stay inside the iteration domains (taking predicates into account);
    every load, store and declared region is inside its buffer; buffers are
    parameters or allocated by an enclosing block; operand types agree; block
    names are unique; thread-bound loops name a known grid axis; and (unless
    ``threads`` is False, for a function still being scheduled) the rule of
    ``check_thread_locals``.
    """
    _Verifier(func).run()
    if threads:
        check_thread_locals(func)


def check_thread_locals(func: PrimFunc) -> None:
    """Every use of a ``local`` buffer sits under loops bound to the same ``thread.*``
    axes with the same extents, or under none, and all uses map thread indices
    to elements the same way.

    Each thread has its own copy of a local buffer (Metal), so the copy a thread
    reads must be the one it wrote: an accumulator may be initialized, updated
    and stored by three separate nests, as long as each nest binds the same
    thread axes and indexes the buffer by them in the same way (the part of
    every index that depends on a thread loop is the same in every use).
    """
    owners: dict[Buffer, dict[frozenset[tuple[str, int]], str]] = {}
    patterns: dict[Buffer, set[tuple[frozenset[tuple[Expr, int]], ...]]] = {}
    tags: dict[str, Var] = {}
    for path in _leaf_paths(func.body):
        node = path[-1]
        threads = [
            n
            for n in path
            if isinstance(n, For)
            and n.thread_tag is not None
            and n.thread_tag.startswith("thread.")
            and isinstance(n.extent, IntImm)
        ]
        where = frozenset((str(n.thread_tag), _extent(n)) for n in threads)
        rename: dict[Var, Expr] = {
            n.loop_var: tags.setdefault(str(n.thread_tag), Var(str(n.thread_tag))) for n in threads
        }
        bindings = _bindings(path)
        for buf, indices in _accesses_in(node):
            if buf.scope != "local":
                continue
            owners.setdefault(buf, {}).setdefault(where, _describe(where))
            if threads:
                patterns.setdefault(buf, set()).add(
                    _thread_part(indices, bindings, rename, set(tags.values()))
                )
    for buf, places in owners.items():
        if len(places) > 1:
            listed = "; ".join(sorted(places.values()))
            raise VerifyError(
                f"{func.name}: local buffer {buf.name} is used under different thread "
                f"axes ({listed}); each thread sees only its own copy"
            )
    for buf, seen in patterns.items():
        if len(seen) > 1:
            raise VerifyError(
                f"{func.name}: uses of local buffer {buf.name} index it by thread "
                "differently; a thread would read elements another thread wrote"
            )


def _extent(loop: For) -> int:
    assert isinstance(loop.extent, IntImm)
    return loop.extent.value


def _bindings(path: tuple[Stmt, ...]) -> dict[Var, Expr]:
    """Block iteration variables on ``path`` in terms of loop variables."""
    out: dict[Var, Expr] = {}
    for n in path:
        if isinstance(n, BlockRealize):
            for iv, value in zip(n.block.iter_vars, n.iter_values, strict=True):
                out[iv.var] = substitute(value, out)
    return out


def _thread_part(
    indices: tuple[Expr, ...],
    bindings: dict[Var, Expr],
    rename: dict[Var, Expr],
    tag_vars: set[Var],
) -> tuple[frozenset[tuple[Expr, int]], ...]:
    """For each index, its terms that involve a thread loop (named by axis)."""
    an = Analyzer()
    out: list[frozenset[tuple[Expr, int]]] = []
    for e in indices:
        e = substitute(substitute(e, bindings), rename)
        terms, _ = an.linear(e)
        out.append(frozenset((t, k) for t, k in terms.items() if free_vars(t) & tag_vars))
    return tuple(out)


def _describe(where: frozenset[tuple[str, int]]) -> str:
    if not where:
        return "no thread loop"
    return ", ".join(f"{tag}={n}" for tag, n in sorted(where))


def _leaf_paths(stmt: Stmt) -> list[tuple[Stmt, ...]]:
    """Paths to statements that hold expressions: stores, calls, conditions, lets."""
    out: list[tuple[Stmt, ...]] = []
    stack: list[tuple[Stmt, ...]] = [(stmt,)]
    while stack:
        path = stack.pop()
        node = path[-1]
        if isinstance(node, BufferStore | Evaluate | IfThenElse | LetStmt):
            out.append(path)
        stack.extend((*path, c) for c in children(node))
    return out


def _accesses_in(node: Stmt) -> list[tuple[Buffer, tuple[Expr, ...]]]:
    """The buffer accesses held by ``node`` itself (not its children), with indices."""
    exprs: list[Expr] = []
    found: list[tuple[Buffer, tuple[Expr, ...]]] = []
    if isinstance(node, BufferStore):
        found.append((node.buffer, node.indices))
        exprs = [node.value, *node.indices]
    elif isinstance(node, Evaluate | LetStmt):
        exprs = [node.value]
    elif isinstance(node, IfThenElse):
        exprs = [node.condition]
    for e in exprs:
        for sub in walk_exprs(e):
            if isinstance(sub, BufferLoad | AddressOf):
                found.append((sub.buffer, sub.indices))
    return found


class _Verifier:
    def __init__(self, func: PrimFunc) -> None:
        self.func = func
        self.defined: set[Var] = set()
        self.block_names: set[str] = set()

    def fail(self, message: str) -> NoReturn:
        raise VerifyError(f"{self.func.name}: {message}")

    def run(self) -> None:
        f = self.func
        seen: set[int] = set()
        for p in f.params:
            if id(p) in seen:
                self.fail(f"parameter {p.name} appears twice")
            seen.add(id(p))
            if p.scope != "global":
                self.fail(f"parameter {p.name} has scope {p.scope}")
        if not (isinstance(f.body, BlockRealize) and not f.body.block.iter_vars):
            self.fail("the body must be a root block (a BlockRealize with no iteration variables)")
        self.stmt(f.body, frozenset(), frozenset(f.params), Analyzer())

    # -- statements ---------------------------------------------------------

    def define(self, v: Var) -> None:
        if v in self.defined:
            self.fail(f"variable {v.name} is defined twice")
        self.defined.add(v)

    def stmt(
        self, s: Stmt, visible: frozenset[Var], buffers: frozenset[Buffer], an: Analyzer
    ) -> None:
        if isinstance(s, SeqStmt):
            for c in s.seq:
                self.stmt(c, visible, buffers, an)
        elif isinstance(s, For):
            self.for_(s, visible, buffers, an)
        elif isinstance(s, BlockRealize):
            self.block_realize(s, visible, buffers, an)
        elif isinstance(s, BufferStore):
            self.access(s.buffer, s.indices, visible, buffers, an, "store to")
            self.expr(s.value, visible, buffers, an)
            if s.value.dtype != s.buffer.dtype:
                self.fail(f"store of {s.value.dtype} into {s.buffer.dtype} buffer {s.buffer.name}")
        elif isinstance(s, IfThenElse):
            self.expr(s.condition, visible, buffers, an)
            self.want_bool(s.condition, "if condition")
            self.stmt(s.then_case, visible, buffers, _assume(an, s.condition))
            if s.else_case is not None:
                self.stmt(s.else_case, visible, buffers, _assume(an, Not(s.condition)))
        elif isinstance(s, LetStmt):
            self.expr(s.value, visible, buffers, an)
            if s.value.dtype != s.var.type:
                self.fail(f"let {s.var.name}: {s.var.type} bound to a {s.value.dtype} value")
            self.define(s.var)
            inner = an.copy()
            if s.var.type.is_int:
                inner.bind(s.var, inner.bound(s.value))
            self.stmt(s.body, visible | {s.var}, buffers, inner)
        elif isinstance(s, Evaluate):
            self.expr(s.value, visible, buffers, an)
        elif isinstance(s, Block):
            self.fail(f"block {s.name} appears without a BlockRealize")
        else:
            self.fail(f"unknown statement {type(s).__name__}")

    def for_(
        self, s: For, visible: frozenset[Var], buffers: frozenset[Buffer], an: Analyzer
    ) -> None:
        self.expr(s.min, visible, buffers, an)
        if not isinstance(s.extent, IntImm):
            self.fail(f"loop {s.loop_var.name} has a non-constant extent")
        if s.extent.value < 1:
            self.fail(f"loop {s.loop_var.name} has extent {s.extent.value}")
        if (s.kind == ForKind.THREAD_BINDING) != (s.thread_tag is not None):
            self.fail(f"loop {s.loop_var.name}: thread_tag is set exactly for thread-bound loops")
        if s.thread_tag is not None and s.thread_tag not in THREAD_TAGS:
            self.fail(f"loop {s.loop_var.name} is bound to unknown axis {s.thread_tag!r}")
        self.define(s.loop_var)
        inner = an.copy()
        lo = an.bound(s.min)
        ext = s.extent.value
        inner.bind(s.loop_var, Interval(lo.lo, None if lo.hi is None else lo.hi + ext - 1))
        self.stmt(s.body, visible | {s.loop_var}, buffers, inner)

    def block_realize(
        self, s: BlockRealize, visible: frozenset[Var], buffers: frozenset[Buffer], an: Analyzer
    ) -> None:
        b = s.block
        if b.name in self.block_names:
            self.fail(f"two blocks are named {b.name!r}")
        self.block_names.add(b.name)
        if len(s.iter_values) != len(b.iter_vars):
            self.fail(
                f"block {b.name} has {len(b.iter_vars)} iteration variables "
                f"but {len(s.iter_values)} bindings"
            )
        self.expr(s.predicate, visible, buffers, an)
        self.want_bool(s.predicate, f"predicate of block {b.name}")
        pred = an.simplify(s.predicate)
        inner = Analyzer()
        for iv, value in zip(b.iter_vars, s.iter_values, strict=True):
            self.expr(value, visible, buffers, an)
            if not value.dtype.is_int:
                self.fail(f"block {b.name}: binding of {iv.var.name} is not an integer")
            if not (isinstance(iv.dom.min, IntImm) and isinstance(iv.dom.extent, IntImm)):
                self.fail(f"block {b.name}: domain of {iv.var.name} is not constant")
            lo, hi = iv.dom.min.value, iv.dom.min.value + iv.dom.extent.value - 1
            if not (_within(an, value, pred, lo, hi) or _enumerate(an, value, pred, lo, hi)):
                self.fail(
                    f"block {b.name}: binding {iv.var.name} = {script(value)} "
                    f"can leave its domain [{lo}, {hi}]"
                )
            self.define(iv.var)
            inner.bind(iv.var, Interval(lo, hi))
        if b.init is not None and not any(iv.kind == IterKind.REDUCE for iv in b.iter_vars):
            self.fail(f"block {b.name} has an init but no reduction variables")
        if b.iter_vars:
            scope = frozenset(iv.var for iv in b.iter_vars)
        else:
            scope = visible
            inner = an
        for buf in b.alloc_buffers:
            if buf in buffers:
                self.fail(f"buffer {buf.name} is allocated twice")
        inside = buffers | frozenset(b.alloc_buffers)
        for region in (*b.reads, *b.writes):
            self.region(region, b, scope, inside, inner)
        if b.init is not None:
            self.stmt(b.init, scope, inside, inner)
        self.stmt(b.body, scope, inside, inner)

    def region(
        self,
        region: BufferRegion,
        block: Block,
        visible: frozenset[Var],
        buffers: frozenset[Buffer],
        an: Analyzer,
    ) -> None:
        buf = region.buffer
        if buf not in buffers:
            self.fail(f"block {block.name} declares a region of unknown buffer {buf.name}")
        if len(region.region) != buf.ndim:
            self.fail(f"block {block.name}: region of {buf.name} has the wrong rank")
        for d, r in enumerate(region.region):
            self.expr(r.min, visible, buffers, an)
            self.expr(r.extent, visible, buffers, an)
            lo = an.bound(r.min)
            hi = _last(an, r.min, an.simplify(r.extent))
            if lo.lo is None or hi is None or lo.lo < 0 or hi >= buf.shape[d]:
                self.fail(
                    f"block {block.name}: region {script(r.min)}+{script(r.extent)} "
                    f"of {buf.name} dimension {d} can leave [0, {buf.shape[d]})"
                )

    # -- expressions --------------------------------------------------------

    def access(
        self,
        buf: Buffer,
        indices: tuple[Expr, ...],
        visible: frozenset[Var],
        buffers: frozenset[Buffer],
        an: Analyzer,
        what: str,
    ) -> None:
        if buf not in buffers:
            self.fail(f"{what} buffer {buf.name}, which is neither a parameter nor allocated")
        if len(indices) != buf.ndim:
            self.fail(f"{what} {buf.name} with {len(indices)} indices (rank {buf.ndim})")
        for d, i in enumerate(indices):
            self.expr(i, visible, buffers, an)
            if not i.dtype.is_int:
                self.fail(f"{what} {buf.name}: index {script(i)} is not an integer")
            b = an.bound(i)
            if b.lo is None or b.hi is None or b.lo < 0 or b.hi >= buf.shape[d]:
                self.fail(
                    f"{what} {buf.name}: index {script(i)} of dimension {d} "
                    f"can leave [0, {buf.shape[d]}) (bounds {b.lo}..{b.hi})"
                )

    def expr(
        self, e: Expr, visible: frozenset[Var], buffers: frozenset[Buffer], an: Analyzer
    ) -> None:
        checker = _ExprChecker(self, visible, buffers, an)
        checker.visit_expr(e)

    def want_bool(self, e: Expr, what: str) -> None:
        if e.dtype != DType.bool_:
            self.fail(f"{what} is {e.dtype}, not bool")


class _ExprChecker(ExprVisitor):
    def __init__(
        self, v: _Verifier, visible: frozenset[Var], buffers: frozenset[Buffer], an: Analyzer
    ) -> None:
        self.v = v
        self.visible = visible
        self.buffers = buffers
        self.an = an

    def visit_var(self, e: Var) -> None:
        if e not in self.visible:
            if e in self.v.defined:
                self.v.fail(f"variable {e.name} is used outside its scope")
            self.v.fail(f"variable {e.name} is used but never defined")

    def visit_binary(self, e: Binary) -> None:
        if e.a.dtype != e.b.dtype:
            self.v.fail(f"operands of {script(e)} have types {e.a.dtype} and {e.b.dtype}")
        super().visit_binary(e)

    def visit_compare(self, e: Compare) -> None:
        if e.a.dtype != e.b.dtype:
            self.v.fail(f"operands of {script(e)} have types {e.a.dtype} and {e.b.dtype}")
        super().visit_compare(e)

    def visit_logical(self, e: Logical) -> None:
        if e.a.dtype != DType.bool_ or e.b.dtype != DType.bool_:
            self.v.fail(f"operands of {script(e)} are not bool")
        super().visit_logical(e)

    def visit_not(self, e: Not) -> None:
        if e.a.dtype != DType.bool_:
            self.v.fail(f"operand of {script(e)} is not bool")
        super().visit_not(e)

    def visit_select(self, e: Select) -> None:
        if e.cond.dtype != DType.bool_:
            self.v.fail(f"condition of {script(e)} is not bool")
        if e.true_value.dtype != e.false_value.dtype:
            self.v.fail(f"branches of {script(e)} have different types")
        super().visit_select(e)

    def visit_cast(self, e: Cast) -> None:
        super().visit_cast(e)

    def visit_call(self, e: Call) -> None:
        super().visit_call(e)

    def visit_buffer_load(self, e: BufferLoad) -> None:
        self.v.access(e.buffer, e.indices, self.visible, self.buffers, self.an, "load from")

    def visit_address_of(self, e: AddressOf) -> None:
        self.v.access(e.buffer, e.indices, self.visible, self.buffers, self.an, "address of")


def _last(an: Analyzer, lo: Expr, extent: Expr) -> int | None:
    """An upper bound of ``lo + extent - 1``. An extent ``min(a, b)`` (an edge tile,
    ``min(64, 50257 - n_0 * 64)``) is bounded through each side, which keeps the
    correlation between the tile's start and its length."""
    if isinstance(extent, Binary) and extent.op == "min":
        bounds = [b for b in (_last(an, lo, extent.a), _last(an, lo, extent.b)) if b is not None]
        return min(bounds) if bounds else None
    return an.bound(lo + extent - 1).hi


def _assume(an: Analyzer, cond: Expr) -> Analyzer:
    """``an`` with the ranges of variables narrowed by the conjuncts ``v < c`` and
    ``v >= c`` of ``cond`` (the branch of an ``if`` where ``cond`` holds)."""
    out = an.copy()
    stack = [an.simplify(cond)]
    while stack:
        c = stack.pop()
        if isinstance(c, Logical) and c.op == "and":
            stack.extend((c.a, c.b))
            continue
        if not (isinstance(c, Compare) and isinstance(c.a, Var) and isinstance(c.b, IntImm)):
            continue
        r = out.bound(c.a)
        if c.op == "lt":
            hi = c.b.value - 1 if r.hi is None else min(r.hi, c.b.value - 1)
            out.bind(c.a, Interval(r.lo, hi))
        elif c.op == "ge":
            lo = c.b.value if r.lo is None else max(r.lo, c.b.value)
            out.bind(c.a, Interval(lo, r.hi))
    return out


# Bindings the interval reasoning cannot bound are checked by enumerating the
# loop variables they use, when there are at most this many combinations.
_ENUMERATION_LIMIT = 1 << 14


def _enumerate(an: Analyzer, value: Expr, predicate: Expr, lo: int, hi: int) -> bool:
    """Decide ``lo <= value <= hi`` wherever ``predicate`` holds by trying every value
    of the variables involved. Too many combinations to try counts as proved: the
    predicate was built by a primitive whose purpose is to guard the binding."""
    used = sorted(
        {e for e in walk_exprs(value) if isinstance(e, Var)}
        | {e for e in walk_exprs(predicate) if isinstance(e, Var)},
        key=lambda v: v.name,
    )
    ranges: list[range] = []
    for v in used:
        r = an.ranges.get(v)
        if r is None or r.lo is None or r.hi is None:
            return False
        ranges.append(range(r.lo, r.hi + 1))
    if prod(len(r) for r in ranges) > _ENUMERATION_LIMIT:
        return True
    names = [f"a{i}" for i in range(len(used))]
    env = dict(zip(used, names, strict=True))
    check = eval(
        f"lambda {', '.join(names)}: (not ({_py(predicate, env)})) "
        f"or ({lo} <= ({_py(value, env)}) <= {hi})"
    )
    return all(check(*point) for point in itertools.product(*ranges))


def _py(e: Expr, env: dict[Var, str]) -> str:
    """Python source of an integer or boolean index expression."""
    if isinstance(e, IntImm):
        return str(e.value)
    if isinstance(e, Var):
        return env[e]
    if isinstance(e, Binary):
        a, b = _py(e.a, env), _py(e.b, env)
        ops = {"add": "+", "sub": "-", "mul": "*", "floordiv": "//", "floormod": "%"}
        if e.op in ops:
            return f"({a} {ops[e.op]} {b})"
        if e.op in ("min", "max"):
            return f"{e.op}({a}, {b})"
        raise ValueError(f"cannot enumerate {e.op}")
    if isinstance(e, Compare):
        sym = {"lt": "<", "le": "<=", "eq": "==", "ne": "!=", "gt": ">", "ge": ">="}[e.op]
        return f"({_py(e.a, env)} {sym} {_py(e.b, env)})"
    if isinstance(e, Logical):
        return f"({_py(e.a, env)} {e.op} {_py(e.b, env)})"
    if isinstance(e, Not):
        return f"(not {_py(e.a, env)})"
    if isinstance(e, Select):
        return f"({_py(e.true_value, env)} if {_py(e.cond, env)} else {_py(e.false_value, env)})"
    raise ValueError(f"cannot enumerate {type(e).__name__}")


def _within(an: Analyzer, value: Expr, predicate: Expr, lo: int, hi: int) -> bool:
    """True if ``lo <= value <= hi`` wherever ``predicate`` holds."""
    b = an.bound(value)
    upper_ok = b.hi is not None and b.hi <= hi
    lower_ok = b.lo is not None and b.lo >= lo
    if upper_ok and lower_ok:
        return True
    conjuncts: list[Expr] = []
    stack = [predicate]
    while stack:
        p = stack.pop()
        if isinstance(p, Logical) and p.op == "and":
            stack.extend((p.a, p.b))
        else:
            conjuncts.append(p)
    replaced = _bound_under(an, value, conjuncts)
    upper_ok = upper_ok or (replaced.hi is not None and replaced.hi <= hi)
    lower_ok = lower_ok or (replaced.lo is not None and replaced.lo >= lo)
    for c in conjuncts:
        if not (isinstance(c, Compare) and isinstance(c.b, IntImm) and c.a.dtype.is_int):
            continue
        k = _multiple(an, value, c.a)
        if k is None:
            continue
        # value = k * lhs + rest, with lhs < c.b (or lhs >= c.b) and rest bounded.
        rest = _rest(an, value, c.a, k)
        if c.op == "lt" and rest.hi is not None and k * (c.b.value - 1) + rest.hi <= hi:
            upper_ok = True
        elif c.op == "ge" and rest.lo is not None and k * c.b.value + rest.lo >= lo:
            lower_ok = True
    return upper_ok and lower_ok


def _bound_under(an: Analyzer, value: Expr, conjuncts: list[Expr]) -> Interval:
    """Bounds of ``value`` where every sub-expression equal to the left side of a
    conjunct ``L < c`` (``L >= c``) takes only the values the conjunct allows."""
    inner = an.copy()
    mapping: dict[Expr, Var] = {}
    for c in conjuncts:
        if not (isinstance(c, Compare) and isinstance(c.b, IntImm) and c.a.dtype.is_int):
            continue
        if c.op not in ("lt", "ge"):
            continue
        b = an.bound(c.a)
        lhs = an.simplify(c.a)
        if c.op == "lt":
            r = Interval(b.lo, c.b.value - 1 if b.hi is None else min(b.hi, c.b.value - 1))
        else:
            r = Interval(c.b.value if b.lo is None else max(b.lo, c.b.value), b.hi)
        t = mapping.get(lhs)
        if t is None:
            t = Var("_pred", lhs.dtype)
            mapping[lhs] = t
            inner.bind(t, r)
        else:
            old = inner.ranges[t]
            lo_v = r.lo if old.lo is None else old.lo if r.lo is None else max(old.lo, r.lo)
            hi_v = r.hi if old.hi is None else old.hi if r.hi is None else min(old.hi, r.hi)
            inner.bind(t, Interval(lo_v, hi_v))
    if not mapping:
        return Interval.everything()

    class _Replace(ExprMutator):
        def visit_expr(self, e: Expr) -> Expr:
            found = mapping.get(e)
            if found is not None:
                return found
            return super().visit_expr(e)

    return inner.bound(_Replace().visit_expr(an.simplify(value)))


def _multiple(an: Analyzer, value: Expr, lhs: Expr) -> int | None:
    """A positive ``k`` such that ``value`` contains ``k * lhs`` term by term, if any."""
    v_terms = _terms(an.simplify(value))
    l_terms = _terms(an.simplify(lhs))
    if not l_terms:
        return None
    # A fused loop variable f appears as f in the binding but as f // c in the
    # predicate of an earlier split: write f as c * (f // c) + f % c.
    for t in l_terms:
        if isinstance(t, Binary) and t.op == "floordiv" and isinstance(t.b, IntImm):
            coef = v_terms.pop(t.a, None)
            if coef is not None:
                c = t.b.value
                v_terms[t] = v_terms.get(t, 0) + coef * c
                mod = Binary("floormod", t.a, t.b)
                v_terms[mod] = v_terms.get(mod, 0) + coef
    t, a = next(iter(l_terms.items()))
    b = v_terms.get(t)
    if b is None or a == 0 or b % a or b // a <= 0:
        return None
    k = b // a
    if all(v_terms.get(x) == k * coef for x, coef in l_terms.items()):
        return k
    return None


def _rest(an: Analyzer, value: Expr, lhs: Expr, k: int) -> Interval:
    """Bounds of ``value - k * lhs``, with fused variables split as in ``_multiple``."""
    direct = an.bound(value - lhs * k)
    terms = _terms(an.simplify(value))
    for t in _terms(an.simplify(lhs)):
        if isinstance(t, Binary) and t.op == "floordiv" and isinstance(t.b, IntImm):
            coef = terms.pop(t.a, None)
            if coef is not None:
                terms[t] = terms.get(t, 0) + coef * t.b.value
                mod = Binary("floormod", t.a, t.b)
                terms[mod] = terms.get(mod, 0) + coef
    for t, coef in _terms(an.simplify(lhs)).items():
        terms[t] = terms.get(t, 0) - k * coef
    out = Interval.point(_constant(an.simplify(value)) - k * _constant(an.simplify(lhs)))
    for t, coef in terms.items():
        if coef:
            out = out + an.bound(t).scale(coef)
    lo = direct.lo if out.lo is None else out.lo if direct.lo is None else max(out.lo, direct.lo)
    hi = direct.hi if out.hi is None else out.hi if direct.hi is None else min(out.hi, direct.hi)
    return Interval(lo, hi)


def _constant(e: Expr) -> int:
    if isinstance(e, IntImm):
        return e.value
    if isinstance(e, Binary) and e.op in ("add", "sub") and isinstance(e.b, IntImm):
        return e.b.value if e.op == "add" else -e.b.value
    return 0


def _terms(e: Expr) -> dict[Expr, int]:
    """The non-constant terms of a canonical linear expression with their coefficients."""
    out: dict[Expr, int] = {}

    def add(x: Expr, sign: int) -> None:
        if isinstance(x, Binary) and x.op in ("add", "sub"):
            add(x.a, sign)
            add(x.b, sign if x.op == "add" else -sign)
        elif isinstance(x, Binary) and x.op == "mul" and isinstance(x.b, IntImm):
            out[x.a] = out.get(x.a, 0) + sign * x.b.value
        elif not isinstance(x, IntImm):
            out[x] = out.get(x, 0) + sign

    add(e, 1)
    return out
