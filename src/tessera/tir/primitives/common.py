"""Tree navigation and dependence facts shared by the schedule primitives.

Statements are immutable, so a primitive finds the path from the function
body to the node it rewrites (by identity), builds the new node, and
rebuilds the ancestors along that path. Removing a node also removes loops
left empty.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from tessera.tir.analysis import block_accesses, is_leaf_block, with_regions
from tessera.tir.arith import Analyzer, Interval
from tessera.tir.expr import Expr, IntImm, Logical, Not, Var
from tessera.tir.functor import StmtMutator, children, free_vars, walk_exprs, walk_stmts
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    For,
    IfThenElse,
    IterKind,
    LetStmt,
    PrimFunc,
    SeqStmt,
    Stmt,
)


class ScheduleError(Exception):
    """A primitive cannot be applied (it would change the result or the IR is not
    in the form it needs)."""


type Path = tuple[Stmt, ...]


# -- finding nodes --------------------------------------------------------------


def iter_paths(root: Stmt, prefix: Path = ()) -> Iterator[Path]:
    """Every node with its path from ``root``, in program order."""
    stack: list[Path] = [(*prefix, root)]
    while stack:
        path = stack.pop()
        yield path
        stack.extend((*path, c) for c in reversed(children(path[-1])))


def block_paths(func: PrimFunc) -> list[Path]:
    """Paths to every BlockRealize except the root, in program order."""
    return [p for p in iter_paths(func.body) if isinstance(p[-1], BlockRealize) and len(p) > 1]


def find_block(func: PrimFunc, name: str) -> Path:
    for path in iter_paths(func.body):
        node = path[-1]
        if isinstance(node, BlockRealize) and node.block.name == name and len(path) > 1:
            return path
    raise ScheduleError(f"no block named {name!r}")


def find_loop(func: PrimFunc, var: Var) -> Path:
    for path in iter_paths(func.body):
        node = path[-1]
        if isinstance(node, For) and node.loop_var is var:
            return path
    raise ScheduleError(f"no loop {var.name} (it was removed or never existed)")


def realize_of(path: Path) -> BlockRealize:
    node = path[-1]
    assert isinstance(node, BlockRealize)
    return node


def loop_of(path: Path) -> For:
    node = path[-1]
    assert isinstance(node, For)
    return node


def loops_above(path: Path) -> list[For]:
    """The loops around the last node of ``path`` inside its innermost enclosing block."""
    out: list[For] = []
    for node in path[:-1]:
        if isinstance(node, Block):
            out = []
        elif isinstance(node, For):
            out.append(node)
    return out


def scope_index(path: Path) -> int:
    """Index in ``path`` of the innermost Block enclosing the last node."""
    for i in range(len(path) - 2, -1, -1):
        if isinstance(path[i], Block):
            return i
    raise ScheduleError("node is not inside a block")


def blocks_under(stmt: Stmt) -> list[BlockRealize]:
    return [s for s in walk_stmts(stmt) if isinstance(s, BlockRealize)]


# -- rebuilding -------------------------------------------------------------------


def rebuild(path: Path, new: Stmt | None) -> Stmt:
    """The root of ``path`` with ``path[-1]`` replaced by ``new`` (``None`` removes it,
    and loops left empty go too)."""
    cur = new
    for i in range(len(path) - 2, -1, -1):
        cur = _replace_child(path[i], path[i + 1], cur)
    if cur is None:
        return SeqStmt(())
    return cur


def _replace_child(parent: Stmt, old: Stmt, new: Stmt | None) -> Stmt | None:
    if isinstance(parent, For):
        if new is None:
            return None
        return For(
            parent.loop_var,
            parent.min,
            parent.extent,
            parent.kind,
            new,
            parent.thread_tag,
            parent.annotations,
        )
    if isinstance(parent, BlockRealize):
        if new is None:
            return None
        assert isinstance(new, Block)
        return BlockRealize(parent.iter_values, parent.predicate, new)
    if isinstance(parent, Block):
        init, body = parent.init, parent.body
        if old is parent.init:
            init = new
        else:
            body = new if new is not None else SeqStmt(())
        return Block(
            parent.name,
            parent.iter_vars,
            parent.reads,
            parent.writes,
            body,
            init,
            parent.alloc_buffers,
            parent.annotations,
        )
    if isinstance(parent, SeqStmt):
        items: list[Stmt] = []
        for c in parent.seq:
            if c is old:
                if new is not None:
                    items.append(new)
            else:
                items.append(c)
        if not items:
            return None
        return flat_seq(items)
    if isinstance(parent, IfThenElse):
        if old is parent.then_case:
            if new is None:
                if parent.else_case is None:
                    return None
                return IfThenElse(Not(parent.condition), parent.else_case)
            return IfThenElse(parent.condition, new, parent.else_case)
        return IfThenElse(parent.condition, parent.then_case, new)
    if isinstance(parent, LetStmt):
        return None if new is None else LetStmt(parent.var, parent.value, new)
    raise ScheduleError(f"cannot rewrite inside {type(parent).__name__}")


def flat_seq(items: Sequence[Stmt]) -> Stmt:
    flat: list[Stmt] = []
    for s in items:
        flat.extend(s.seq if isinstance(s, SeqStmt) else (s,))
    return flat[0] if len(flat) == 1 else SeqStmt(tuple(flat))


def seq_items(s: Stmt) -> list[Stmt]:
    return list(s.seq) if isinstance(s, SeqStmt) else [s]


def with_body(func: PrimFunc, body: Stmt) -> PrimFunc:
    return PrimFunc(func.name, func.params, body, func.attrs)


def replace_block(block: Block, **changes: object) -> Block:
    fields = {
        "name": block.name,
        "iter_vars": block.iter_vars,
        "reads": block.reads,
        "writes": block.writes,
        "body": block.body,
        "init": block.init,
        "alloc_buffers": block.alloc_buffers,
        "annotations": block.annotations,
    }
    fields.update(changes)
    return Block(**fields)  # type: ignore[arg-type]


def replace_for(loop: For, **changes: object) -> For:
    fields = {
        "loop_var": loop.loop_var,
        "min": loop.min,
        "extent": loop.extent,
        "kind": loop.kind,
        "body": loop.body,
        "thread_tag": loop.thread_tag,
        "annotations": loop.annotations,
    }
    fields.update(changes)
    return For(**fields)  # type: ignore[arg-type]


def exclusive_root(path: Path) -> int:
    """Index in ``path`` of the outermost loop that contains only ``path[-1]``
    (the subtree to remove when moving the block); the block itself if none."""
    i = len(path) - 1
    while i - 1 >= 0 and isinstance(path[i - 1], For):
        i -= 1
    return i


# -- simplification -----------------------------------------------------------------


class Simplify(StmtMutator):
    """Simplifies bindings, predicates, regions and indices with the loop ranges in scope."""

    def __init__(self, analyzer: Analyzer | None = None) -> None:
        self.an = analyzer or Analyzer()

    def visit_expr(self, e: Expr) -> Expr:
        out = self.an.simplify(e)
        return e if out == e else out

    def visit_for(self, s: For) -> Stmt:
        self.an.bind(s.loop_var, _loop_interval(self.an, s))
        try:
            return super().visit_for(s)
        finally:
            self.an.unbind(s.loop_var)

    def visit_block(self, s: Block) -> Stmt:
        saved = dict(self.an.ranges)
        if s.iter_vars:
            self.an.ranges = {}
            for iv in s.iter_vars:
                self.an.bind(iv.var, iv.dom)
        try:
            return super().visit_block(s)
        finally:
            self.an.ranges = saved


def _loop_interval(an: Analyzer, s: For) -> Interval:
    lo = an.bound(s.min)
    if not isinstance(s.extent, IntImm):
        return Interval(lo.lo, None)
    return Interval(lo.lo, None if lo.hi is None else lo.hi + s.extent.value - 1)


def simplify_stmt(s: Stmt, analyzer: Analyzer | None = None) -> Stmt:
    return Simplify(analyzer).visit_stmt(s)


def analyzer_for(path: Path) -> Analyzer:
    """An analyzer knowing the ranges of the loops on ``path`` (inside its scope block)."""
    an = Analyzer()
    for node in path[:-1]:
        if isinstance(node, Block) and node.iter_vars:
            an = Analyzer({iv.var: iv.dom for iv in node.iter_vars})
        elif isinstance(node, For):
            an.bind(node.loop_var, _loop_interval(an, node))
    return an


# -- dependence facts ----------------------------------------------------------------


@dataclass
class BlockFacts:
    """What a leaf block reads and writes (from its body)."""

    reads: set[Buffer]
    writes: set[Buffer]


def facts(block: Block) -> BlockFacts:
    acc = block_accesses(block)
    writes = set(acc.stores) | set(acc.addressed)
    reads = set(acc.loads) | set(acc.addressed)
    return BlockFacts(reads, writes)


def is_reduction(block: Block) -> bool:
    return any(iv.kind == IterKind.REDUCE for iv in block.iter_vars)


def leaf_blocks(stmt: Stmt) -> list[BlockRealize]:
    return [r for r in blocks_under(stmt) if is_leaf_block(r.block)]


def writers(func: PrimFunc, buf: Buffer) -> list[BlockRealize]:
    return [r for r in leaf_blocks(func.body) if buf in facts(r.block).writes]


def readers(func: PrimFunc, buf: Buffer) -> list[BlockRealize]:
    return [r for r in leaf_blocks(func.body) if buf in facts(r.block).reads]


def loop_roles(realize: BlockRealize, loops: Sequence[For]) -> list[str]:
    """For each loop: ``S`` (feeds only spatial bindings), ``R`` (only reduction
    bindings), ``X`` (both) or ``N`` (none)."""
    spatial: set[Var] = set()
    reduce: set[Var] = set()
    for iv, value in zip(realize.block.iter_vars, realize.iter_values, strict=True):
        target = reduce if iv.kind == IterKind.REDUCE else spatial
        target.update(free_vars(value))
    out = []
    for loop in loops:
        v = loop.loop_var
        s, r = v in spatial, v in reduce
        out.append("X" if s and r else "S" if s else "R" if r else "N")
    return out


def check_reductions(func: PrimFunc) -> None:
    """Every reduction keeps its meaning under the current loops.

    A loop must not feed both spatial and reduction variables of a block, and
    a loop the block does not depend on must not sit inside the block's
    reduction loops (it would accumulate the same terms several times, or
    re-run the init in the middle of the reduction).
    """
    for path in block_paths(func):
        r = realize_of(path)
        if not is_reduction(r.block):
            continue
        loops = loops_above(path)
        roles = loop_roles(r, loops)
        seen_r = False
        for loop, role in zip(loops, roles, strict=True):
            if role == "X":
                raise ScheduleError(
                    f"loop {loop.loop_var.name} feeds both spatial and reduction "
                    f"variables of block {r.block.name}"
                )
            if role == "R":
                seen_r = True
            elif role == "N" and seen_r:
                raise ScheduleError(
                    f"loop {loop.loop_var.name} would repeat the reduction of block {r.block.name}"
                )


def refresh_regions(stmt: Stmt) -> Stmt:
    """Recompute the read/write regions of every leaf block in ``stmt``."""

    class _Refresh(StmtMutator):
        def visit_block(self, s: Block) -> Stmt:
            inner = super().visit_block(s)
            assert isinstance(inner, Block)
            if is_leaf_block(inner):
                updated = with_regions(inner)
                if updated.reads == inner.reads and updated.writes == inner.writes:
                    return inner
                return updated
            return inner

    return _Refresh().visit_stmt(stmt)


def conjoin(a: Expr, b: Expr) -> Expr:
    if a == TRUE:
        return b
    if b == TRUE:
        return a
    return Logical("and", a, b)


def used_names(func: PrimFunc) -> set[str]:
    names: set[str] = set()
    for p in iter_paths(func.body):
        node = p[-1]
        if isinstance(node, For):
            names.add(node.loop_var.name)
        elif isinstance(node, Block):
            names.update(iv.var.name for iv in node.iter_vars)
            names.update(b.name for b in node.alloc_buffers)
    names.update(b.name for b in func.params)
    for e in walk_exprs(func.body):
        if isinstance(e, Var):
            names.add(e.name)
    return names


def block_names(func: PrimFunc) -> set[str]:
    return {r.block.name for r in blocks_under(func.body)}


def fresh(base: str, used: set[str]) -> str:
    """``base`` or ``base_<n>``, whichever is not in ``used`` (and record it)."""
    name, n = base, 1
    while name in used:
        name = f"{base}_{n}"
        n += 1
    used.add(name)
    return name


def program_order(func: PrimFunc) -> list[BlockRealize]:
    """Leaf blocks in program order."""
    return leaf_blocks(func.body)


def only(items: Sequence[BlockRealize], r: BlockRealize) -> bool:
    """True if ``items`` is exactly ``[r]`` (by identity)."""
    return len(items) == 1 and items[0] is r


def index_of(blocks: Sequence[BlockRealize], target: BlockRealize) -> int:
    for i, b in enumerate(blocks):
        if b is target:
            return i
    raise ScheduleError(f"block {target.block.name} not found")
