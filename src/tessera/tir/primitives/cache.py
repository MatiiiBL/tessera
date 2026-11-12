"""Cache stages and reduction decomposition: cache_read, cache_write, decompose_reduction."""

from __future__ import annotations

from tessera.tir.analysis import is_leaf_block, with_regions
from tessera.tir.expr import Expr, IntImm, Logical, Var
from tessera.tir.functor import Substitute, free_vars, substitute, substitute_stmt
from tessera.tir.primitives.blocks import _covers_buffer, _finalize, _leaves, _store_dims
from tessera.tir.primitives.common import (
    Path,
    ScheduleError,
    block_names,
    conjoin,
    facts,
    find_block,
    find_loop,
    flat_seq,
    fresh,
    is_reduction,
    loop_of,
    loop_roles,
    loops_above,
    only,
    realize_of,
    rebuild,
    replace_block,
    replace_for,
    scope_index,
    seq_items,
    used_names,
    with_body,
    writers,
)
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferStore,
    For,
    ForKind,
    IterKind,
    IterVar,
    PrimFunc,
    Range,
    Scope,
    SeqStmt,
    Stmt,
)

_SCOPES = ("global", "local", "shared")


def _copy_nest(func: PrimFunc, name: str, src: Buffer, dst: Buffer) -> Stmt:
    """Loops and a block copying all of ``src`` into ``dst`` (same shape)."""
    names = used_names(func)
    ivs: list[IterVar] = []
    loops: list[Var] = []
    for d, n in enumerate(src.shape):
        loops.append(Var(fresh(f"ax{d}", names)))
        ivs.append(IterVar(Var(fresh(f"v{d}", names)), Range.of(n)))
    idx = tuple(iv.var for iv in ivs)
    body = BufferStore(dst, src[idx], idx)
    block = with_regions(Block(name, tuple(ivs), (), (), body))
    stmt: Stmt = BlockRealize(tuple(loops), TRUE, block)
    for v, n in zip(reversed(loops), reversed(src.shape), strict=True):
        stmt = For(v, IntImm(0), IntImm(n), ForKind.SERIAL, stmt)
    return stmt


def _scope_parts(path: Path) -> tuple[Path, Block, list[Stmt], int]:
    """The path to the BlockRealize of the scope block of ``path[-1]``, that block,
    its body items and the index of the item containing ``path[-1]``."""
    si = scope_index(path)
    scope = path[si]
    assert isinstance(scope, Block)
    items = seq_items(scope.body)
    child = path[si + 1]
    if isinstance(scope.body, SeqStmt):
        child = path[si + 2]
    pos = next(i for i, s in enumerate(items) if s is child)
    return path[:si], scope, items, pos


def _with_scope(func: PrimFunc, scope_path: Path, new_scope: Block) -> PrimFunc:
    realize = scope_path[-1]
    assert isinstance(realize, BlockRealize)
    new_realize = BlockRealize(realize.iter_values, realize.predicate, new_scope)
    return with_body(func, rebuild(scope_path, new_realize))


def _check_scope(scope: str) -> None:
    if scope not in _SCOPES:
        raise ScheduleError(f"unknown storage scope {scope!r} (one of {_SCOPES})")


def _swap_buffer(func: PrimFunc, path: Path, old: Buffer, new: Buffer) -> PrimFunc:
    """``func`` with block ``path[-1]`` using ``new`` wherever it used ``old``."""
    r = realize_of(path)
    block = Substitute({}, {old: new}).visit_stmt(r.block)
    assert isinstance(block, Block)
    return with_body(func, rebuild(path, BlockRealize(r.iter_values, r.predicate, block)))


def _insert_in_scope(func: PrimFunc, name: str, nest: Stmt, cache: Buffer, after: bool) -> PrimFunc:
    """Insert ``nest`` next to the scope item holding block ``name`` and allocate ``cache``."""
    scope_path, scope_block, items, pos = _scope_parts(find_block(func, name))
    at = pos + 1 if after else pos
    new_scope = replace_block(
        scope_block,
        body=flat_seq([*items[:at], nest, *items[at:]]),
        alloc_buffers=(*scope_block.alloc_buffers, cache),
    )
    return _with_scope(func, scope_path, new_scope)


def cache_read(func: PrimFunc, name: str, read_index: int, scope: Scope) -> tuple[PrimFunc, str]:
    _check_scope(scope)
    path = find_block(func, name)
    b = realize_of(path).block
    if not is_leaf_block(b):
        raise ScheduleError(f"cannot cache reads of block {name}: it contains other blocks")
    if not 0 <= read_index < len(b.reads):
        raise ScheduleError(f"block {name} has {len(b.reads)} read regions, not {read_index + 1}")
    buf = b.reads[read_index].buffer
    if buf in facts(b).writes:
        raise ScheduleError(f"block {name} also writes {buf.name}")
    _, _, items, pos = _scope_parts(path)
    if any(buf in facts(w.block).writes for w in _leaves(items[pos])):
        raise ScheduleError(
            f"cannot cache {buf.name} for block {name}: it is written inside the same loop nest"
        )
    cache = Buffer(fresh(f"{buf.name}_{scope}", used_names(func)), buf.shape, buf.dtype, scope)
    copy_name = fresh(cache.name, block_names(func))
    nest = _copy_nest(func, copy_name, buf, cache)
    out = _swap_buffer(func, path, buf, cache)
    out = _insert_in_scope(out, name, nest, cache, after=False)
    return _finalize(out), copy_name


def cache_write(func: PrimFunc, name: str, write_index: int, scope: Scope) -> tuple[PrimFunc, str]:
    _check_scope(scope)
    path = find_block(func, name)
    r = realize_of(path)
    b = r.block
    if not is_leaf_block(b):
        raise ScheduleError(f"cannot cache writes of block {name}: it contains other blocks")
    if not 0 <= write_index < len(b.writes):
        raise ScheduleError(
            f"block {name} has {len(b.writes)} write regions, not {write_index + 1}"
        )
    buf = b.writes[write_index].buffer
    if not only(writers(func, buf), r):
        raise ScheduleError(f"cannot cache writes of block {name}: other blocks write {buf.name}")
    dims = _store_dims(b, buf, "cache writes of")
    _covers_buffer(b, buf, dims, "cache writes of")
    _, _, items, pos = _scope_parts(path)
    for other in _leaves(items[pos]):
        if other is not r and buf in (facts(other.block).reads | facts(other.block).writes):
            raise ScheduleError(
                f"cannot cache writes of block {name}: block {other.block.name} in the same "
                f"loop nest uses {buf.name}"
            )
    cache = Buffer(fresh(f"{buf.name}_{scope}", used_names(func)), buf.shape, buf.dtype, scope)
    copy_name = fresh(cache.name, block_names(func))
    nest = _copy_nest(func, copy_name, cache, buf)
    out = _swap_buffer(func, path, buf, cache)
    out = _insert_in_scope(out, name, nest, cache, after=True)
    return _finalize(out), copy_name


def decompose_reduction(func: PrimFunc, name: str, loop: Var) -> tuple[PrimFunc, str]:
    bpath = find_block(func, name)
    r = realize_of(bpath)
    b = r.block
    if not is_leaf_block(b) or not is_reduction(b) or b.init is None:
        raise ScheduleError(f"block {name} is not a reduction with an init")
    lpath = find_loop(func, loop)
    target = loop_of(lpath)
    loops = loops_above(bpath)
    if target not in loops:
        raise ScheduleError(f"loop {loop.name} is not around block {name}")
    roles = loop_roles(r, loops)
    ti = loops.index(target)
    for f, role in zip(loops[:ti], roles[:ti], strict=True):
        if role in ("R", "X"):
            raise ScheduleError(
                f"cannot split the init of {name} at loop {loop.name}: loop {f.loop_var.name} "
                "outside it is a reduction loop"
            )
    names = used_names(func)
    copied: list[tuple[For, Var]] = []
    for f, role in zip(loops[ti:], roles[ti:], strict=True):
        if role == "X":
            raise ScheduleError(
                f"loop {f.loop_var.name} feeds both spatial and reduction variables"
            )
        if role == "N":
            raise ScheduleError(
                f"cannot split the init of {name}: loop {f.loop_var.name} does not drive it"
            )
        if role == "S":
            copied.append((f, Var(fresh(f"{f.loop_var.name}_init", names), f.loop_var.dtype)))
    loop_map: dict[Var, Expr] = {f.loop_var: v for f, v in copied}
    spatial = [
        (iv, val)
        for iv, val in zip(b.iter_vars, r.iter_values, strict=True)
        if iv.kind == IterKind.SPATIAL
    ]
    var_map: dict[Var, Expr] = {}
    new_ivs: list[IterVar] = []
    values: list[Expr] = []
    for iv, val in spatial:
        nv = Var(fresh(iv.var.name, names), iv.var.dtype)
        var_map[iv.var] = nv
        new_ivs.append(IterVar(nv, iv.dom, IterKind.SPATIAL))
        values.append(substitute(val, loop_map))
    reduce_loops = {f.loop_var for f, role in zip(loops, roles, strict=True) if role == "R"}
    pred: Expr = TRUE
    for c in _conjuncts(r.predicate):
        used = free_vars(c)
        if used & reduce_loops:
            if used - reduce_loops & {f.loop_var for f, _ in copied}:
                raise ScheduleError(f"the predicate of {name} mixes spatial and reduction loops")
            continue
        pred = conjoin(pred, substitute(c, loop_map))
    init_body = substitute_stmt(b.init, var_map)
    init_name = fresh(f"{name}_init", block_names(func))
    init_block = with_regions(Block(init_name, tuple(new_ivs), (), (), init_body))
    stmt: Stmt = BlockRealize(tuple(values), pred, init_block)
    for f, v in reversed(copied):
        stmt = replace_for(f, loop_var=v, body=stmt)
    new_r = BlockRealize(r.iter_values, r.predicate, replace_block(b, init=None))
    out = with_body(func, rebuild(bpath, new_r))
    lpath = find_loop(out, loop)
    out = with_body(out, rebuild(lpath, flat_seq([stmt, loop_of(lpath)])))
    return _finalize(out), init_name


def _conjuncts(e: Expr) -> list[Expr]:
    if isinstance(e, Logical) and e.op == "and":
        return _conjuncts(e.a) + _conjuncts(e.b)
    return [] if e == TRUE else [e]
