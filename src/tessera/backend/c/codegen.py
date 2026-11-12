"""PrimFunc to C11.

Each PrimFunc becomes ``static void <name>(T* restrict <param>, ...)``, one
pointer per parameter. The loop nest is printed as it is scheduled:

- a block opens a scope that binds its iteration variables to their values,
  tests its predicate and runs its reduction ``init`` when every reduce
  variable is at its first value;
- ``local`` buffers become stack arrays placed and sized by
  ``alloc.plan_allocations``; above ``STACK_LIMIT``, and for ``global``
  intermediates, storage that each running thread allocates once and keeps;
- a ``PARALLEL`` loop is outlined into a function that runs a contiguous range
  of iterations, and the kernel hands it to the thread pool with the values
  it captures (on at most ``parallel.threads`` threads if the loop has that
  annotation); a parallel loop inside another runs serially;
- ``VECTORIZED`` and ``UNROLLED`` loops carry ``#pragma clang loop`` hints,
  which clang honours for the loops templates produce (constant trip counts,
  unit-stride innermost accesses, no predicates);
- ``Evaluate(Call("c.<name>", ...))`` calls a registered microkernel.

The output depends only on the function, so equal functions give equal text
(and hit the same build cache entry).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from tessera.backend.c.alloc import (
    AllocationPlan,
    Compacted,
    Path,
    compacted_index,
    plan_allocations,
)
from tessera.backend.c.microkernels import get_microkernel
from tessera.dtype import DType
from tessera.target import CpuIsa
from tessera.tir.expr import (
    MATH_INTRINSICS,
    AddressOf,
    Binary,
    BufferLoad,
    Call,
    Cast,
    Compare,
    Expr,
    FloatImm,
    IntImm,
    Logical,
    Not,
    Select,
    Var,
)
from tessera.tir.stmt import (
    Block,
    BlockRealize,
    Buffer,
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

# Loop annotation: run a parallel loop on at most this many threads.
PARALLEL_THREADS = "parallel.threads"

# Local buffers up to this size live on the stack. Python threads other than
# the main one have 512 KiB stacks on macOS, and the calling thread runs the
# first chunk of every parallel loop.
STACK_LIMIT = 64 * 1024

_STORAGE = {
    DType.float32: "float",
    DType.float16: "_Float16",
    DType.int64: "int64_t",
    DType.int32: "int32_t",
    DType.int8: "int8_t",
    DType.uint8: "uint8_t",
    DType.bool_: "uint8_t",  # NumPy's bool is one byte holding 0 or 1
}

_MATH_F32 = {
    "exp": "tsr_expf",
    "log": "tsr_logf",
    "sqrt": "sqrtf",
    "tanh": "tsr_tanhf",
    "erf": "tsr_erff",
    "abs": "fabsf",
    "floor": "floorf",
    "ceil": "ceilf",
    "fma": "fmaf",
    "sigmoid": "tsr_sigmoidf",
    "rsqrt": "tsr_rsqrtf",
}

# C operator precedence, higher binds tighter.
_PRIMARY, _UNARY, _MUL, _ADD, _SHIFT, _REL, _EQ, _BITAND, _AND, _OR, _COND = (
    16,
    15,
    13,
    12,
    11,
    10,
    9,
    8,
    5,
    4,
    3,
)
_INFIX = {
    "add": ("+", _ADD),
    "sub": ("-", _ADD),
    "mul": ("*", _MUL),
    "div": ("/", _MUL),
    "lt": ("<", _REL),
    "le": ("<=", _REL),
    "gt": (">", _REL),
    "ge": (">=", _REL),
    "eq": ("==", _EQ),
    "ne": ("!=", _EQ),
}

_C_KEYWORDS = (
    "auto break case char const continue default do double else enum extern float for goto if "
    "inline int long register restrict return short signed sizeof static struct switch typedef "
    "union unsigned void volatile while bool true false"
)
# Names the generated code uses itself: outlined-function parameters, the C
# library functions it calls and the fixed-width types.
_GENERATED_NAMES = (
    "begin end args a main NULL INFINITY NAN int8_t int16_t int32_t int64_t uint8_t uint16_t "
    "uint32_t uint64_t size_t expf logf sqrtf tanhf erff fabsf floorf ceilf fmaf fminf fmaxf "
    "powf fmodf exp log sqrt tanh erf fabs floor ceil fma fmin fmax pow abs free malloc memcpy "
    "memset"
)
_RESERVED = frozenset((_C_KEYWORDS + " " + _GENERATED_NAMES).split())


class CodegenError(Exception):
    """The function uses something the C backend cannot express."""


@dataclass(frozen=True)
class KernelCode:
    """C source of one kernel: its helper functions, then the kernel itself."""

    name: str
    params: tuple[Buffer, ...]
    text: str
    headers: tuple[str, ...]  # microkernel headers it needs, in first-use order
    parallel: bool  # whether it calls the thread pool


def c_type(dtype: DType) -> str:
    """The C type a buffer of ``dtype`` stores its elements as."""
    return _STORAGE[dtype]


def generate_kernel(func: PrimFunc, isa: CpuIsa = "generic") -> KernelCode:
    """Translate ``func`` to C for a target with ``isa``."""
    return _Emitter(func, isa).run()


def fold(e: Expr) -> Expr:
    """Fold integer constants and drop identities (``x * 1``, ``x + 0``), so the
    printed indices stay short."""
    if isinstance(e, Binary):
        a, b = fold(e.a), fold(e.b)
        if isinstance(a, IntImm) and isinstance(b, IntImm) and e.dtype.is_int:
            value = _fold_ints(e.op, a.value, b.value)
            if value is not None:
                return IntImm(value, a.type)
        if e.dtype.is_int:
            if e.op == "add":
                if _is_int(a, 0):
                    return b
                if _is_int(b, 0):
                    return a
                if isinstance(b, IntImm) and b.value < 0:
                    return Binary("sub", a, IntImm(-b.value, b.type))
            elif e.op == "sub":
                if _is_int(b, 0):
                    return a
                if isinstance(b, IntImm) and b.value < 0:
                    return Binary("add", a, IntImm(-b.value, b.type))
            elif e.op == "mul":
                if _is_int(a, 1):
                    return b
                if _is_int(b, 1):
                    return a
                if _is_int(a, 0) or _is_int(b, 0):
                    return IntImm(0, e.dtype)
            elif e.op == "floordiv" and _is_int(b, 1):
                return a
            elif e.op == "floormod" and _is_int(b, 1):
                return IntImm(0, e.dtype)
        return e if a is e.a and b is e.b else Binary(e.op, a, b)
    if isinstance(e, Compare):
        return Compare(e.op, fold(e.a), fold(e.b))
    if isinstance(e, Logical):
        return Logical(e.op, fold(e.a), fold(e.b))
    if isinstance(e, Not):
        return Not(fold(e.a))
    if isinstance(e, Select):
        return Select(fold(e.cond), fold(e.true_value), fold(e.false_value))
    if isinstance(e, Cast):
        inner = fold(e.value)
        return inner if inner.dtype is e.type else Cast(e.type, inner)
    if isinstance(e, Call):
        return Call(e.name, tuple(fold(a) for a in e.args), e.type)
    if isinstance(e, BufferLoad):
        return BufferLoad(e.buffer, tuple(fold(i) for i in e.indices))
    if isinstance(e, AddressOf):
        return AddressOf(e.buffer, tuple(fold(i) for i in e.indices))
    return e


def substitute(e: Expr, mapping: Mapping[Var, Expr]) -> Expr:
    """``e`` with every variable in ``mapping`` replaced by its value."""
    if isinstance(e, Var):
        return mapping.get(e, e)
    if isinstance(e, Binary):
        return Binary(e.op, substitute(e.a, mapping), substitute(e.b, mapping))
    if isinstance(e, Compare):
        return Compare(e.op, substitute(e.a, mapping), substitute(e.b, mapping))
    if isinstance(e, Logical):
        return Logical(e.op, substitute(e.a, mapping), substitute(e.b, mapping))
    if isinstance(e, Not):
        return Not(substitute(e.a, mapping))
    if isinstance(e, Select):
        return Select(
            substitute(e.cond, mapping),
            substitute(e.true_value, mapping),
            substitute(e.false_value, mapping),
        )
    if isinstance(e, Cast):
        return Cast(e.type, substitute(e.value, mapping))
    if isinstance(e, Call):
        return Call(e.name, tuple(substitute(a, mapping) for a in e.args), e.type)
    if isinstance(e, BufferLoad):
        return BufferLoad(e.buffer, tuple(substitute(i, mapping) for i in e.indices))
    if isinstance(e, AddressOf):
        return AddressOf(e.buffer, tuple(substitute(i, mapping) for i in e.indices))
    return e


def _fold_ints(op: str, a: int, b: int) -> int | None:
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "min":
        return min(a, b)
    if op == "max":
        return max(a, b)
    if b == 0:
        return None
    if op == "floordiv":
        return a // b
    if op == "floormod":
        return a % b
    if op == "div":
        return int(a / b)
    return None


def _is_int(e: Expr, value: int) -> bool:
    return isinstance(e, IntImm) and e.value == value


def float_literal(value: float, dtype: DType = DType.float32) -> str:
    """A C literal that reads back as exactly ``value`` rounded to ``dtype``."""
    if math.isnan(value):
        text = "NAN"
    elif math.isinf(value):
        text = "INFINITY" if value > 0 else "-INFINITY"
    else:
        text = str(np.float32(value))
        if "." not in text and "e" not in text:
            text += ".0"
        text += "f"
    return f"(_Float16){text}" if dtype is DType.float16 else text


class _Names:
    """Unique, readable C identifiers for variables and buffers."""

    def __init__(self, reserved: set[str]) -> None:
        self.used = set(_RESERVED) | reserved
        self.names: dict[Var | Buffer, str] = {}

    def __call__(self, obj: Var | Buffer) -> str:
        name = self.names.get(obj)
        if name is None:
            name = self.fresh(obj.name)
            self.names[obj] = name
        return name

    def fresh(self, hint: str) -> str:
        base = "".join(c if c.isalnum() or c == "_" else "_" for c in hint) or "v"
        if base[0].isdigit():
            base = "v" + base
        if base.lower().startswith(("tsr_", "tessera_", "__")):
            base = "x" + base
        name, n = base, 1
        while name in self.used:
            name = f"{base}_{n}"
            n += 1
        self.used.add(name)
        return name


class _Emitter:
    def __init__(self, func: PrimFunc, isa: CpuIsa) -> None:
        self.func = func
        self.isa = isa
        self.kernel = _Names(set()).fresh(func.name)
        self.names = _Names({self.kernel})
        self.plan: AllocationPlan = plan_allocations(func.body)
        self.compacted: dict[Buffer, Compacted] = {
            c.buffer: c for allocs in self.plan.local.values() for c in allocs
        }
        self.lines: list[str] = []
        self.depth = 1
        self.helpers: list[str] = []
        self.headers: list[str] = []
        self.n_parallel = 0
        self.in_parallel = False
        # Inside a parallel body: what it refers to and what it defines itself.
        self.refs: dict[Var | Buffer, None] = {}
        self.defined: set[Var | Buffer] = set()
        self.nonneg: set[Var] = set()
        self.subst: dict[Var, Expr] = {}  # variables of one-iteration loops

    # -- driver ---------------------------------------------------------------

    def run(self) -> KernelCode:
        params = [f"{c_type(b.dtype)}* restrict {self.names(b)}" for b in self.func.params]
        for buf in self.plan.hoisted:
            self.persistent(self.names(buf), c_type(buf.dtype), buf.numel * buf.dtype.itemsize)
        self.stmt(self.func.body, ())
        signature = f"static void {self.kernel}({', '.join(params) or 'void'})"
        kernel = "\n".join([signature + " {", *self.lines, "}"])
        text = "\n\n".join([*self.helpers, kernel]) + "\n"
        return KernelCode(
            self.kernel, self.func.params, text, tuple(self.headers), self.n_parallel > 0
        )

    def persistent(self, name: str, ctype: str, nbytes: int) -> None:
        """Declare ``name`` as storage the running thread allocates once and keeps
        across iterations and calls: allocating megabytes each time costs page
        faults, and freeing them, unmapping."""
        storage = self.names.fresh(f"{name}_storage")
        self.line(f"static _Thread_local {ctype}* {storage};")
        self.line(f"if (!{storage}) {storage} = ({ctype}*)tsr_alloc({nbytes});")
        self.line(f"{ctype}* restrict {name} = {storage};")

    def line(self, text: str) -> None:
        self.lines.append("  " * self.depth + text)

    # -- names ----------------------------------------------------------------

    def var(self, v: Var) -> str:
        if self.in_parallel:
            self.refs[v] = None
        return self.names(v)

    def buf(self, b: Buffer) -> str:
        if self.in_parallel:
            self.refs[b] = None
        return self.names(b)

    def define(self, v: Var | Buffer) -> str:
        if self.in_parallel:
            self.defined.add(v)
        return self.names(v)

    # -- statements -----------------------------------------------------------

    def stmt(self, s: Stmt, path: Path) -> None:
        allocs = self.plan.local.get(path)
        if not allocs:
            self.node(s, path)
            return
        self.line("{")
        self.depth += 1
        for c in allocs:
            name = self.define(c.buffer)
            ctype = c_type(c.buffer.dtype)
            nbytes = c.numel * c.buffer.dtype.itemsize
            if nbytes <= STACK_LIMIT:
                self.line(f"{ctype} {name}[{c.numel}] __attribute__((aligned(64)));")
            else:
                # Too big for the stack: storage the thread keeps and reuses at
                # every iteration (a thread runs its iterations one at a time).
                self.persistent(name, ctype, nbytes)
        self.node(s, path)
        self.depth -= 1
        self.line("}")

    def node(self, s: Stmt, path: Path) -> None:
        if isinstance(s, For):
            self.for_(s, path)
        elif isinstance(s, BlockRealize):
            self.block(s, path)
        elif isinstance(s, SeqStmt):
            for i, child in enumerate(s.seq):
                self.stmt(child, (*path, i))
        elif isinstance(s, BufferStore):
            self.store(s)
        elif isinstance(s, Evaluate):
            self.evaluate(s)
        elif isinstance(s, IfThenElse):
            self.line(f"if ({self.expr(s.condition)}) {{")
            self.nested(s.then_case, (*path, 0))
            if s.else_case is not None:
                self.line("} else {")
                self.nested(s.else_case, (*path, 1))
            self.line("}")
        elif isinstance(s, LetStmt):
            value = self.expr(s.value)
            name = self.define(s.var)
            if self.is_nonneg(s.value):
                self.nonneg.add(s.var)
            self.line("{")
            self.depth += 1
            self.line(f"const {self.scalar_type(s.var.dtype)} {name} = {value};")
            self.stmt(s.body, (*path, 0))
            self.depth -= 1
            self.line("}")
        elif isinstance(s, Block):
            raise CodegenError(f"block {s.name!r} is not inside a BlockRealize")
        else:
            raise CodegenError(f"cannot generate C for {type(s).__name__}")

    def nested(self, s: Stmt, path: Path) -> None:
        self.depth += 1
        self.stmt(s, path)
        self.depth -= 1

    def for_(self, s: For, path: Path) -> None:
        if s.kind is ForKind.THREAD_BINDING:
            raise CodegenError(f"loop {s.loop_var.name} is bound to {s.thread_tag} (Metal only)")
        lo = fold(self.substitute(s.min))
        extent = fold(self.substitute(s.extent))
        if isinstance(extent, IntImm) and extent.value == 1:
            # A loop of one iteration (left by splits): its variable is a value.
            self.subst[s.loop_var] = lo
            if self.is_nonneg(lo):
                self.nonneg.add(s.loop_var)
            self.stmt(s.body, (*path, 0))
            return
        if s.kind is ForKind.PARALLEL and not self.in_parallel:
            self.parallel(s, lo, extent, path)
            return
        lo_text = self.expr(lo)
        hi_text = self.expr(fold(Binary("add", lo, extent)))
        name = self.define(s.loop_var)
        if self.is_nonneg(lo):
            self.nonneg.add(s.loop_var)
        ctype = self.scalar_type(s.loop_var.dtype)
        if s.kind is ForKind.VECTORIZED:
            self.line("#pragma clang loop vectorize(enable) interleave(enable)")
        elif s.kind is ForKind.UNROLLED:
            full = isinstance(extent, IntImm)
            self.line(f"#pragma clang loop unroll({'full' if full else 'enable'})")
        elif s.kind is ForKind.PARALLEL:
            self.line("// parallel loop inside a parallel loop: runs serially")
        self.line(f"for ({ctype} {name} = {lo_text}; {name} < {hi_text}; ++{name}) {{")
        self.nested(s.body, (*path, 0))
        self.line("}")

    def parallel(self, s: For, lo: Expr, extent: Expr, path: Path) -> None:
        lo_text = self.expr(lo)
        hi_text = self.expr(fold(Binary("add", lo, extent)))
        fname = self.names.fresh(f"{self.kernel}_parallel{self.n_parallel}")
        self.n_parallel += 1

        outer = (self.lines, self.depth, self.refs, self.defined)
        self.lines, self.depth, self.refs, self.defined = [], 1, {}, set()
        self.in_parallel = True
        name = self.define(s.loop_var)
        if self.is_nonneg(lo):
            self.nonneg.add(s.loop_var)
        ctype = self.scalar_type(s.loop_var.dtype)
        self.line(f"for ({ctype} {name} = begin; {name} < end; ++{name}) {{")
        self.nested(s.body, (*path, 0))
        self.line("}")
        body = self.lines
        # Kernel parameters first, in their order; then everything else in
        # order of first use.
        free = [r for r in self.refs if r not in self.defined]
        first = [p for p in self.func.params if p in free]
        captured: list[Var | Buffer] = [*first, *(r for r in free if r not in first)]
        self.lines, self.depth, self.refs, self.defined = outer
        self.in_parallel = False

        members, params, values = [], [], []
        for obj in captured:
            cname = self.names(obj)
            if isinstance(obj, Buffer):
                ptr = f"{c_type(obj.dtype)}*"
                members.append(f"  {ptr} {cname};")
                params.append(f"{ptr} restrict {cname}")
            else:
                stype = self.scalar_type(obj.dtype)
                members.append(f"  {stype} {cname};")
                params.append(f"const {stype} {cname}")
            values.append(cname)
        params += ["int64_t begin", "int64_t end"]
        helper = [f"static void {fname}_body({', '.join(params)}) {{", *body, "}", ""]
        call_args = ", ".join([*(f"a->{v}" for v in values), "begin", "end"])
        helper.append(f"static void {fname}(void* args, int64_t begin, int64_t end) {{")
        if captured:
            helper.insert(0, "typedef struct {\n" + "\n".join(members) + f"\n}} {fname}_args;\n")
            helper.append(f"  const {fname}_args* a = (const {fname}_args*)args;")
        else:
            helper.append("  (void)args;")
        helper.append(f"  {fname}_body({call_args});")
        helper.append("}")
        self.helpers.append("\n".join(helper))

        threads = dict(s.annotations).get(PARALLEL_THREADS)
        if isinstance(threads, int):
            launch = f"tsr_parallel_for_n({lo_text}, {hi_text}, {threads}, {fname}"
        else:
            launch = f"tsr_parallel_for({lo_text}, {hi_text}, {fname}"
        if captured:
            self.line("{")
            self.depth += 1
            self.line(f"{fname}_args args = {{{', '.join(values)}}};")
            self.line(f"{launch}, &args);")
            self.depth -= 1
            self.line("}")
        else:
            self.line(f"{launch}, NULL);")

    def block(self, s: BlockRealize, path: Path) -> None:
        block = s.block
        root = path == () and not block.iter_vars and block.init is None and _is_true(s.predicate)
        if root:
            self.stmt(block.body, (*path, 0))
            return
        self.line(f"{{  // {block.name}")
        self.depth += 1
        for iv, value in zip(block.iter_vars, s.iter_values, strict=True):
            bound = fold(self.substitute(value))
            if isinstance(bound, IntImm):
                # A constant (a dimension of extent one): use the value itself.
                self.subst[iv.var] = bound
                if bound.value >= 0:
                    self.nonneg.add(iv.var)
                continue
            text = self.expr(bound)
            name = self.define(iv.var)
            if self.is_nonneg(value):
                self.nonneg.add(iv.var)
            self.line(f"const {self.scalar_type(iv.var.dtype)} {name} = {text};")
        guarded = not _is_true(s.predicate)
        if guarded:
            self.line(f"if ({self.expr(fold(s.predicate))}) {{")
            self.depth += 1
        if block.init is not None:
            first = [
                f"{self.expr(iv.var)} == {self.expr(fold(iv.dom.min))}"
                for iv in block.iter_vars
                if iv.kind is IterKind.REDUCE
            ]
            if first:
                self.line(f"if ({' && '.join(first)}) {{")
                self.nested(block.init, (*path, 1))
                self.line("}")
            else:
                self.stmt(block.init, (*path, 1))
        self.stmt(block.body, (*path, 0))
        if guarded:
            self.depth -= 1
            self.line("}")
        self.depth -= 1
        self.line("}")

    def store(self, s: BufferStore) -> None:
        target = self.element(s.buffer, s.indices)
        value = fold(s.value)
        if (
            isinstance(value, Binary)
            and value.op in ("add", "mul")
            and isinstance(value.a, BufferLoad)
            and value.a.buffer is s.buffer
            and tuple(fold(i) for i in value.a.indices) == tuple(fold(i) for i in s.indices)
        ):
            op = "+=" if value.op == "add" else "*="
            self.line(f"{target} {op} {self.expr(value.b)};")
            return
        self.line(f"{target} = {self.expr(value)};")

    def evaluate(self, s: Evaluate) -> None:
        value = fold(s.value)
        if not isinstance(value, Call) or value.name in MATH_INTRINSICS:
            self.line(f"(void)({self.expr(value)});")
            return
        kernel = get_microkernel(value.name)
        if kernel is None:
            raise CodegenError(f"no C microkernel is registered as {value.name!r}")
        if not kernel.available_on(self.isa):
            raise CodegenError(
                f"microkernel {value.name!r} needs ISA {kernel.isa!r}, target has {self.isa!r}"
            )
        if kernel.header not in self.headers:
            self.headers.append(kernel.header)
        args = ", ".join(self.expr(a) for a in value.args)
        self.line(f"{kernel.symbol}({args});")

    # -- expressions ----------------------------------------------------------

    def expr(self, e: Expr) -> str:
        return self.emit(fold(self.substitute(e)))[0]

    def substitute(self, e: Expr) -> Expr:
        return substitute(e, self.subst) if self.subst else e

    def operand(self, e: Expr, min_prec: int) -> str:
        text, prec = self.emit(e)
        return text if prec >= min_prec else f"({text})"

    def emit(self, e: Expr) -> tuple[str, int]:
        if isinstance(e, IntImm):
            return str(e.value), (_UNARY if e.value < 0 else _PRIMARY)
        if isinstance(e, FloatImm):
            text = float_literal(e.value, e.type)
            return text, (_UNARY if text.startswith(("-", "(")) else _PRIMARY)
        if isinstance(e, Var):
            if e in self.subst:
                return self.emit(fold(self.subst[e]))
            return self.var(e), _PRIMARY
        if isinstance(e, BufferLoad):
            return self.element(e.buffer, e.indices), _PRIMARY
        if isinstance(e, AddressOf):
            flat = self.flat_index(e.buffer, e.indices)
            name = self.buf(e.buffer)
            if _is_int(flat, 0):
                return name, _PRIMARY
            return f"{name} + {self.operand(flat, _MUL)}", _ADD
        if isinstance(e, Binary):
            return self.binary(e)
        if isinstance(e, Compare):
            op, prec = _INFIX[e.op]
            return f"{self.operand(e.a, prec)} {op} {self.operand(e.b, prec + 1)}", prec
        if isinstance(e, Logical):
            # Mixed && and || are always parenthesized (clang warns otherwise).
            prec = _AND if e.op == "and" else _OR
            op = "&&" if e.op == "and" else "||"
            a, b = self.logical_operand(e.a, e.op), self.logical_operand(e.b, e.op)
            return f"{a} {op} {b}", prec
        if isinstance(e, Not):
            return f"!{self.operand(e.a, _UNARY)}", _UNARY
        if isinstance(e, Select):
            cond = self.operand(e.cond, _OR)
            t = self.operand(e.true_value, _OR)
            f = self.operand(e.false_value, _OR)
            return f"{cond} ? {t} : {f}", _COND
        if isinstance(e, Cast):
            if e.type is DType.bool_:
                return f"{self.operand(e.value, _EQ + 1)} != 0", _EQ
            return f"({c_type(e.type)}){self.operand(e.value, _UNARY)}", _UNARY
        if isinstance(e, Call):
            return self.call(e), _PRIMARY
        raise CodegenError(f"cannot generate C for {type(e).__name__}")

    def logical_operand(self, e: Expr, op: str) -> str:
        if isinstance(e, Logical) and e.op == op:
            return self.emit(e)[0]
        return self.operand(e, _EQ)

    def binary(self, e: Binary) -> tuple[str, int]:
        a, b = e.a, e.b
        is_float = e.dtype.is_float
        if e.op in _INFIX:
            op, prec = _INFIX[e.op]
            return f"{self.operand(a, prec)} {op} {self.operand(b, prec + 1)}", prec
        if e.op in ("floordiv", "floormod"):
            if is_float:
                fn = "tsr_floordivf" if e.op == "floordiv" else "tsr_floormodf"
                return self.wrap16(f"{fn}({self.expr32(a)}, {self.expr32(b)})", e.dtype), _PRIMARY
            if isinstance(b, IntImm) and b.value > 0 and b.value & (b.value - 1) == 0:
                # Shifts and masks round toward negative infinity for any sign.
                if e.op == "floordiv":
                    shift = b.value.bit_length() - 1
                    return f"{self.operand(a, _ADD + 1)} >> {shift}", _SHIFT
                return f"{self.operand(a, _BITAND + 1)} & {b.value - 1}", _BITAND
            if self.is_nonneg(a) and self.is_positive(b):
                op = "/" if e.op == "floordiv" else "%"
                return f"{self.operand(a, _MUL)} {op} {self.operand(b, _MUL + 1)}", _MUL
            fn = "tsr_floordiv" if e.op == "floordiv" else "tsr_floormod"
            return f"{fn}({self.expr(a)}, {self.expr(b)})", _PRIMARY
        if e.op in ("min", "max"):
            if is_float:
                fn = "fminf" if e.op == "min" else "fmaxf"
                return self.wrap16(f"{fn}({self.expr32(a)}, {self.expr32(b)})", e.dtype), _PRIMARY
            fn = "tsr_min" if e.op == "min" else "tsr_max"
            return f"{fn}({self.expr(a)}, {self.expr(b)})", _PRIMARY
        if e.op == "pow":
            if is_float:
                return self.wrap16(f"powf({self.expr32(a)}, {self.expr32(b)})", e.dtype), _PRIMARY
            return f"tsr_powi({self.expr(a)}, {self.expr(b)})", _PRIMARY
        raise CodegenError(f"unknown binary operator {e.op!r}")

    def call(self, e: Call) -> str:
        if e.name not in MATH_INTRINSICS:
            raise CodegenError(f"call to {e.name!r} outside Evaluate")
        if e.type.is_int and e.name == "abs":
            return f"tsr_absi({self.expr(e.args[0])})"
        if not e.type.is_float:
            raise CodegenError(f"math intrinsic {e.name!r} on {e.type}")
        args = ", ".join(self.expr32(a) for a in e.args)
        fn = _MATH_F32[e.name]
        return self.wrap16(f"{fn}({args})", e.type)

    def expr32(self, e: Expr) -> str:
        """An argument to a float function: float16 values are widened."""
        if e.dtype is DType.float16:
            return f"(float){self.operand(e, _UNARY)}"
        return self.expr(e)

    @staticmethod
    def wrap16(text: str, dtype: DType) -> str:
        return f"(_Float16){text}" if dtype is DType.float16 else text

    def element(self, buf: Buffer, indices: tuple[Expr, ...]) -> str:
        return f"{self.buf(buf)}[{self.expr(self.flat_index(buf, indices))}]"

    def flat_index(self, buf: Buffer, indices: tuple[Expr, ...]) -> Expr:
        if len(indices) != buf.ndim:
            raise CodegenError(f"{buf.name} has {buf.ndim} dimensions, indexed with {len(indices)}")
        compact = self.compacted.get(buf)
        shape = buf.shape if compact is None else compact.shape
        strides = [1] * buf.ndim
        for dim in reversed(range(buf.ndim - 1)):
            strides[dim] = strides[dim + 1] * shape[dim + 1]
        # Outermost dimension first, so the sum prints without parentheses.
        flat: Expr | None = None
        for dim in range(buf.ndim):
            index = fold(indices[dim])
            if compact is not None:
                index = compacted_index(compact, dim, index, self.plan)
            index = fold(self.substitute(index))
            term = fold(Binary("mul", index, IntImm(strides[dim], index.dtype)))
            flat = term if flat is None else fold(Binary("add", flat, term))
        return flat if flat is not None else IntImm(0)

    # -- facts ------------------------------------------------------------------

    def is_nonneg(self, e: Expr) -> bool:
        if isinstance(e, IntImm):
            return e.value >= 0
        if isinstance(e, Var):
            return e in self.nonneg
        if isinstance(e, Binary) and e.op in ("add", "mul", "floordiv", "floormod", "min", "max"):
            return self.is_nonneg(e.a) and self.is_nonneg(e.b)
        if isinstance(e, Cast) and e.type.is_int and e.value.dtype.is_int:
            return self.is_nonneg(e.value)
        if isinstance(e, Select):
            return self.is_nonneg(e.true_value) and self.is_nonneg(e.false_value)
        return False

    def is_positive(self, e: Expr) -> bool:
        return isinstance(e, IntImm) and e.value > 0

    @staticmethod
    def scalar_type(dtype: DType) -> str:
        return "bool" if dtype is DType.bool_ else c_type(dtype)


def _is_true(e: Expr) -> bool:
    return isinstance(e, IntImm) and e.value != 0
