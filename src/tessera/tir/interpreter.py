"""Runs PrimFuncs on NumPy arrays (``tir.interpret``).

The interpreter does not walk the tree per element. It translates the
function once into Python source, loops into ``for`` loops and blocks into
their bindings, predicate, init test and body, compiles it, and caches the
compiled kernel on the function object. Buffers live in flat
``array.array`` storage (typed C arrays, about three times faster to index
from Python than NumPy arrays), and intrinsic references see them as NumPy
views without copies.

Semantics match the C backend's: a block runs only where its predicate
holds, its init runs when every reduction variable is at the start of its
domain, ``floordiv``/``floormod`` round toward negative infinity, integer
``div`` truncates, float arithmetic follows IEEE (division by zero gives
infinities, ``exp`` overflows to infinity, ``min``/``max`` propagate NaN).
Arithmetic is done in double precision and rounded to float32 on every
store. Thread-bound loops run one after the other, which matches Metal for
the IR the schedule produces, with one rule taken from Metal: a ``local``
buffer allocated outside the ``thread.*`` loops belongs to each thread, so
the interpreter keeps one copy of it per thread index. Buffers a function
allocates start filled with NaN (floats) or zeros, so reading an element
nothing wrote shows up in the result.
"""

from __future__ import annotations

import array
import math
import re
import weakref
from collections.abc import Callable, Sequence
from math import prod
from typing import Any

import numpy as np
import numpy.typing as npt

from tessera.dtype import DType
from tessera.tir import intrin
from tessera.tir.arith import Analyzer
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
from tessera.tir.functor import StmtVisitor, walk_stmts
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferStore,
    Evaluate,
    For,
    IfThenElse,
    IterKind,
    LetStmt,
    PrimFunc,
    SeqStmt,
    Stmt,
)
from tessera.tir.verify import check_thread_locals

type NDArray = npt.NDArray[Any]

# Python allows 20 nested loop blocks per function; deeper nests are outlined.
_MAX_DEPTH = 15

_TYPECODE = {
    DType.float32: "f",
    DType.float16: "f",
    DType.int64: "q",
    DType.int32: "i",
    DType.int8: "b",
    DType.uint8: "B",
    DType.bool_: "B",
}
_STORAGE_NUMPY: dict[str, np.dtype[Any]] = {
    "f": np.dtype(np.float32),
    "q": np.dtype(np.int64),
    "i": np.dtype(np.int32),
    "b": np.dtype(np.int8),
    "B": np.dtype(np.uint8),
}
_NAN_F = array.array("f", [math.nan]).tobytes()

_COMPARE = {"lt": "<", "le": "<=", "eq": "==", "ne": "!=", "gt": ">", "ge": ">="}


# -- runtime helpers (globals of the generated code) -------------------------


def _alloc(typecode: str, n: int) -> array.array[Any]:
    if typecode == "f":
        return array.array("f", _NAN_F * n)
    return array.array(typecode, bytes(array.array(typecode).itemsize * n))


def _fdiv(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        if a == 0 or a != a:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _idiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _ffloordiv(a: float, b: float) -> float:
    q = _fdiv(a, b)
    return float(math.floor(q)) if math.isfinite(q) else q


def _ffmod(a: float, b: float) -> float:
    try:
        return a % b
    except ZeroDivisionError:
        return math.nan


def _pow(a: float, b: float) -> float:
    try:
        r = a**b
    except ZeroDivisionError:
        return math.inf
    except OverflowError:
        return math.inf
    if isinstance(r, complex):
        return math.nan
    return float(r)


def _fmin(a: float, b: float) -> float:
    return a if a <= b or a != a else b


def _fmax(a: float, b: float) -> float:
    return a if a >= b or a != a else b


def _exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def _log(x: float) -> float:
    if x > 0:
        return math.log(x)
    return -math.inf if x == 0 else math.nan


def _sqrt(x: float) -> float:
    return math.sqrt(x) if x >= 0 else math.nan


def _rsqrt(x: float) -> float:
    if x > 0:
        return 1.0 / math.sqrt(x)
    return math.inf if x == 0 else math.nan


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = _exp(x)
    return e / (1.0 + e)


def _floor(x: float) -> float:
    return float(math.floor(x)) if math.isfinite(x) else x


def _ceil(x: float) -> float:
    return float(math.ceil(x)) if math.isfinite(x) else x


def _f2i(x: float) -> int:
    if x != x:
        return 0
    if math.isinf(x):
        return -(2**63) if x < 0 else 2**63 - 1
    return int(x)


def _wrap(bits: int, signed: bool) -> Callable[[int], int]:
    mask = (1 << bits) - 1
    half = 1 << (bits - 1)

    def wrap(x: int) -> int:
        x &= mask
        return x - (1 << bits) if signed and x >= half else x

    return wrap


def _check(i: int, n: int) -> int:
    if not 0 <= i < n:
        raise IndexError(f"index {i} outside [0, {n})")
    return i


_MATH = {
    "exp": "_exp",
    "log": "_log",
    "sqrt": "_sqrt",
    "rsqrt": "_rsqrt",
    "tanh": "_tanh",
    "erf": "_erf",
    "sigmoid": "_sigmoid",
    "abs": "abs",
    "floor": "_floor",
    "ceil": "_ceil",
}

_HELPERS: dict[str, Any] = {
    "_alloc": _alloc,
    "_fdiv": _fdiv,
    "_idiv": _idiv,
    "_ffloordiv": _ffloordiv,
    "_ffmod": _ffmod,
    "_pow": _pow,
    "_fmin": _fmin,
    "_fmax": _fmax,
    "_exp": _exp,
    "_log": _log,
    "_sqrt": _sqrt,
    "_rsqrt": _rsqrt,
    "_tanh": math.tanh,
    "_erf": math.erf,
    "_sigmoid": _sigmoid,
    "_floor": _floor,
    "_ceil": _ceil,
    "_f2i": _f2i,
    "_ck": _check,
    "_inf": math.inf,
    "_nan": math.nan,
    "_Ptr": intrin.Pointer,
    "_frombuffer": np.frombuffer,
    "_wrap_i32": _wrap(32, True),
    "_wrap_i8": _wrap(8, True),
    "_wrap_u8": _wrap(8, False),
    "_wrap_i64": _wrap(64, True),
}
_WRAP = {
    DType.int32: "_wrap_i32",
    DType.int8: "_wrap_i8",
    DType.uint8: "_wrap_u8",
    DType.int64: "_wrap_i64",
}


# -- code generation -----------------------------------------------------------


class _Gen:
    """Translates one PrimFunc into the source of a Python function."""

    def __init__(self, func: PrimFunc, check_bounds: bool) -> None:
        self.func = func
        self.check_bounds = check_bounds
        self.names: dict[int, str] = {}
        self.keep: list[object] = []
        self.used: set[str] = set(_HELPERS)
        self.globals: dict[str, Any] = dict(_HELPERS)
        self.functions: list[list[str]] = []
        self.lines: list[str] = []
        self.indent = 1
        self.depth = 0
        self.live: list[str] = []
        self.views: dict[Buffer, str] = {}
        self.an = Analyzer()
        self.addressed = _addressed_buffers(func.body)
        # Metal semantics: a local buffer allocated outside the thread loops belongs
        # to each thread, so it gets one copy per thread index.
        self.thread_extents = _thread_extents(func.body)
        self.copies = prod(self.thread_extents.values())
        self.per_thread: set[Buffer] = set()
        self.threads: dict[str, str] = {}

    # names -----------------------------------------------------------------

    def name(self, obj: Var | Buffer, prefix: str) -> str:
        key = id(obj)
        found = self.names.get(key)
        if found is not None:
            return found
        base = prefix + re.sub(r"\W", "_", obj.name)
        text, n = base, 1
        while text in self.used:
            text = f"{base}_{n}"
            n += 1
        self.used.add(text)
        self.names[key] = text
        self.keep.append(obj)
        return text

    def alias(self, var: Var, text: str) -> None:
        self.names[id(var)] = text
        self.keep.append(var)

    def emit(self, text: str) -> None:
        self.lines.append("    " * self.indent + text)

    # expressions -----------------------------------------------------------

    def expr(self, e: Expr) -> str:
        if isinstance(e, IntImm):
            return str(e.value) if e.type != DType.bool_ else ("True" if e.value else "False")
        if isinstance(e, FloatImm):
            v = e.value
            if math.isnan(v):
                return "_nan"
            if math.isinf(v):
                return "_inf" if v > 0 else "(-_inf)"
            return repr(v)
        if isinstance(e, Var):
            found = self.names.get(id(e))
            if found is None:
                raise ValueError(f"variable {e.name} is used but not defined")
            return found
        if isinstance(e, Binary):
            return self.binary(e)
        if isinstance(e, Compare):
            return f"({self.expr(e.a)} {_COMPARE[e.op]} {self.expr(e.b)})"
        if isinstance(e, Logical):
            return f"({self.expr(e.a)} {e.op} {self.expr(e.b)})"
        if isinstance(e, Not):
            return f"(not {self.expr(e.a)})"
        if isinstance(e, Select):
            c, t, f = (self.expr(x) for x in (e.cond, e.true_value, e.false_value))
            return f"({t} if {c} else {f})"
        if isinstance(e, Cast):
            return self.cast(e)
        if isinstance(e, Call):
            return self.call(e)
        if isinstance(e, BufferLoad):
            return f"{self.name(e.buffer, 'b_')}[{self.flat(e.buffer, e.indices)}]"
        if isinstance(e, AddressOf):
            return f"_Ptr({self.view(e.buffer)}, {self.flat(e.buffer, e.indices)})"
        raise TypeError(f"cannot interpret expression {e!r}")

    def binary(self, e: Binary) -> str:
        a, b = self.expr(e.a), self.expr(e.b)
        is_float = e.dtype.is_float
        op = e.op
        if op in ("add", "sub", "mul"):
            return f"({a} {'+' if op == 'add' else '-' if op == 'sub' else '*'} {b})"
        if op == "div":
            return f"_fdiv({a}, {b})" if is_float else f"_idiv({a}, {b})"
        if op == "floordiv":
            return f"_ffloordiv({a}, {b})" if is_float else f"({a} // {b})"
        if op == "floormod":
            return f"_ffmod({a}, {b})" if is_float else f"({a} % {b})"
        if op in ("min", "max"):
            if is_float:
                return f"_f{op}({a}, {b})"
            return f"{op}({a}, {b})"
        if op == "pow":
            return f"_pow({a}, {b})" if is_float else f"({a} ** {b})"
        raise ValueError(f"unknown operator {op}")

    def cast(self, e: Cast) -> str:
        v = self.expr(e.value)
        src, dst = e.value.dtype, e.type
        if dst.is_float:
            return v if src.is_float else f"float({v})"
        if dst == DType.bool_:
            return f"({v} != 0)"
        if src.is_float:
            v = f"_f2i({v})"
        elif src == DType.bool_:
            v = f"int({v})"
        if _narrower(dst, src):
            return f"{_WRAP[dst]}({v})"
        return v

    def call(self, e: Call) -> str:
        args = [self.expr(a) for a in e.args]
        if e.name in MATH_INTRINSICS:
            if e.name == "fma":
                return f"({args[0]} * {args[1]} + {args[2]})"
            return f"{_MATH[e.name]}({args[0]})"
        ref = intrin.get(e.name).reference
        key = "_I_" + re.sub(r"\W", "_", e.name)
        self.globals[key] = ref
        return f"{key}({', '.join(args)})"

    def flat(self, buf: Buffer, indices: tuple[Expr, ...]) -> str:
        if buf in self.per_thread:
            return f"{self.thread_index()} * {buf.numel} + {self.flat_in_copy(buf, indices)}"
        return self.flat_in_copy(buf, indices)

    def thread_index(self) -> str:
        """The current thread's position among all threads of the function."""
        out = "0"
        for tag in ("thread.z", "thread.y", "thread.x"):
            extent = self.thread_extents.get(tag)
            if extent is None:
                continue
            here = self.threads.get(tag, "0")
            out = f"({out} * {extent} + {here})"
        return out

    def flat_in_copy(self, buf: Buffer, indices: tuple[Expr, ...]) -> str:
        if not indices:
            return "0"
        if self.check_bounds:
            parts = [
                f"_ck({self.expr(i)}, {n})" + (f" * {s}" if s != 1 else "")
                for i, n, s in zip(indices, buf.shape, buf.strides, strict=True)
            ]
            return " + ".join(parts)
        total: Expr = IntImm(0)
        for i, s in zip(indices, buf.strides, strict=True):
            total = total + i * s
        return self.expr(self.an.simplify(total))

    def view(self, buf: Buffer) -> str:
        found = self.views.get(buf)
        if found is None:
            raise ValueError(f"buffer {buf.name} is addressed but has no view")
        return found

    # statements --------------------------------------------------------------

    def stmt(self, s: Stmt) -> None:
        if isinstance(s, SeqStmt):
            if not s.seq:
                self.emit("pass")
            for c in s.seq:
                self.stmt(c)
        elif isinstance(s, For):
            self.for_(s)
        elif isinstance(s, BlockRealize):
            self.block_realize(s)
        elif isinstance(s, BufferStore):
            target = f"{self.name(s.buffer, 'b_')}[{self.flat(s.buffer, s.indices)}]"
            value = self.expr(s.value)
            if s.buffer.dtype.is_int and s.buffer.dtype != DType.int64:
                value = f"{_WRAP[s.buffer.dtype]}({value})"
            self.emit(f"{target} = {value}")
        elif isinstance(s, IfThenElse):
            self.emit(f"if {self.expr(s.condition)}:")
            self.nested(s.then_case)
            if s.else_case is not None:
                self.emit("else:")
                self.nested(s.else_case)
        elif isinstance(s, LetStmt):
            name = self.name(s.var, "v_")
            self.emit(f"{name} = {self.expr(s.value)}")
            self.live.append(name)
            self.stmt(s.body)
            self.live.pop()
        elif isinstance(s, Evaluate):
            self.emit(self.expr(s.value))
        else:
            raise TypeError(f"cannot interpret statement {type(s).__name__}")

    def nested(self, s: Stmt) -> None:
        self.indent += 1
        self.stmt(s)
        self.indent -= 1

    def for_(self, s: For) -> None:
        if self.depth >= _MAX_DEPTH:
            self.outline(s)
            return
        var = self.name(s.loop_var, "v_")
        assert isinstance(s.extent, IntImm)
        if isinstance(s.min, IntImm) and s.min.value == 0:
            bounds = str(s.extent.value)
        else:
            lo = self.expr(s.min)
            bounds = f"{lo}, {lo} + {s.extent.value}"
        self.emit(f"for {var} in range({bounds}):")
        self.depth += 1
        self.live.append(var)
        tag = s.thread_tag if s.thread_tag and s.thread_tag.startswith("thread.") else None
        outer_thread = self.threads.get(tag) if tag else None
        if tag:
            self.threads[tag] = var
        self.nested(s.body)
        if tag:
            if outer_thread is None:
                del self.threads[tag]
            else:
                self.threads[tag] = outer_thread
        self.live.pop()
        self.depth -= 1

    def outline(self, s: For) -> None:
        """Emit ``s`` as a separate function (resets the nesting count)."""
        fname = f"_sub{len(self.functions)}"
        params = list(dict.fromkeys(self.live))
        self.emit(f"{fname}({', '.join(params)})")
        saved = (self.lines, self.indent, self.depth)
        self.lines, self.indent, self.depth = [f"def {fname}({', '.join(params)}):"], 1, 0
        self.functions.append(self.lines)
        self.for_(s)
        self.lines, self.indent, self.depth = saved

    def block_realize(self, s: BlockRealize) -> None:
        b = s.block
        self.emit(f"# block {b.name}")
        pushed = 0
        for iv, value in zip(b.iter_vars, s.iter_values, strict=True):
            if isinstance(value, Var):
                self.alias(iv.var, self.expr(value))
            else:
                name = self.name(iv.var, "v_")
                self.emit(f"{name} = {self.expr(value)}")
                self.live.append(name)
                pushed += 1
        guarded = s.predicate != TRUE
        if guarded:
            self.emit(f"if {self.expr(s.predicate)}:")
            self.indent += 1
        self.block_body(b)
        if guarded:
            self.indent -= 1
        del self.live[len(self.live) - pushed :]

    def block_body(self, b: Block) -> None:
        pushed = 0
        for buf in b.alloc_buffers:
            name = self.name(buf, "b_")
            size = buf.numel
            if buf.scope == "local" and self.thread_extents and not self.threads:
                self.per_thread.add(buf)
                size *= self.copies
            self.emit(f"{name} = _alloc({_TYPECODE[buf.dtype]!r}, {size})")
            self.live.append(name)
            pushed += 1
            if buf in self.addressed:
                pushed += self.make_view(buf, name)
        if b.init is not None:
            conds = [
                f"{self.expr(iv.var)} == {self.expr(iv.dom.min)}"
                for iv in b.iter_vars
                if iv.kind == IterKind.REDUCE
            ]
            self.emit(f"if {' and '.join(conds)}:")
            self.nested(b.init)
        self.stmt(b.body)
        del self.live[len(self.live) - pushed :]

    def make_view(self, buf: Buffer, storage: str) -> int:
        view = self.name(buf, "np_")
        dt = _STORAGE_NUMPY[_TYPECODE[buf.dtype]]
        self.emit(f"{view} = _frombuffer({storage}, {str(dt)!r})")
        self.views[buf] = view
        self.live.append(view)
        return 1

    def generate(self) -> str:
        f = self.func
        params = [self.name(p, "b_") for p in f.params]
        self.lines = []
        self.functions.append(self.lines)
        self.lines.append("def _kernel(_args):")
        for i, name in enumerate(params):
            self.emit(f"{name} = _args[{i}]")
            self.live.append(name)
            if f.params[i] in self.addressed:
                self.make_view(f.params[i], name)
        self.stmt(f.body)
        if len(self.lines) == 1:
            self.emit("pass")
        # The kernel goes last so the outlined functions are defined when it runs.
        ordered = self.functions[1:] + self.functions[:1]
        return "\n\n".join("\n".join(fn) for fn in ordered) + "\n"


def _thread_extents(s: Stmt) -> dict[str, int]:
    """The number of threads along each ``thread.*`` axis (the largest bound loop)."""
    out: dict[str, int] = {}
    for node in walk_stmts(s):
        if isinstance(node, For) and node.thread_tag and node.thread_tag.startswith("thread."):
            assert isinstance(node.extent, IntImm)
            out[node.thread_tag] = max(out.get(node.thread_tag, 1), node.extent.value)
    return out


def _narrower(dst: DType, src: DType) -> bool:
    return dst.itemsize < src.itemsize or (dst == DType.uint8 and src != DType.uint8)


class _Addressed(StmtVisitor):
    def __init__(self) -> None:
        self.found: set[Buffer] = set()

    def visit_address_of(self, e: AddressOf) -> None:
        self.found.add(e.buffer)
        super().visit_address_of(e)


def _addressed_buffers(s: Stmt) -> set[Buffer]:
    v = _Addressed()
    v.visit_stmt(s)
    return v.found


class _Written(StmtVisitor):
    def __init__(self) -> None:
        self.found: set[Buffer] = set()

    def visit_buffer_store(self, s: BufferStore) -> None:
        self.found.add(s.buffer)
        super().visit_buffer_store(s)

    def visit_address_of(self, e: AddressOf) -> None:
        self.found.add(e.buffer)
        super().visit_address_of(e)


class Kernel:
    """A PrimFunc compiled to Python."""

    def __init__(self, func: PrimFunc, check_bounds: bool) -> None:
        check_thread_locals(func)
        gen = _Gen(func, check_bounds)
        self.source = gen.generate()
        namespace = gen.globals
        code = compile(self.source, f"<tir {func.name}>", "exec")
        exec(code, namespace)
        self.fn: Callable[[list[array.array[Any]]], None] = namespace["_kernel"]
        self.params = func.params
        w = _Written()
        w.visit_stmt(func.body)
        self.written = [p in w.found for p in func.params]

    def __call__(self, args: Sequence[NDArray]) -> None:
        if len(args) != len(self.params):
            raise ValueError(f"expected {len(self.params)} arguments, got {len(args)}")
        storages: list[array.array[Any]] = []
        for p, a in zip(self.params, args, strict=True):
            if tuple(a.shape) != p.shape:
                raise ValueError(f"argument {p.name}: shape {a.shape} != {p.shape}")
            if a.dtype != p.dtype.numpy:
                raise ValueError(f"argument {p.name}: dtype {a.dtype} != {p.dtype}")
            tc = _TYPECODE[p.dtype]
            storage = array.array(tc)
            storage.frombytes(np.ascontiguousarray(a, dtype=_STORAGE_NUMPY[tc]).tobytes())
            storages.append(storage)
        self.fn(storages)
        for p, a, storage, written in zip(self.params, args, storages, self.written, strict=True):
            if written:
                data = np.frombuffer(storage, dtype=_STORAGE_NUMPY[_TYPECODE[p.dtype]])
                a[...] = data.reshape(p.shape).astype(a.dtype)


_CACHE: weakref.WeakKeyDictionary[PrimFunc, dict[bool, Kernel]] = weakref.WeakKeyDictionary()


def compile_kernel(func: PrimFunc, *, check_bounds: bool = False) -> Kernel:
    """The compiled Python kernel of ``func`` (cached on the function object)."""
    per_func = _CACHE.get(func)
    if per_func is None:
        per_func = {}
        _CACHE[func] = per_func
    kernel = per_func.get(check_bounds)
    if kernel is None:
        kernel = Kernel(func, check_bounds)
        per_func[check_bounds] = kernel
    return kernel


def interpret(func: PrimFunc, args: Sequence[NDArray], *, check_bounds: bool = False) -> None:
    """Run ``func`` on NumPy arrays, one per parameter, writing outputs in place.

    Arrays must have the parameters' shapes and dtypes. With ``check_bounds``
    every buffer index is checked (slower; for debugging schedules).
    """
    compile_kernel(func, check_bounds=check_bounds)(args)


def python_source(func: PrimFunc, *, check_bounds: bool = False) -> str:
    """The Python source the interpreter runs for ``func``."""
    return compile_kernel(func, check_bounds=check_bounds).source
