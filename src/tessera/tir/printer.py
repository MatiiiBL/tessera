"""A readable text form of loop-IR functions (``tir.script``).

The output looks like Python: loops print as ``for i in range(16):``, blocks
as ``with block("C"):`` followed by their iteration variables, the regions
they read and write, an optional ``with init():`` and their body. Variables
and buffers are printed by name; when two different objects share a name the
later one gets a numeric suffix, so the text is unambiguous.

The same printer, with every name replaced by its position of first
appearance, gives the canonical form that ``structural_hash`` hashes.
"""

from __future__ import annotations

import math

from tessera.dtype import INDEX, DType
from tessera.tir.expr import (
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
    TRUE,
    Annotations,
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
    Range,
    SeqStmt,
    Stmt,
)

_BINARY_SYMBOL = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "floordiv": "//",
    "floormod": "%",
    "pow": "**",
}
_BINARY_PREC = {"add": 6, "sub": 6, "mul": 7, "div": 7, "floordiv": 7, "floormod": 7, "pow": 9}
_COMPARE_SYMBOL = {"lt": "<", "le": "<=", "eq": "==", "ne": "!=", "gt": ">", "ge": ">="}
_ATOM = 10


class _Names:
    """Display names for objects compared by identity."""

    def __init__(self, anonymous: bool, prefix: str) -> None:
        self.anonymous = anonymous
        self.prefix = prefix
        self.by_id: dict[int, str] = {}
        self.keep: list[object] = []  # keeps ids stable while printing
        self.used: set[str] = set()

    def get(self, obj: object, name: str) -> str:
        key = id(obj)
        found = self.by_id.get(key)
        if found is not None:
            return found
        if self.anonymous:
            text = f"{self.prefix}{len(self.by_id)}"
        else:
            text, n = name, 1
            while text in self.used:
                text = f"{name}_{n}"
                n += 1
        self.used.add(text)
        self.by_id[key] = text
        self.keep.append(obj)
        return text


class Printer:
    """Prints expressions, statements and functions; one instance per printed tree."""

    def __init__(self, anonymous: bool = False) -> None:
        self.anonymous = anonymous
        self.vars = _Names(anonymous, "v")
        self.buffers = _Names(anonymous, "b")
        self.blocks = _Names(anonymous, "blk")
        self.lines: list[str] = []

    # -- expressions --------------------------------------------------------

    def expr(self, e: Expr) -> str:
        return self._expr(e)[0]

    def _expr(self, e: Expr) -> tuple[str, int]:
        if isinstance(e, IntImm):
            return self._int(e), _ATOM
        if isinstance(e, FloatImm):
            return self._float(e), _ATOM
        if isinstance(e, Var):
            name = self.vars.get(e, e.name)
            typed = self.anonymous and e.type != INDEX
            return (f"{name}:{e.type}" if typed else name), _ATOM
        if isinstance(e, Binary):
            if e.op in ("min", "max"):
                return f"{e.op}({self.expr(e.a)}, {self.expr(e.b)})", _ATOM
            prec = _BINARY_PREC[e.op]
            a = self._wrap(e.a, prec, right=e.op == "pow")
            b = self._wrap(e.b, prec, right=e.op != "pow")
            return f"{a} {_BINARY_SYMBOL[e.op]} {b}", prec
        if isinstance(e, Compare):
            return f"{self._wrap(e.a, 5)} {_COMPARE_SYMBOL[e.op]} {self._wrap(e.b, 5)}", 4
        if isinstance(e, Logical):
            prec = 2 if e.op == "and" else 1
            return f"{self._wrap(e.a, prec)} {e.op} {self._wrap(e.b, prec, right=True)}", prec
        if isinstance(e, Not):
            return f"not {self._wrap(e.a, 3)}", 3
        if isinstance(e, Select):
            args = ", ".join(self.expr(x) for x in (e.cond, e.true_value, e.false_value))
            return f"select({args})", _ATOM
        if isinstance(e, Cast):
            return f"{e.type}({self.expr(e.value)})", _ATOM
        if isinstance(e, Call):
            suffix = "" if e.type == DType.float32 else f"[{e.type}]"
            return f"{e.name}{suffix}({', '.join(self.expr(a) for a in e.args)})", _ATOM
        if isinstance(e, BufferLoad):
            return f"{self.buffer(e.buffer)}[{self._indices(e.indices)}]", _ATOM
        if isinstance(e, AddressOf):
            return f"&{self.buffer(e.buffer)}[{self._indices(e.indices)}]", _ATOM
        raise TypeError(f"cannot print expression {e!r}")

    def _wrap(self, e: Expr, prec: int, right: bool = False) -> str:
        text, p = self._expr(e)
        if p < prec or (right and p == prec):
            return f"({text})"
        return text

    def _indices(self, indices: tuple[Expr, ...]) -> str:
        return ", ".join(self.expr(i) for i in indices) if indices else ""

    def _int(self, e: IntImm) -> str:
        if e.type == DType.bool_:
            return "True" if e.value else "False"
        if e.type == INDEX:
            return str(e.value)
        return f"{e.type}({e.value})"

    def _float(self, e: FloatImm) -> str:
        v = e.value
        if math.isfinite(v):
            text = repr(v)
        else:
            text = "nan" if math.isnan(v) else ("-inf" if v < 0 else "inf")
        return text if e.type == DType.float32 else f"{e.type}({text})"

    def buffer(self, b: Buffer) -> str:
        return self.buffers.get(b, b.name)

    def buffer_type(self, b: Buffer) -> str:
        return f"{b.dtype}[{', '.join(map(str, b.shape))}]"

    def region(self, r: BufferRegion) -> str:
        parts = []
        for rng in r.region:
            if isinstance(rng.extent, IntImm) and rng.extent.value == 1:
                parts.append(self.expr(rng.min))
            else:
                lo = self.expr(rng.min)
                if isinstance(rng.min, IntImm) and isinstance(rng.extent, IntImm):
                    hi = str(rng.min.value + rng.extent.value)
                else:
                    hi = f"{self._wrap(rng.min, 6)} + {self._wrap(rng.extent, 6, right=True)}"
                parts.append(f"{lo}:{hi}")
        return f"{self.buffer(r.buffer)}[{', '.join(parts)}]"

    def _range(self, r: Range) -> str:
        if isinstance(r.min, IntImm) and r.min.value == 0:
            return self.expr(r.extent)
        if isinstance(r.min, IntImm) and isinstance(r.extent, IntImm):
            return f"({r.min.value}, {r.min.value + r.extent.value})"
        return f"({self.expr(r.min)}, {self._wrap(r.min, 6)} + {self._wrap(r.extent, 6, True)})"

    def _annotations(self, annotations: Annotations) -> str:
        items = ", ".join(f"{k!r}: {v!r}" for k, v in annotations)
        return f"{{{items}}}"

    # -- statements ---------------------------------------------------------

    def emit(self, indent: int, text: str) -> None:
        self.lines.append("    " * indent + text)

    def stmt(self, s: Stmt, indent: int) -> None:
        if isinstance(s, SeqStmt):
            if not s.seq:
                self.emit(indent, "pass")
            for c in s.seq:
                self.stmt(c, indent)
        elif isinstance(s, For):
            self._for(s, indent)
        elif isinstance(s, BlockRealize):
            self._block_realize(s, indent)
        elif isinstance(s, BufferStore):
            target = f"{self.buffer(s.buffer)}[{self._indices(s.indices)}]"
            self.emit(indent, f"{target} = {self.expr(s.value)}")
        elif isinstance(s, IfThenElse):
            self.emit(indent, f"if {self.expr(s.condition)}:")
            self.stmt(s.then_case, indent + 1)
            if s.else_case is not None:
                self.emit(indent, "else:")
                self.stmt(s.else_case, indent + 1)
        elif isinstance(s, LetStmt):
            name = self.vars.get(s.var, s.var.name)
            self.emit(indent, f"{name}: {s.var.type} = {self.expr(s.value)}")
            self.stmt(s.body, indent)
        elif isinstance(s, Evaluate):
            self.emit(indent, self.expr(s.value))
        elif isinstance(s, Block):
            self._block(s, (), TRUE, indent)
        else:
            raise TypeError(f"cannot print statement {s!r}")

    def _for(self, s: For, indent: int) -> None:
        var = self.vars.get(s.loop_var, s.loop_var.name)
        if isinstance(s.min, IntImm) and s.min.value == 0:
            bounds = self.expr(s.extent)
        else:
            bounds = f"{self.expr(s.min)}, {self._wrap(s.min, 6)} + {self._wrap(s.extent, 6, True)}"
        fn = "range" if s.kind == ForKind.SERIAL else s.kind.value
        extra = ""
        if s.thread_tag is not None:
            extra += f", thread={s.thread_tag!r}"
        if s.annotations:
            extra += f", annotations={self._annotations(s.annotations)}"
        self.emit(indent, f"for {var} in {fn}({bounds}{extra}):")
        self.stmt(s.body, indent + 1)

    def _block_realize(self, s: BlockRealize, indent: int) -> None:
        self._block(s.block, s.iter_values, s.predicate, indent)

    def _block(self, b: Block, values: tuple[Expr, ...], predicate: Expr, indent: int) -> None:
        self.emit(indent, f"with block({self.blocks.get(b, b.name)!r}):")
        inner = indent + 1
        for i, iv in enumerate(b.iter_vars):
            name = self.vars.get(iv.var, iv.var.name)
            kind = "spatial" if iv.kind == IterKind.SPATIAL else "reduce"
            binding = f", {self.expr(values[i])}" if i < len(values) else ""
            self.emit(inner, f"{name} = {kind}({self._range(iv.dom)}{binding})")
        if predicate != TRUE:
            self.emit(inner, f"where({self.expr(predicate)})")
        if b.reads:
            self.emit(inner, f"reads({', '.join(self.region(r) for r in b.reads)})")
        if b.writes:
            self.emit(inner, f"writes({', '.join(self.region(r) for r in b.writes)})")
        if b.annotations:
            self.emit(inner, f"annotate({self._annotations(b.annotations)})")
        for buf in b.alloc_buffers:
            self._alloc(buf, inner)
        if b.init is not None:
            self.emit(inner, "with init():")
            self.stmt(b.init, inner + 1)
        self.stmt(b.body, inner)

    def _alloc(self, buf: Buffer, indent: int) -> None:
        scope = "" if buf.scope == "global" else f", {buf.scope!r}"
        self.emit(indent, f"{self.buffer(buf)} = alloc_buffer({self.buffer_type(buf)}{scope})")

    def func(self, f: PrimFunc) -> str:
        name = "f" if self.anonymous else f.name
        params = ", ".join(f"{self.buffer(p)}: {self.buffer_type(p)}" for p in f.params)
        self.emit(0, "@prim_func")
        self.emit(0, f"def {name}({params}):")
        if f.attrs:
            self.emit(1, f"func_attr({self._annotations(f.attrs)})")
        body = f.body
        if _is_plain_root(body):
            assert isinstance(body, BlockRealize)
            for buf in body.block.alloc_buffers:
                self._alloc(buf, 1)
            self.stmt(body.block.body, 1)
        else:
            self.stmt(body, 1)
        return "\n".join(self.lines)


def _is_plain_root(s: Stmt) -> bool:
    return (
        isinstance(s, BlockRealize)
        and not s.iter_values
        and s.predicate == TRUE
        and s.block.name == "root"
        and not s.block.iter_vars
        and not s.block.reads
        and not s.block.writes
        and s.block.init is None
        and not s.block.annotations
    )


def script(node: PrimFunc | Stmt | Expr) -> str:
    """The readable text form of a function, statement or expression."""
    p = Printer()
    if isinstance(node, PrimFunc):
        return p.func(node)
    if isinstance(node, Expr):
        return p.expr(node)
    p.stmt(node, 0)
    return "\n".join(p.lines)


def canonical_text(func: PrimFunc) -> str:
    """``func`` printed with every name replaced by its order of first appearance."""
    return Printer(anonymous=True).func(func)
