"""Recognizers for operators that exporters spell out with primitive nodes.

* LayerNorm: ``(x - mean) / sqrt(var + eps)`` with ``mean`` and ``var`` from
  ReduceMean over trailing axes (division or multiplication by a reciprocal
  square root), followed by optional per-element scale and shift, which are
  absorbed into the LayerNorm's scale and bias.
* GELU, exact: ``x * 0.5 * (1 + erf(x / sqrt(2)))`` in any association of
  the product, with the erf argument written as a division or a product.
* GELU, tanh approximation: ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))``
  with the cube written as Pow or as products.
* SiLU ``x * sigmoid(x)``, HardSwish ``x * HardSigmoid(x)`` (alpha 1/6,
  beta 1/2), and Rsqrt ``1 / sqrt(x)``.

Constants are matched with a relative tolerance of 1e-4, because exporters
round them (``1.4142135`` for sqrt(2)).
"""

from __future__ import annotations

import math

import numpy as np

from tessera.ir.graph import Node, Value
from tessera.passes.rewrite import Rewriter, Rule, scalar


def _near(value: float | None, target: float) -> bool:
    return value is not None and math.isclose(value, target, rel_tol=1e-4)


def _producer(value: Value, op: str) -> Node | None:
    node = value.producer
    return node if node is not None and node.op == op else None


def _operands(node: Node, op: str) -> list[tuple[Value, Value]]:
    """For a commutative binary ``op`` node, both (operand, other) orders."""
    if node.op != op:
        return []
    a, b = node.inputs
    return [(a, b), (b, a)]


def _factors(value: Value, rw: Rewriter, depth: int = 3) -> list[Value]:
    """Leaves of a product tree (through Mul nodes read only once)."""
    node = value.producer
    if node is None or node.op != "Mul" or depth == 0:
        return [value]
    out: list[Value] = []
    for v in node.inputs:
        inner = v.producer
        if inner is not None and inner.op == "Mul" and rw.single_reader(v):
            out += _factors(v, rw, depth - 1)
        else:
            out.append(v)
    return out


def _one_plus(value: Value, op: str) -> Value | None:
    """The argument ``z`` when ``value`` is ``1 + op(z)`` (in either order)."""
    add = _producer(value, "Add")
    if add is None:
        return None
    for term, one in _operands(add, "Add"):
        inner = _producer(term, op)
        if inner is not None and _near(scalar(one), 1.0):
            return inner.inputs[0]
    return None


def _is_scaled(value: Value, x: Value, factor: float) -> bool:
    """``value`` is ``x * factor`` or ``x / (1 / factor)``."""
    node = value.producer
    if node is None:
        return False
    if node.op == "Mul":
        return any(a is x and _near(scalar(c), factor) for a, c in _operands(node, "Mul"))
    if node.op == "Div":
        return node.inputs[0] is x and _near(scalar(node.inputs[1]), 1 / factor)
    return False


def _is_cube(value: Value, x: Value) -> bool:
    node = value.producer
    if node is None:
        return False
    if node.op == "Pow":
        return node.inputs[0] is x and _near(scalar(node.inputs[1]), 3.0)
    if node.op == "Mul":
        for a, b in _operands(node, "Mul"):
            square = _producer(b, "Mul")
            if a is x and square is not None and square.inputs == [x, x]:
                return True
            if a is x and _producer(b, "Pow") is not None:
                power = _producer(b, "Pow")
                assert power is not None
                if power.inputs[0] is x and _near(scalar(power.inputs[1]), 2.0):
                    return True
    return False


def _tanh_argument_matches(z: Value, x: Value) -> bool:
    """``z`` is ``sqrt(2/pi) * (x + 0.044715 * x^3)``."""
    scaled = z.producer
    if scaled is None or scaled.op != "Mul":
        return False
    for inner, c in _operands(scaled, "Mul"):
        if not _near(scalar(c), math.sqrt(2 / math.pi)):
            continue
        add = _producer(inner, "Add")
        if add is None:
            continue
        for a, cubic in _operands(add, "Add"):
            term = _producer(cubic, "Mul")
            if a is not x or term is None:
                continue
            for cube, k in _operands(term, "Mul"):
                if _near(scalar(k), 0.044715) and _is_cube(cube, x):
                    return True
    return False


def _gelu(rw: Rewriter, node: Node) -> Value | None:
    if node.op != "Mul" or not node.output.dtype.is_float:
        return None
    factors = _factors(node.output, rw)
    if len(factors) != 3:
        return None
    for half in factors:
        if not _near(scalar(half), 0.5):
            continue
        rest = [f for f in factors if f is not half]
        for x, phi in (rest, rest[::-1]):
            if x.type != node.output.type:
                continue
            z = _one_plus(phi, "Erf")
            if z is not None and _is_scaled(z, x, 1 / math.sqrt(2)):
                return rw.op("Gelu", x, name=node.name, approximate="none")
            z = _one_plus(phi, "Tanh")
            if z is not None and _tanh_argument_matches(z, x):
                return rw.op("Gelu", x, name=node.name, approximate="tanh")
    return None


def _silu_hardswish(rw: Rewriter, node: Node) -> Value | None:
    for x, gate in _operands(node, "Mul"):
        g = gate.producer
        if g is None or g.inputs[:1] != [x] or x.type != node.output.type:
            continue
        if g.op == "Sigmoid":
            return rw.op("Silu", x, name=node.name)
        if (
            g.op == "HardSigmoid"
            and _near(g.float_attr("alpha"), 1 / 6)
            and _near(g.float_attr("beta"), 0.5)
        ):
            return rw.op("HardSwish", x, name=node.name)
    return None


def _rsqrt(rw: Rewriter, node: Node) -> Value | None:
    if node.op != "Div" or not _near(scalar(node.inputs[0]), 1.0):
        return None
    root = _producer(node.inputs[1], "Sqrt")
    if root is None or root.output.type != node.output.type:
        return None
    return rw.op("Rsqrt", root.inputs[0], name=node.name)


# -- LayerNorm ----------------------------------------------------------------------


def _trailing_mean(value: Value, of: Value | None = None) -> tuple[Value, int] | None:
    """(x, axis) when ``value`` is ReduceMean(x) over axes axis..rank-1, keeping dims."""
    node = _producer(value, "ReduceMean")
    if node is None or node.int_attr("keepdims") != 1:
        return None
    x = node.inputs[0]
    axes = node.ints_attr("axes")
    rank = x.type.rank
    if not axes or axes != tuple(range(axes[0], rank)) or (of is not None and x is not of):
        return None
    return x, axes[0]


def _is_square(value: Value, d: Value) -> bool:
    node = value.producer
    if node is None:
        return False
    if node.op == "Pow":
        return node.inputs[0] is d and _near(scalar(node.inputs[1]), 2.0)
    return node.op == "Mul" and node.inputs == [d, d]


def _variance_eps(value: Value, d: Value, axis: int) -> float | None:
    """epsilon when ``value`` is ``mean(d * d) + eps`` over the same axes."""
    add = _producer(value, "Add")
    if add is None:
        return None
    for var, eps in _operands(add, "Add"):
        epsilon = scalar(eps)
        found = _trailing_mean(var)
        if epsilon is None or eps.type.numel != 1 or found is None or found[1] != axis:
            continue
        if _is_square(found[0], d):
            return epsilon
    return None


def _normalizations(node: Node) -> list[tuple[Value, Value]]:
    """(d, v) pairs such that ``node`` computes ``d / sqrt(v)`` or ``d * rsqrt(v)``."""
    out: list[tuple[Value, Value]] = []
    if node.op == "Div":
        root = _producer(node.inputs[1], "Sqrt")
        if root is not None:
            out.append((node.inputs[0], root.inputs[0]))
    elif node.op == "Mul":
        for d, inv in _operands(node, "Mul"):
            r = _producer(inv, "Rsqrt")
            if r is not None:
                out.append((d, r.inputs[0]))
    return out


def _layernorm(rw: Rewriter, node: Node) -> Value | None:
    if not node.output.dtype.is_float:
        return None
    for d, denominator in _normalizations(node):
        sub = _producer(d, "Sub")
        if sub is None:
            continue
        x, mean = sub.inputs
        found = _trailing_mean(mean, of=x)
        if found is None or x.type != node.output.type:
            continue
        axis = found[1]
        eps = _variance_eps(denominator, d, axis)
        if eps is None:
            continue
        normalized = x.shape[axis:]
        ones = rw.const(np.ones(normalized, x.dtype.numpy), f"{node.name}_scale")
        zeros = rw.const(np.zeros(normalized, x.dtype.numpy), f"{node.name}_bias")
        return rw.op("LayerNorm", x, ones, zeros, name=node.name, axis=axis, epsilon=eps)
    return None


def _absorb_affine(rw: Rewriter, node: Node) -> Value | None:
    """LayerNorm(...) * c and LayerNorm(...) + c fold c into its scale and bias."""
    if node.op not in ("Mul", "Add"):
        return None
    for value, c in _operands(node, node.op):
        ln = _producer(value, "LayerNorm")
        if ln is None or c.data is None or not rw.single_reader(value):
            continue
        if value.type != node.output.type:
            continue
        axis = ln.int_attr("axis")
        rank = value.type.rank
        aligned = (1,) * (rank - c.type.rank) + c.shape
        if any(d != 1 for d in aligned[:axis]):
            continue  # c varies along an axis that is not normalized
        per_element = np.broadcast_to(c.data.reshape(aligned[axis:]), value.shape[axis:])
        x, scale, bias = ln.inputs
        if scale.data is None or bias.data is None:
            continue
        dtype = x.dtype.numpy
        if node.op == "Mul":
            scale = rw.const((scale.data * per_element).astype(dtype), f"{ln.name}_scale")
            bias = rw.const((bias.data * per_element).astype(dtype), f"{ln.name}_bias")
        else:
            bias = rw.const((bias.data + per_element).astype(dtype), f"{ln.name}_bias")
        return rw.op(
            "LayerNorm",
            x,
            scale,
            bias,
            name=ln.name,
            axis=axis,
            epsilon=ln.float_attr("epsilon"),
        )
    return None


PATTERNS: list[Rule] = [_layernorm, _absorb_affine, _gelu, _silu_hardswish, _rsqrt]
