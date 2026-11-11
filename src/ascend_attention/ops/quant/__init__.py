"""FP8 quantisation, as **fake-quant** primitives: quantise and immediately dequantise.

Why this shape and not "a quantised GEMM kernel": the thing that decides whether a quantised model is
correct is the *scale arithmetic* - which axis a scale belongs to, whether the cast saturates or produces
inf, whether the scale multiplies before or after the rounding. A kernel gets those right by construction
or not at all; a reference makes them checkable, and every later kernel is graded against it. So this
family has no speed claim (definitional 1.0) and says so in its ratio note.

The three semantics that are easy to get silently wrong, and the choices made here:

* **Saturate, do not overflow.** ``x / scale`` is clamped to the format's finite maximum (448 for e4m3,
  57344 for e5m2) *before* the cast. Torch's own cast of an out-of-range value gives you ``inf``/``nan``
  depending on the format, which a serving stack would then propagate - clamping is what production
  kernels do, and it is what makes "quantise an activation outlier" a bounded error instead of a crash.
* **The scale divides before rounding and multiplies after.** ``dequant(quant(x)) = clamp(x/s).to(f8).to(x.dtype) * s``.
  Dividing after the cast is a different function (it rounds the wrong quantity) and is off by up to one
  ulp of the scale.
* **fp8 values are the *definition*.** The cast is ``torch.float8_e4m3fn``/``e5m2`` - the same bit layout
  an Ascend fp8 kernel uses - so the reference cannot drift from the hardware's grid; only rounding mode
  (round-to-nearest-even) is shared, which both sides do.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ...plan import OpPlan
from ...spec import REGISTRY, ImplSpec, ShapeSpec, dim_in, dim_min

__all__ = [
    "FORMATS",
    "GEMM_OP",
    "OP",
    "fmts",
    "fp8_gemm_reference",
    "max_finite",
    "plan_dequant",
    "plan_gemm",
    "quant_dequant",
    "run_dequant",
    "run_gemm",
]

OP = "quant_dequant_fp8"
GEMM_OP = "quant_fp8_gemm"

#: Format name -> (torch dtype, largest finite magnitude). The maximum is not derived from the exponent
#: bits at run time on purpose: it is a constant of the format, and a test pins it against torch's own
#: ``finfo``.
FORMATS = {
    "e4m3": (torch.float8_e4m3fn, 448.0),
    "e5m2": (torch.float8_e5m2, 57344.0),
}

_DTYPES = ("bfloat16", "float16", "float32")
_AXES = ("tensor", "token", "channel")


def fmts(fmt: str):
    try:
        return FORMATS[str(fmt)]
    except KeyError:
        raise ValueError(f"fp8 format {fmt!r} is not one of {sorted(FORMATS)}") from None


def max_finite(fmt: str) -> float:
    return fmts(fmt)[1]


def quant_dequant(
    x: torch.Tensor,
    scale: torch.Tensor | float,
    fmt: str = "e4m3",
    axis: str = "tensor",
) -> torch.Tensor:
    """Fake-quantise ``x`` to fp8 and return the dequantised values in ``x.dtype``.

    ``axis`` says what the scale applies to, and is checked against the scale's shape so a per-token
    scale cannot be broadcast along the wrong dimension silently:

    * ``"tensor"``  - one scalar scale for the whole tensor;
    * ``"token"``   - ``[rows]``, one per row of the leading axis (the activation case);
    * ``"channel"`` - ``[cols]``, one per column of the trailing axis (the weight case).
    """
    f8, limit = fmts(fmt)
    if str(axis) not in _AXES:
        raise ValueError(f"axis {axis!r} is not one of {_AXES}")
    if not torch.is_floating_point(x):
        raise ValueError(f"quant_dequant expects a float tensor, got {x.dtype}")

    s = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    if not torch.isfinite(s).all():
        raise ValueError("quant_dequant: scale must be finite")
    if (s <= 0).any():
        raise ValueError("quant_dequant: scale must be > 0 (a zero scale divides by zero, a negative one flips the grid)")

    if axis == "tensor":
        if s.numel() != 1:
            raise ValueError(f"axis='tensor' wants a scalar scale, got shape {tuple(s.shape)}")
        shape = ()
    elif axis == "token":
        if s.dim() != 1 or s.numel() != x.shape[0]:
            raise ValueError(
                f"axis='token' wants a [{x.shape[0]}] scale, got {tuple(s.shape)}"
            )
        shape = (-1,) + (1,) * (x.dim() - 1)
    else:
        if s.dim() != 1 or s.numel() != x.shape[-1]:
            raise ValueError(
                f"axis='channel' wants a [{x.shape[-1]}] scale, got {tuple(s.shape)}"
            )
        shape = (1,) * (x.dim() - 1) + (-1,)

    s = s.reshape(shape)
    scaled = (x / s).clamp(-limit, limit)
    return scaled.to(f8).to(x.dtype) * s


def fp8_gemm_reference(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    fmt: str = "e4m3",
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """``(a * a_scale) @ (b * b_scale)`` where ``a``/``b`` are fp8-stored operands.

    The two operands carry different scale axes on purpose, because that is what serving stacks do:
    activations are quantised per token (``a_scale``: ``[M]``) and weights per output channel
    (``b_scale``: ``[N]``). Dequantising *before* the matmul is the reference definition; a real
    kernel keeps the operands in fp8 and folds the scales afterwards, which is only equivalent if the
    scales are applied to the right axes - that equivalence is what the tests check.
    """
    if str(fmt) not in FORMATS:
        raise ValueError(f"fp8 format {fmt!r} is not one of {sorted(FORMATS)}")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"fp8_gemm expects 2-D operands, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"fp8_gemm inner dims disagree: {tuple(a.shape)} vs {tuple(b.shape)}")
    # fp8 has no arithmetic: ``promote_types(fp8, fp8)`` is fp8, and multiplying that on CPU falls into a
    # slow, silently-different path. The compute dtype is therefore explicit, not promoted.
    target = out_dtype or torch.bfloat16
    a_scale = torch.as_tensor(a_scale, dtype=target, device=a.device)
    b_scale = torch.as_tensor(b_scale, dtype=target, device=b.device)
    if a_scale.numel() not in (1, a.shape[0]):
        raise ValueError(f"a_scale must be scalar or [{a.shape[0]}], got {tuple(a_scale.shape)}")
    if b_scale.numel() not in (1, b.shape[1]):
        raise ValueError(f"b_scale must be scalar or [{b.shape[1]}], got {tuple(b_scale.shape)}")
    a_deq = a.to(target) * a_scale.reshape(-1, 1) if a_scale.numel() > 1 else a.to(target) * a_scale
    b_deq = b.to(target) * b_scale.reshape(1, -1) if b_scale.numel() > 1 else b.to(target) * b_scale
    return a_deq @ b_deq


_DEQUANT_SHAPE = ShapeSpec(
    ndim=2,
    dims=(dim_min(1), dim_in(1024, 4096, 7168, 8192, 152064)),
    dtypes=frozenset(_DTYPES),
    contiguous=False,
)

DEQUANT_REFERENCE = ImplSpec(
    op=OP,
    impl="torch.fake_quant",
    form="torch",
    device="any",
    shape=_DEQUANT_SHAPE,
    kv_layouts=frozenset({"ND"}),
    graph="capture_safe",
    plan_kind="stateless",
    enabled_by_default=True,
    measured_ratio=1.0,
    bench="bench/bench_quant.py",
    ratio_note=(
        "definitional 1.0: the only implementation and the definition every future fp8 kernel is graded "
        "against. Two elementwise passes plus a cast; the ledger records absolute latency at the "
        "activation shape (4096 x 7168)."
    ),
    note=(
        "fake-quant: clamp(x/scale) -> fp8 cast -> dequantise with the same scale. Saturates at the "
        "format maximum instead of overflowing to inf; per-tensor/per-token/per-channel scale, axis "
        "checked against the scale shape."
    ),
)

GEMM_REFERENCE = ImplSpec(
    op=GEMM_OP,
    impl="torch.dequant_gemm",
    form="torch",
    device="any",
    shape=_DEQUANT_SHAPE,
    kv_layouts=frozenset({"ND"}),
    graph="capture_safe",
    plan_kind="stateless",
    enabled_by_default=True,
    measured_ratio=1.0,
    bench="bench/bench_quant.py",
    ratio_note=(
        "definitional 1.0: dequantise both operands, then matmul. A real fp8 kernel keeps the operands "
        "quantised and folds the scales afterwards; this is the definition that equivalence is checked "
        "against, not a fast path."
    ),
    note=(
        "fp8 x fp8 -> out_dtype: activation scale per token [M], weight scale per channel [N]. The "
        "scale-axis equivalence (dequant-then-matmul == matmul-then-scale) is pinned by tests."
    ),
)

REGISTRY.register(DEQUANT_REFERENCE)
REGISTRY.register(GEMM_REFERENCE)


def plan_dequant(impl: ImplSpec, **pure: Any) -> OpPlan:
    rows = int(pure["rows"])
    cols = int(pure["cols"])
    fmt = str(pure.get("fmt", "e4m3"))
    axis = str(pure.get("axis", "tensor"))
    dtype = str(pure.get("dtype", "bfloat16"))
    fmts(fmt)
    if axis not in _AXES:
        raise ValueError(f"{OP}: axis {axis!r} is not one of {_AXES}")
    if dtype not in _DTYPES:
        raise ValueError(f"{OP}: dtype {dtype!r} is outside {sorted(_DTYPES)}")
    if impl.shape is not None and not impl.shape.matches((rows, cols), dtype):
        raise ValueError(f"{OP}/{impl.impl}: ({rows}, {cols}) dtype={dtype} is outside the declared contract")

    def execute(x: torch.Tensor, scale) -> torch.Tensor:
        if tuple(x.shape) != (rows, cols):
            raise ValueError(f"{OP}/{impl.impl}: x must be {(rows, cols)}, got {tuple(x.shape)}")
        return quant_dequant(x, scale, fmt=fmt, axis=axis)

    return OpPlan(
        op=OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(rows, cols),
        dtypes=(dtype,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={"fmt": fmt, "axis": axis, "max_finite": max_finite(fmt)},
        execute=execute,
    )


def plan_gemm(impl: ImplSpec, **pure: Any) -> OpPlan:
    m = int(pure["m"])
    k = int(pure["k"])
    n = int(pure["n"])
    fmt = str(pure.get("fmt", "e4m3"))
    out_dtype = pure.get("out_dtype")
    fmts(fmt)

    def execute(a: torch.Tensor, a_scale, b: torch.Tensor, b_scale) -> torch.Tensor:
        if tuple(a.shape) != (m, k) or tuple(b.shape) != (k, n):
            raise ValueError(
                f"{GEMM_OP}/{impl.impl}: a must be {(m, k)} and b {(k, n)}, got "
                f"{tuple(a.shape)} and {tuple(b.shape)}"
            )
        return fp8_gemm_reference(a, a_scale, b, b_scale, fmt=fmt, out_dtype=out_dtype)

    return OpPlan(
        op=GEMM_OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(m, n),
        dtypes=(str(out_dtype or "bfloat16"),),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={"m": m, "k": k, "n": n, "fmt": fmt},
        execute=execute,
    )


def run_dequant(p: OpPlan, x: torch.Tensor, scale):
    return p(x, scale)


def run_gemm(p: OpPlan, a, a_scale, b, b_scale):
    return p(a, a_scale, b, b_scale)


REGISTRY.op(
    OP,
    doc="FP8 fake-quant: clamp -> cast -> dequantise with the same scale (saturating). The definition "
    "future fp8 kernels are graded against.",
)
REGISTRY.op(
    GEMM_OP,
    doc="FP8 x FP8 GEMM reference: dequantise (per-token A, per-channel B) then matmul.",
)
REGISTRY.planner(OP, plan_dequant)
REGISTRY.planner(GEMM_OP, plan_gemm)
