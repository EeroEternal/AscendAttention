"""``gdn_gating`` — the bring-up operator for the ``stateless`` plan kind.

Why this one first: it is the only GDN file in TileInfer that is importable without a kernel, it has
a dedicated independent oracle (``reference_gdn_gating``), and it is a pure elementwise composition
— so it exercises the whole spine (spec -> registry -> gate -> plan -> run -> manifest) with zero
device risk.  Its device kernel rating lives on the candidate that will replace it (P2).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch

from ...plan import OpPlan
from ...spec import REGISTRY, ImplSpec, ShapeSpec, dim_min, dim_mul
from .gating import fused_gdn_gating, reference_gdn_gating
from .leaves import chunk_local_cumsum, l2norm_fwd

__all__ = [
    "OP",
    "BASELINE",
    "CONV_OP",
    "CONV_BASELINE",
    "CONV_KERNEL",
    "plan_gating",
    "run_gating",
    "plan_conv",
    "run_conv",
    "chunk_local_cumsum",
    "l2norm_fwd",
]

OP = "gdn_gating"

#: The torch composition is the baseline *by definition*: it is what the kernel must beat, and what
#: ``TILEINFER_OPS_CANDIDATES=off`` falls back to.  Ratio 1.0 is therefore a fact, not a claim.
BASELINE = ImplSpec(
    op=OP,
    impl="torch.gating",
    form="torch",
    device="any",
    shape=ShapeSpec(
        ndim=2,
        dims=(None, dim_min(1)),
        dtypes=frozenset({"bfloat16", "float16", "float32"}),
        contiguous=True,
    ),
    kv_layouts=frozenset({"ND"}),
    graph="eager_only",
    plan_kind="stateless",
    enabled_by_default=True,
    measured_ratio=1.0,
    ratio_note=(
        "baseline by definition (the reference this op is judged against); no A/B against the "
        "vendor triton fused_gdn_gating_patch has been recorded yet, so this is parity, not a win"
    ),
    bench="bench/bench_gdn_gating.py",
    provenance="tileinfer/kernels/gdn/__init__.py (E20 evidence, 2026-11-10)",
    note="bring-up sample for the stateless plan kind; out-of-place, fixed shape",
)

REGISTRY.register(BASELINE)


def plan_gating(
    impl: ImplSpec,
    *,
    shape: Optional[Sequence[int]] = None,
    dtype: Any = None,
    **pure: Any,
) -> OpPlan:
    """Pure-data plan: needs only ``(T, H)`` and the dtype — never the tensors themselves."""
    if shape is None:
        raise ValueError("gdn_gating.plan needs shape=(T, H)")
    shape = tuple(int(d) for d in shape)
    if len(shape) != 2:
        raise ValueError(f"gdn_gating expects a 2-D (T, H) shape, got {shape}")
    dtype_name = str(dtype) if dtype is not None else "bfloat16"
    if "." in dtype_name:
        dtype_name = dtype_name.rsplit(".", 1)[-1]
    if impl.shape is not None and not impl.shape.matches(shape, dtype_name):
        raise ValueError(f"gdn_gating: shape {shape} / dtype {dtype_name} not accepted")

    expected_dtype = dtype_name
    expected_shape = shape

    def execute(
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cheap argument checks: metadata only, no sync, no allocation beyond the outputs.
        _check(a, "a", expected_shape)
        _check(b, "b", expected_shape)
        fn = reference_gdn_gating if impl.impl.endswith("reference") else fused_gdn_gating
        return fn(A_log, a, b, dt_bias, beta=beta, threshold=threshold)

    return OpPlan(
        op=OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=expected_shape,
        dtypes=(expected_dtype,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={"threshold": 20.0},
        execute=execute,
    )


def _check(tensor: torch.Tensor, name: str, shape: Tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"gdn_gating: {name} has shape {tuple(tensor.shape)} != plan {shape}")
    if not tensor.is_contiguous():
        raise ValueError(f"gdn_gating: {name} must be contiguous (plan declares contiguous=True)")


def run_gating(p: OpPlan, A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
               dt_bias: torch.Tensor, **kw: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    return p(A_log, a, b, dt_bias, **kw)


REGISTRY.planner(OP, plan_gating)


# --------------------------------------------------------------------- gdn_conv_update
#
# The first op whose candidate has a *recorded* op-level win (TileInfer measured 42.5x on the
# kernel) and whose end-to-end win is **zero** (0.995x).  It is therefore registered, measured and
# deliberately left opt-in: enabling it by default would buy nothing and would hide the discrepancy
# between "the kernel is fast" and "the step is faster".  That gap is the whole reason the ledger
# records both levels.

CONV_OP = "gdn_conv_update"

CONV_SHAPE = ShapeSpec(
    ndim=1,
    dims=(dim_mul(32),),
    dtypes=frozenset({"bfloat16"}),
    contiguous=True,
)

CONV_BASELINE = ImplSpec(
    op=CONV_OP,
    impl="torch.causal_conv1d_update",
    form="torch",
    device="any",
    shape=CONV_SHAPE,
    kv_layouts=frozenset({"ND"}),
    graph="eager_only",
    plan_kind="stateless",
    enabled_by_default=True,
    measured_ratio=1.0,
    ratio_note="baseline by definition (the torch port the engine runs without a kernel)",
    bench="bench/bench_gdn_conv.py",
    provenance="tileinfer/vllm/ops/fla/causal_conv1d.py",
    note="4-tap depthwise conv + bias + SiLU + state shift, token-major decode form",
)

CONV_KERNEL = ImplSpec(
    op=CONV_OP,
    impl="tilelang.conv950",
    form="tilelang",
    device="ascend950",
    shape=CONV_SHAPE,
    kv_layouts=frozenset({"ND"}),
    # allocates its output and mutates the conv state in place: correct, but not capture-safe
    graph="eager_only",
    plan_kind="kernel",
    enabled_by_default=False,
    measured_ratio=14.1505,
    ratio_note=(
        "op level 14.15x measured in this repo (C=4096, L=8, W=4, bf16; median of 6 alternating "
        "pairs, bench/results/gdn_conv_c4096.json, 2026-11-10); TileInfer recorded 42.5x at its own "
        "shape but **0.995x end-to-end**, so the candidate stays opt-in until an end-to-end arm "
        "passes"
    ),
    bench="bench/bench_gdn_conv.py",
    requires=frozenset({"tilelang"}),
    provenance="tileinfer/kernels/gdn/conv_update_ascend950.py",
    note="width==4, channels %% 32 == 0, bf16 state, dense [L, C, W-1] state view only",
)

REGISTRY.register(CONV_BASELINE)
REGISTRY.register(CONV_KERNEL)


def plan_conv(impl: ImplSpec, **pure: Any) -> OpPlan:
    """Plan one decode conv update.  The kernel is *compiled here* (plan stage), never in run."""
    import torch

    channels = int(pure["channels"])
    width = int(pure.get("width", 4))
    lines = int(pure["lines"])
    dtype_name = str(pure.get("dtype", "bfloat16"))
    if impl.shape is not None and not impl.shape.matches((channels,), dtype_name):
        raise ValueError(
            f"gdn_conv_update/{impl.impl}: channels={channels} dtype={dtype_name} "
            "is outside the declared contract"
        )

    if impl.form == "tilelang":
        if width != 4:
            raise ValueError("the conv kernel implements width == 4 only")
        from ...kernels.gdn.conv_update_ascend950 import (
            build_conv_update_kernel,
            gdn_conv_update_kernel_available,
        )

        if not gdn_conv_update_kernel_available():
            raise RuntimeError(
                "gdn_conv_update/tilelang.conv950 needs the TileLang Ascend toolchain"
            )
        kernel = build_conv_update_kernel(
            channels=channels, width=width, lines=lines, dtype=dtype_name
        )

        def execute(x, state, weight, bias, line: int, out=None):
            out = torch.empty(channels, dtype=getattr(torch, dtype_name), device=state.device) if out is None else out
            kernel(
                x.reshape(-1).contiguous(),
                state,
                weight,
                bias if bias.dtype == torch.float32 else bias.float(),
                int(line),
                out,
            )
            return out

    else:
        from ..fla.causal_conv1d import causal_conv1d_update

        def execute(x, state, weight, bias, line: int, out=None):
            out = causal_conv1d_update(
                x.reshape(1, -1),
                state,
                weight,
                bias,
                "silu",
                conv_state_indices=torch.tensor([line], dtype=torch.int32, device=state.device),
            ).reshape(-1)
            return out if out is None else out

    return OpPlan(
        op=CONV_OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(channels,),
        dtypes=(dtype_name,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={"channels": channels, "width": width, "lines": lines},
        handle=kernel if impl.form == "tilelang" else None,
        execute=execute,
    )


def run_conv(p: OpPlan, x, state, weight, bias, line: int, **kw: Any):
    return p(x, state, weight, bias, line, **kw)


REGISTRY.planner(CONV_OP, plan_conv)
