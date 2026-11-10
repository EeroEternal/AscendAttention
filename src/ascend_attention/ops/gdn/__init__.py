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
from ...spec import REGISTRY, ImplSpec, ShapeSpec, dim_min
from .gating import fused_gdn_gating, reference_gdn_gating

__all__ = ["OP", "BASELINE", "plan_gating", "run_gating"]

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
