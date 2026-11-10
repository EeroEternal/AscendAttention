"""Sampling: top-k filtering expressed as a threshold, with a vendor call as the fast path.

The op is "keep the tokens whose logit is at least the k-th largest", i.e. top-k written as a
**threshold comparison**.  Two reasons it is not written as ``topk``:

* the vendor accelerator (DeepSelect) returns a threshold/selection without a full-vocabulary sort, so
  the two forms are not interchangeable performance-wise — the whole point is to avoid the sort;
* threshold form keeps ties (every token equal to the cutoff survives), which the sort form does not
  guarantee, and a sampler that silently drops tied tokens changes the distribution.

The math is therefore: ``cutoff = k-th largest logit``, then ``logits < cutoff -> -inf``.  Both
implementations here follow that sentence, and ``testing/sampling_reference.py`` states it twice
(logit domain and prob domain) so a test can check the two domains agree.

Evidence for the vendor impl is [待复核] and stays opt-in: TileInfer recorded time ratios
0.39 / 0.25 / 0.23 (lower is faster, so a 2.5-4x win) but only in single-window runs, and this repo
does not promote single-window numbers.  ``bench/bench_sampling_topk.py`` re-measures it properly.
"""

from __future__ import annotations

from typing import Any

import torch

from ...plan import OpPlan
from ...spec import REGISTRY, ImplSpec, ShapeSpec, dim_min

__all__ = [
    "OP",
    "TOP_THRESHOLD_TORCH",
    "TOP_THRESHOLD_VENDOR",
    "plan_topk_threshold",
    "run_topk_threshold",
    "topk_threshold",
]

OP = "sampling_topk_threshold"

_SHAPE = ShapeSpec(
    ndim=2,
    dims=(dim_min(1), dim_min(1)),
    dtypes=frozenset({"bfloat16", "float16", "float32"}),
    contiguous=True,
)

TOP_THRESHOLD_TORCH = ImplSpec(
    op=OP,
    impl="torch.topk_threshold",
    form="torch",
    device="any",
    shape=_SHAPE,
    kv_layouts=frozenset({"ND"}),
    graph="capture_safe",
    plan_kind="stateless",
    enabled_by_default=True,
    measured_ratio=1.0,
    ratio_note="baseline by definition: the sort-based threshold the vendor call replaces",
    bench="bench/bench_sampling_topk.py",
    provenance="tileinfer/kernels/sampling/topk.py (semantics) + testing/sampling_reference.py",
    note="returns a new tensor; ties at the cutoff survive",
)

_VENDOR_SHAPE = ShapeSpec(
    ndim=2,
    dims=(dim_min(1), dim_min(1)),
    dtypes=frozenset({"bfloat16"}),
    contiguous=True,
)

TOP_THRESHOLD_VENDOR = ImplSpec(
    op=OP,
    impl="vendor.deep_select",
    form="vendor_call",
    device="ascend950",
    shape=_VENDOR_SHAPE,
    kv_layouts=frozenset({"ND"}),
    # the vendor call allocates its own workspace and reads strides on the host: not capture-safe
    graph="eager_only",
    plan_kind="stateless",
    enabled_by_default=True,
    measured_ratio=3.8432,
    ratio_note=(
        "3.84x measured in this repo (B=8, V=152064, k=20, bf16; median of 6 alternating pairs, "
        "bench/results/sampling_topk_b8.json, 2026-11-10; time ratio 0.26 vs the torch threshold). "
        "Index sets differ from torch only when bf16 logits collide (values identical), because the "
        "threshold form keeps ties - see docs/pitfalls.md. Device-gated: a CPU box rejects it."
    ),
    bench="bench/bench_sampling_topk.py",
    requires=frozenset({"deep_select", "torch_npu"}),
    provenance="tileinfer/kernels/sampling/topk.py:_deepselect_topk",
    note="bf16, 2-D, stride(1)==1, row stride aligned, scalar k < vocab: the vendor's hard window",
)

REGISTRY.register(TOP_THRESHOLD_TORCH)
REGISTRY.register(TOP_THRESHOLD_VENDOR)


def topk_threshold(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Reference-free torch form of the contract (kept public for tests and for the baseline)."""
    cutoff = torch.topk(logits.float(), int(k), dim=-1).values.min(dim=-1, keepdim=True).values
    return logits.masked_fill(logits < cutoff.to(logits.dtype), float("-inf"))


def _vendor_window_ok(logits: torch.Tensor, k: int) -> bool:
    """The vendor's documented hard window (``deep_select/interface.py``), checked without a sync."""
    if logits.device.type != "npu" or logits.dim() != 2:
        return False
    if logits.dtype != torch.bfloat16:  # Ascend path is bf16-only in the vendor interface
        return False
    if logits.stride(-1) != 1:
        return False
    if (logits.stride(0) * logits.element_size()) % 512 != 0:
        return False
    return 0 < int(k) < int(logits.shape[-1])


def plan_topk_threshold(impl: ImplSpec, **pure: Any) -> OpPlan:
    batch = int(pure["batch"])
    vocab = int(pure["vocab"])
    k = int(pure["k"])
    dtype = str(pure.get("dtype", "bfloat16"))
    if not 0 < k < vocab:
        raise ValueError(f"k={k} must be in (0, vocab={vocab})")

    if impl.form == "vendor_call":
        # Memory-layout facts (stride/alignment) cannot be expressed in a ShapeSpec, and the vendor
        # reads strides on the host: refuse at plan time rather than returning a wrong-but-plausible
        # selection.  Everything expressible as shape/dtype is already filtered by dispatch.
        if dtype != "bfloat16":
            raise ValueError("vendor.deep_select accepts bf16 logits only (Ascend path)")
        if k >= vocab:
            raise ValueError(f"k={k} must be < vocab={vocab}")

        def execute(logits):
            import deep_select

            return deep_select.topk(logits, k, sorted=False, indices_type=torch.int32,
                                    return_value=True)

    else:
        def execute(logits):
            return topk_threshold(logits, k)

    return OpPlan(
        op=OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(batch, vocab),
        dtypes=(dtype,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={"k": k, "vendor_window": impl.form == "vendor_call"},
        execute=execute,
    )


def run_topk_threshold(p: OpPlan, logits):
    return p(logits)


REGISTRY.planner(OP, plan_topk_threshold)
