"""MoE layout ops: the two pure-data pieces the grouped-GEMM kernels need around them.

Scope, decided before porting rather than after (the plan asks for a verdict, and the verdict is
mostly "no"):

* ``moe_slot_layout`` — compact (expert-sorted) rows -> per-expert ``block_m`` padded slots.  Pure
  torch (searchsorted/cumsum/scatter), static shapes, no host sync.  **Migrated**, because every MoE
  kernel needs it and a torch version is what the engine falls back to.
* ``moe_prenz_pack`` — host-side ND -> compact-NZ weight packing, so a kernel can load weights with a
  raw DMA instead of a GM->L1 nd2nz conversion.  Pure torch, round-trip testable.  **Migrated**; the
  physical order is derived from the compiler's layout, so it is only proven for *bookkeeping* until a
  device test exercises it — the docstring says so and the test says so.
* ``grouped_gemm`` (0.86x, a measured loss), ``down_prenz`` (wrong results, ~1e36), ``align_gate_up`` /
  ``align_down`` (below the 1.0x gate) and ``shared_expert_fold`` (needs an A/B before it may claim
  anything) are **archived, not migrated**.  See ``docs/archive-moe.md`` for the numbers and why.

Neither migrated op is a speed claim: both are baselines with no alternative implementation, so their
``measured_ratio`` is the definitional 1.0 and the bench script records absolute latency only.
"""

from __future__ import annotations

from typing import Any

from ...plan import OpPlan
from ...spec import REGISTRY, ImplSpec, ShapeSpec, dim_in, dim_min, dim_mul
from .compute import (
    fused_experts,
    moe_align_block_size,
    silu_and_mul,
    topk_softmax,
    unquantized_fused_moe,
)
from .prenz import pack_nd_to_compact_nz, unpack_compact_nz_to_nd
from .slot_layout import em_upper_bound, slot_layout_from_group_list

__all__ = [
    "ALIGN_OP",
    "EXPERTS_OP",
    "SILU_OP",
    "TOPK_OP",
    "UNQUANT_OP",
    "fused_experts",
    "moe_align_block_size",
    "plan_align_block_size",
    "plan_experts",
    "plan_silu_and_mul",
    "plan_topk_softmax",
    "run_align_block_size",
    "run_experts",
    "run_silu_and_mul",
    "run_topk_softmax",
    "silu_and_mul",
    "topk_softmax",
    "unquantized_fused_moe",
    "EM_OP",
    "PRENZ_OP",
    "EM_IMPL",
    "PRENZ_IMPL",
    "plan_slot_layout",
    "run_slot_layout",
    "plan_prenz_pack",
    "run_prenz_pack",
    "em_upper_bound",
    "slot_layout_from_group_list",
    "pack_nd_to_compact_nz",
    "unpack_compact_nz_to_nd",
]

EM_OP = "moe_slot_layout"
PRENZ_OP = "moe_prenz_pack"

EM_IMPL = ImplSpec(
    op=EM_OP,
    impl="torch.slot_layout",
    form="torch",
    device="any",
    shape=ShapeSpec(
        ndim=1,
        dims=(dim_min(1),),
        dtypes=frozenset({"int32", "int64"}),
        contiguous=True,
    ),
    kv_layouts=frozenset({"ND"}),
    # index arithmetic only: no allocation of its own beyond the output buffers, no host reads
    graph="capture_safe",
    plan_kind="stateless",
    enabled_by_default=True,
    measured_ratio=1.0,
    ratio_note=(
        "definitional baseline: the only implementation, so there is nothing to beat. Bench records "
        "absolute latency (bench/bench_moe_layout.py), not a ratio"
    ),
    bench="bench/bench_moe_layout.py",
    provenance="tileinfer/kernels/moe/slot_layout.py",
    note="group_list (expert cumsum) -> slot_pos/sorted_ids/expert_ids, static shapes, no host sync",
)

PRENZ_IMPL = ImplSpec(
    op=PRENZ_OP,
    impl="torch.prenz_pack",
    form="torch",
    device="any",
    shape=ShapeSpec(
        ndim=2,
        dims=(dim_min(1), dim_mul(16)),
        dtypes=frozenset({"bfloat16", "float16"}),
        contiguous=True,
    ),
    kv_layouts=frozenset({"ND"}),
    graph="capture_safe",
    plan_kind="stateless",
    enabled_by_default=True,
    measured_ratio=1.0,
    ratio_note=(
        "definitional baseline: a host-side one-off weight pack. Bench records absolute latency; the "
        "only claim is the round-trip identity and the physical order formula"
    ),
    bench="bench/bench_moe_layout.py",
    provenance="tileinfer/kernels/moe/prenz.py",
    note="ND -> compact-NZ (C0 = 256/bits); colours are derived from the compiler layout, not measured",
)

REGISTRY.register(EM_IMPL)
REGISTRY.register(PRENZ_IMPL)


def plan_slot_layout(impl: ImplSpec, **pure: Any) -> OpPlan:
    num_rows = int(pure["num_rows"])
    rows_static = int(pure["rows_static"])
    em_static = int(pure["em_static"])
    num_experts = int(pure["num_experts"])
    block_m = int(pure["block_m"])
    if rows_static < num_rows:
        raise ValueError(f"rows_static={rows_static} must be >= num_rows={num_rows}")
    # The slot buffer is em_static x block_m rows, not num_rows rows: tiles are padded, so a
    # compact-row count would truncate the tail tiles (and index_copy_ would raise in the kernel-less
    # torch path, which is how this was found).
    if rows_static != em_static * block_m:
        raise ValueError(
            f"rows_static={rows_static} must equal em_static*block_m={em_static * block_m}"
        )
    if em_static < em_upper_bound(num_rows, num_experts, block_m):
        raise ValueError(
            f"em_static={em_static} is below the static upper bound "
            f"{em_upper_bound(num_rows, num_experts, block_m)}; a short buffer would drop tiles"
        )

    def execute(group_list):
        return slot_layout_from_group_list(
            group_list,
            num_rows=num_rows,
            rows_static=rows_static,
            em_static=em_static,
            num_experts=num_experts,
            block_m=block_m,
        )

    return OpPlan(
        op=EM_OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(num_rows,),
        dtypes=("int32",),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={
            "rows_static": rows_static,
            "em_static": em_static,
            "num_experts": num_experts,
            "block_m": block_m,
        },
        execute=execute,
    )


def run_slot_layout(p: OpPlan, group_list):
    return p(group_list)


def plan_prenz_pack(impl: ImplSpec, **pure: Any) -> OpPlan:
    rows = int(pure["rows"])
    cols = int(pure["cols"])
    dtype = str(pure.get("dtype", "bfloat16"))
    bits = 16 if dtype in ("bfloat16", "float16") else 32
    c0 = 256 // bits
    if cols % c0:
        raise ValueError(f"cols={cols} must be divisible by C0={c0} for {dtype}")
    # ``unpack`` is a plan-time flag, not a run-time argument: the direction is pure data, and letting
    # it vary per call would make the plan lie about what it does.
    unpack = bool(pure.get("unpack", False))

    def execute(tile):
        return unpack_compact_nz_to_nd(tile) if unpack else pack_nd_to_compact_nz(tile)

    return OpPlan(
        op=PRENZ_OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(rows, cols),
        dtypes=(dtype,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={"rows": rows, "cols": cols, "c0": c0, "unpack": unpack},
        execute=execute,
    )


def run_prenz_pack(p: OpPlan, tile):
    return p(tile)


REGISTRY.planner(EM_OP, plan_slot_layout)
REGISTRY.planner(PRENZ_OP, plan_prenz_pack)

# =============================================================================================
# MoE compute: routing, alignment, and the unquantised expert MLP
#
# These are the ops the boundary review found owned by nobody (see ``docs/boundary.md``): routing and
# expert compute are arithmetic with shape/dtype constraints, so they belong here and the platform
# only binds engine symbols onto them.  All five are pure torch and sync-free, so they are the
# fallback the engine gets when no kernel is available - and they are what makes the platform's
# "13 required ops" resolvable instead of 8.
#
# ``measured_ratio`` is the definitional 1.0: there is no alternative implementation to measure
# against yet, so claiming a speed-up would be a fabrication.  The bench records absolute latency.
# =============================================================================================

SILU_OP = "moe_silu_and_mul"
TOPK_OP = "moe_topk_softmax"
ALIGN_OP = "moe_align_block_size"
EXPERTS_OP = "moe_fused_experts"
UNQUANT_OP = "moe_unquantized_fused_moe"

_MOE_DTYPES = frozenset({"bfloat16", "float16", "float32"})
_MOE_BENCH = "bench/bench_moe_compute.py"


def _moe_impl(op: str, impl: str, *, note: str, graph: str = "capture_safe") -> ImplSpec:
    return ImplSpec(
        op=op,
        impl=impl,
        form="torch",
        device="any",
        shape=ShapeSpec(ndim=2, dims=(dim_min(1), dim_min(1)), dtypes=_MOE_DTYPES, contiguous=True),
        kv_layouts=frozenset({"ND"}),
        graph=graph,
        plan_kind="stateless",
        enabled_by_default=True,
        measured_ratio=1.0,
        bench=_MOE_BENCH,
        ratio_note=(
            "definitional 1.0: this is the only implementation of the op, so there is nothing to "
            "compare against; the ledger records absolute latency only"
        ),
        provenance="tileinfer/vllm/ops/moe.py (fused_experts / topk_softmax) and kernels/moe/align.py",
        note=note,
    )


SILU_IMPL = _moe_impl(
    SILU_OP,
    "torch.silu_and_mul",
    note="SwiGLU gate applied to a fused gate/up projection; last dim halved, f32 compute for half inputs",
)
TOPK_IMPL = _moe_impl(
    TOPK_OP,
    "torch.topk_softmax",
    note=(
        "softmax over experts then top-k; renormalize is a plan-time choice (off changes the sums); "
        "ids are int32, which is what the alignment op and the grouped GEMMs index with"
    ),
)
ALIGN_IMPL = _moe_impl(
    ALIGN_OP,
    "torch.align_block_size",
    note=(
        "expert-major, block-padded pair layout for a grouped GEMM; buffers are the engine's fixed "
        "sizes and the tail is sentinel-filled, so no allocation happens per call"
    ),
)
EXPERTS_IMPL = _moe_impl(
    EXPERTS_OP,
    "torch.fused_experts",
    note=(
        "unquantised routed MLP (w2 @ silu(w1 @ x), weighted sum over top-k); quant_config is refused "
        "rather than ignored; expert_map implements expert parallelism"
    ),
)
UNQUANT_IMPL = _moe_impl(
    UNQUANT_OP,
    "torch.unquantized_fused_moe",
    note="the 'method' entry point: same math as moe_fused_experts, argument names fixed for serving",
)

for _impl in (SILU_IMPL, TOPK_IMPL, ALIGN_IMPL, EXPERTS_IMPL, UNQUANT_IMPL):
    REGISTRY.register(_impl)


def plan_silu_and_mul(impl: ImplSpec, **pure: Any) -> OpPlan:
    width = int(pure["width"])
    if width % 2:
        raise ValueError(f"silu_and_mul needs an even fused width, got {width}")
    dtype = str(pure.get("dtype", "bfloat16"))

    def execute(x):
        return silu_and_mul(x)

    return OpPlan(
        op=SILU_OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(width,),
        dtypes=(dtype,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={"width": width, "out_width": width // 2},
        execute=execute,
    )


def plan_topk_softmax(impl: ImplSpec, **pure: Any) -> OpPlan:
    num_experts = int(pure["num_experts"])
    top_k = int(pure["top_k"])
    renormalize = bool(pure.get("renormalize", True))
    if not 0 < top_k <= num_experts:
        raise ValueError(f"top_k={top_k} must be in 1..{num_experts}")
    dtype = str(pure.get("dtype", "bfloat16"))

    def execute(scores):
        return topk_softmax(scores, top_k, renormalize=renormalize)

    return OpPlan(
        op=TOPK_OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(num_experts,),
        dtypes=(dtype,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={"num_experts": num_experts, "top_k": top_k, "renormalize": renormalize},
        execute=execute,
    )


def plan_align_block_size(impl: ImplSpec, **pure: Any) -> OpPlan:
    num_tokens = int(pure["num_tokens"])
    top_k = int(pure["top_k"])
    num_experts = int(pure["num_experts"])
    block_size = int(pure["block_size"])
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    def execute(topk_ids):
        return moe_align_block_size(
            topk_ids, num_experts, block_size, num_tokens=num_tokens
        )

    return OpPlan(
        op=ALIGN_OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(num_tokens, top_k),
        dtypes=("int32",),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={
            "num_tokens": num_tokens,
            "top_k": top_k,
            "num_experts": num_experts,
            "block_size": block_size,
        },
        execute=execute,
    )


def plan_experts(impl: ImplSpec, **pure: Any) -> OpPlan:
    num_tokens = int(pure["num_tokens"])
    hidden = int(pure["hidden"])
    inter = int(pure["inter"])
    num_experts = int(pure["num_experts"])
    top_k = int(pure["top_k"])
    dtype = str(pure.get("dtype", "bfloat16"))
    activation = str(pure.get("activation", "silu"))
    on_input = bool(pure.get("apply_router_weight_on_input", False))
    return_experts = bool(pure.get("expert_map", False))

    if activation not in ("silu", "swish"):
        raise NotImplementedError(f"activation {activation!r} is not implemented (silu/swish only)")

    def execute(hidden_states, w1, w2, topk_weights, topk_ids, *rest):
        expert_map = rest[0] if rest else None
        return fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            on_input,
            expert_map=expert_map,
        )

    return OpPlan(
        op=EXPERTS_OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(num_tokens, hidden),
        dtypes=(dtype,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={
            "num_tokens": num_tokens,
            "hidden": hidden,
            "inter": inter,
            "num_experts": num_experts,
            "top_k": top_k,
            "activation": activation,
            "apply_router_weight_on_input": on_input,
            "expert_map": return_experts,
        },
        execute=execute,
    )


def plan_unquantized_fused_moe(impl: ImplSpec, **pure: Any) -> OpPlan:
    plan = plan_experts(impl, **pure)
    num_experts = int(pure["num_experts"])

    def execute(hidden_states, w1, w2, topk_weights, topk_ids, *rest):
        expert_map = rest[0] if rest else None
        return unquantized_fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            num_experts=num_experts,
            activation=str(pure.get("activation", "silu")),
            apply_router_weight_on_input=bool(pure.get("apply_router_weight_on_input", False)),
            expert_map=expert_map,
        )

    return OpPlan(
        op=UNQUANT_OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=plan.shape,
        dtypes=plan.dtypes,
        kv_layouts=plan.kv_layouts,
        graph=impl.graph,
        meta={**plan.meta, "op": UNQUANT_OP},
        execute=execute,
    )


def run_silu_and_mul(p: OpPlan, x):
    return p(x)


def run_topk_softmax(p: OpPlan, scores):
    return p(scores)


def run_align_block_size(p: OpPlan, topk_ids):
    return p(topk_ids)


def run_experts(p: OpPlan, hidden_states, w1, w2, topk_weights, topk_ids, expert_map=None):
    return p(hidden_states, w1, w2, topk_weights, topk_ids, expert_map)


REGISTRY.planner(SILU_OP, plan_silu_and_mul)
REGISTRY.planner(TOPK_OP, plan_topk_softmax)
REGISTRY.planner(ALIGN_OP, plan_align_block_size)
REGISTRY.planner(EXPERTS_OP, plan_experts)
REGISTRY.planner(UNQUANT_OP, plan_unquantized_fused_moe)
