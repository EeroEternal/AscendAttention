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
from .prenz import pack_nd_to_compact_nz, unpack_compact_nz_to_nd
from .slot_layout import em_upper_bound, slot_layout_from_group_list

__all__ = [
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
