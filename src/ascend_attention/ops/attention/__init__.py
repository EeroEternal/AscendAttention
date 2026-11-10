"""``paged_decode_attention`` — the first compiled-kernel operator.

Two implementations, one op name:

* ``torch.reference`` — the portable oracle.  Registered so ``supports()`` can always answer "yes,
  portably", and so the numerics of everything else can be checked against it.  Never advertised,
  never a default.
* ``tilelang.decode950`` — the Ascend 950 kernel migrated from TileInfer in P1.  It is **not**
  enabled by default and carries no ``measured_ratio``: an op-level figure against the vendor path
  has not been recorded on this repo's own harness yet, and an unmeasured implementation must not
  become a default (``spec.ImplSpec`` refuses that combination at registration time).

The kernel module is imported lazily inside :func:`plan_decode`, so importing this package stays
toolchain-free.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from ...plan import OpPlan
from ...spec import REGISTRY, ImplSpec, ShapeSpec, dim_in, dim_min

__all__ = ["OP", "REFERENCE", "TILELANG_DECODE950", "plan_decode", "run_decode"]

OP = "paged_decode_attention"

#: The portable oracle (also the numerics reference for the kernel tests).
REFERENCE = ImplSpec(
    op=OP,
    impl="torch.reference",
    form="torch",
    device="any",
    shape=ShapeSpec(
        ndim=4,
        dims=(dim_min(1), dim_min(1), 1, dim_in(8, 16, 32, 64, 128, 256)),
        dtypes=frozenset({"bfloat16", "float16", "float32"}),
        contiguous=True,
    ),
    kv_layouts=frozenset({"ND"}),
    graph="eager_only",
    plan_kind="attention",
    enabled_by_default=False,
    measured_ratio=None,
    ratio_note="oracle, not a candidate: no measurement is possible or desired",
    bench=None,
    provenance="tileinfer/attention/backends/reference.py + tileinfer/testing/reference.py",
    note="portable oracle; accepts any head_dim the reference attention can compute",
)

#: The Ascend 950 decode kernel as it runs in TileInfer today (decode only in this phase).
TILELANG_DECODE950 = ImplSpec(
    op=OP,
    impl="tilelang.decode950",
    form="tilelang",
    device="ascend950",
    shape=ShapeSpec(
        ndim=4,
        dims=(dim_min(1), dim_min(1), 1, dim_in(128, 256)),
        dtypes=frozenset({"bfloat16", "float16"}),
        contiguous=True,
    ),
    kv_layouts=frozenset({"ND"}),
    # Audited 2026-11-10 on Ascend950PR_9579 with two independent checks: (1) capture_guard saw zero
    # host reads and one allocation over a run, (2) a real torch.npu.NPUGraph capture + replay returned
    # bitwise identical output (max_abs_diff 0.0).  The per-call metadata refresh allocates (allowed
    # under capture, the shapes are static) but never syncs, which is exactly the tier definition.
    graph="capture_safe",
    plan_kind="attention",
    enabled_by_default=False,
    measured_ratio=None,
    ratio_note=(
        "no op-level A/B against the vendor path has been recorded in this repo yet; "
        "TileInfer's end-to-end numbers (2.53x TPOT / 11.8x TTFT) belong to the whole library, "
        "not to this op"
    ),
    bench="bench/bench_paged_decode.py",
    requires=frozenset({"tilelang", "torch_npu"}),
    provenance="tileinfer/kernels/attention/paged_decode_ascend950.py",
    note=(
        "decode only; head_dim in {128, 256}; NHD cache; pool-agnostic ABI (declared capacity).  "
        "capture_safe verified by capture_guard (0 syncs) and by NPUGraph capture+replay (0.0 diff)"
    ),
)

REGISTRY.register(REFERENCE)
REGISTRY.register(TILELANG_DECODE950)


def _backend_for(impl: ImplSpec) -> str:
    return "reference" if impl.form == "torch" else "tilelang-ascend950"


def plan_decode(impl: ImplSpec, **pure: Any) -> OpPlan:
    """Plan one decode step.  Pure data in: shapes, dtypes, page geometry, the KV handle.

    ``k_cache``/``v_cache`` are state, not inputs — they are captured by the plan's executor, which
    is why ``run(plan, q)`` takes the query alone.
    """
    k_cache = pure.get("k_cache")
    v_cache = pure.get("v_cache")
    if k_cache is None or v_cache is None:
        raise ValueError("paged_decode_attention.plan needs k_cache and v_cache")
    num_qo_heads = int(pure["num_qo_heads"])
    num_kv_heads = int(pure["num_kv_heads"])
    head_dim = int(pure["head_dim"])
    kv_indptr = pure.get("kv_indptr")
    if pure.get("batch") is not None:
        batch = int(pure["batch"])
    elif kv_indptr is not None:
        batch = int(kv_indptr.numel()) - 1
    else:
        batch = 1

    dtype_name = str(pure.get("dtype") or k_cache.dtype).rsplit(".", 1)[-1]
    if impl.shape is not None and not impl.shape.matches(
        (max(batch, 1), num_qo_heads, 1, head_dim), dtype_name
    ):
        raise ValueError(
            f"paged_decode_attention/{impl.impl}: q=({batch}, {num_qo_heads}, 1, {head_dim}) "
            f"dtype={dtype_name} is outside the declared contract"
        )
    if impl.form == "tilelang":
        from ...utils import have_tilelang, is_npu_available

        if not (have_tilelang() and is_npu_available()):
            raise RuntimeError(
                "tilelang.decode950 needs the TileLang Ascend toolchain; "
                "source scripts/env-ascend102.sh or use impl='torch.reference'"
            )

    from ...api import BatchAttention

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name)
    if torch_dtype is None:
        raise ValueError(f"paged_decode_attention: unsupported dtype {dtype_name!r}")
    attn = BatchAttention(backend=_backend_for(impl), device=pure.get("device"), dtype=torch_dtype)
    attention_plan = attn.plan(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_indptr=kv_indptr,
        kv_indices=pure.get("kv_indices"),
        kv_last_page_len=pure.get("kv_last_page_len"),
        qo_indptr=pure.get("qo_indptr"),
        page_table=pure.get("page_table"),
        mode="decode",
        page_size=pure.get("page_size"),
        sm_scale=pure.get("sm_scale"),
        kv_layout=str(pure.get("kv_layout", "NHD")),
        causal=bool(pure.get("causal", True)),
        block_q=int(pure.get("block_q", 128)),
        kv_tile_pages=int(pure.get("kv_tile_pages", 0)),
        tokens_per_split=int(pure.get("tokens_per_split", 0)),
        load_balance=bool(pure.get("load_balance", True)),
        k_cache=k_cache,
        v_cache=v_cache,
        finalize=bool(pure.get("finalize", True)),
    )

    def execute(q: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return attn.run(q, k_cache, v_cache, plan=attention_plan, out=out)

    return OpPlan(
        op=OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(max(batch, 1), num_qo_heads, 1, head_dim),
        dtypes=(dtype_name,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": int(attention_plan.page_size),
            "kv_layout": str(pure.get("kv_layout", "NHD")),
            "mode": "decode",
        },
        state=attention_plan,
        handle=attention_plan.backend_state.get("kernel"),
        execute=execute,
    )


def run_decode(p: OpPlan, q: torch.Tensor, **kw: Any) -> torch.Tensor:
    return p(q, **kw)


REGISTRY.planner(OP, plan_decode)


# ---------------------------------------------------------------------------------------------
# Prefill: the reference implementation is the whole story in this phase, and that is a decision.
# ---------------------------------------------------------------------------------------------
PREFILL = "paged_prefill_attention"

PREFILL_REFERENCE = ImplSpec(
    op=PREFILL,
    impl="torch.reference",
    form="torch",
    device="any",
    shape=ShapeSpec(
        ndim=4,
        dims=(dim_min(1), dim_min(1), dim_min(1), dim_in(8, 16, 32, 64, 128, 256)),
        dtypes=frozenset({"bfloat16", "float16", "float32"}),
        contiguous=False,
    ),
    kv_layouts=frozenset({"ND"}),
    # The reference loops over requests and chunks in Python: correct, but a captured graph would
    # freeze a tile count that a real prefill step does not have.
    graph="eager_only",
    plan_kind="attention",
    enabled_by_default=True,
    measured_ratio=1.0,
    bench="bench/bench_prefill_reference.py",
    ratio_note=(
        "definitional 1.0: the only prefill implementation in this phase.  The TileLang prefill kernel "
        "is deliberately not migrated yet - TileInfer's prefill is a tile-count-driven path that is "
        "*not* plan-backed (backends/tilelang_ascend950.py:89-95), and pretending otherwise would put a "
        "compile key in the manifest that no run uses."
    ),
    provenance="tileinfer/attention/backends/reference.py + tileinfer/testing/reference.py",
    note=(
        "prefill/append over a paged cache; causal by default; qo_len per request via qo_indptr or a "
        "uniform qo_len.  Slower than a kernel by construction: it exists so the interface has a "
        "second mode, not to be fast."
    ),
)

#: The prefill *kernel* is in this repo but not wired: its orchestration (tile counts, split/merge,
#: frontier bookkeeping) is tile-count-driven rather than plan-backed, which is the one thing this
#: library refuses to absorb.  Declared so the inventory tells the truth instead of implying the kernel
#: does not exist.
PREFILL_KERNEL_UNWIRED = ImplSpec(
    op=PREFILL,
    impl="tilelang.prefill950",
    form="tilelang",
    device="ascend950",
    shape=ShapeSpec(
        ndim=4,
        dims=(dim_min(1), dim_min(1), dim_min(1), dim_in(128, 256)),
        dtypes=frozenset({"bfloat16", "float16"}),
        contiguous=True,
    ),
    kv_layouts=frozenset({"ND"}),
    graph="eager_only",
    plan_kind="attention",
    enabled_by_default=False,
    measured_ratio=None,
    bench=None,
    ratio_note="not wired, so not measured: there is no path to benchmark yet",
    requires=frozenset({"tilelang", "torch_npu"}),
    provenance=(
        "tileinfer/kernels/attention/paged_decode_ascend950.py:paged_prefill_ascend950 (+split/head/"
        "frontier variants) and tileinfer/attention/prefill.py (orchestration, not migrated)"
    ),
    note=(
        "kernel present in-tree, orchestration deliberately not migrated: a prefill step's tile count "
        "depends on the batch, which a per-shape plan cannot express (backends/tilelang_ascend950.py:89-95)"
    ),
)

REGISTRY.register(PREFILL_REFERENCE)
REGISTRY.register(PREFILL_KERNEL_UNWIRED)


def plan_prefill(impl: ImplSpec, **pure: Any) -> OpPlan:
    """Plan one prefill/append step: ``q`` is ``[batch, num_qo_heads, qo_len, head_dim]``."""
    k_cache = pure.get("k_cache")
    v_cache = pure.get("v_cache")
    if k_cache is None or v_cache is None:
        raise ValueError("paged_prefill_attention.plan needs k_cache and v_cache")
    num_qo_heads = int(pure["num_qo_heads"])
    num_kv_heads = int(pure["num_kv_heads"])
    head_dim = int(pure["head_dim"])
    qo_indptr = pure.get("qo_indptr")
    if qo_indptr is not None:
        batch = int(qo_indptr.numel()) - 1
        qo_len = int(qo_indptr[1:].sub(qo_indptr[:-1]).max())
    elif pure.get("qo_len") is not None:
        batch = int(pure.get("batch", 1))
        qo_len = int(pure["qo_len"])
    else:
        raise ValueError("paged_prefill_attention.plan needs qo_indptr or (batch, qo_len)")
    if qo_len < 1:
        raise ValueError(f"paged_prefill_attention: qo_len must be >= 1, got {qo_len}")

    dtype_name = str(pure.get("dtype") or k_cache.dtype).rsplit(".", 1)[-1]
    if impl.shape is not None and not impl.shape.matches(
        (max(batch, 1), num_qo_heads, qo_len, head_dim), dtype_name
    ):
        raise ValueError(
            f"paged_prefill_attention/{impl.impl}: q=({batch}, {num_qo_heads}, {qo_len}, {head_dim}) "
            f"dtype={dtype_name} is outside the declared contract"
        )

    if impl.form == "tilelang":
        raise NotImplementedError(
            f"paged_prefill_attention/{impl.impl}: the prefill kernel is in-tree but its orchestration is "
            "tile-count-driven and not plan-backed (see ImplSpec.note); use impl='torch.reference' or "
            "drive the kernel from the platform's prefill path"
        )

    from ...api import BatchAttention

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name)
    if torch_dtype is None:
        raise ValueError(f"paged_prefill_attention: unsupported dtype {dtype_name!r}")
    attn = BatchAttention(backend=_backend_for(impl), device=pure.get("device"), dtype=torch_dtype)
    attention_plan = attn.plan(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_indptr=pure.get("kv_indptr"),
        kv_indices=pure.get("kv_indices"),
        kv_last_page_len=pure.get("kv_last_page_len"),
        qo_indptr=qo_indptr,
        page_table=pure.get("page_table"),
        mode="prefill",
        page_size=pure.get("page_size"),
        sm_scale=pure.get("sm_scale"),
        kv_layout=str(pure.get("kv_layout", "NHD")),
        causal=bool(pure.get("causal", True)),
        block_q=int(pure.get("block_q", 128)),
        k_cache=k_cache,
        v_cache=v_cache,
    )

    def execute(q: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return attn.run(q, k_cache, v_cache, plan=attention_plan, out=out)

    return OpPlan(
        op=PREFILL,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(max(batch, 1), num_qo_heads, qo_len, head_dim),
        dtypes=(dtype_name,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "qo_len": qo_len,
            "mode": "prefill",
            "causal": bool(pure.get("causal", True)),
        },
        state=attention_plan,
        execute=execute,
    )


def run_prefill(p: OpPlan, q: torch.Tensor, **kw: Any) -> torch.Tensor:
    return p(q, **kw)


REGISTRY.op(PREFILL, doc="Prefill/append attention over a paged KV cache (causal by default).")
REGISTRY.planner(PREFILL, plan_prefill)
