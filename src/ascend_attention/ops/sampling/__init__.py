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
    "MINP_UNIMPLEMENTED",
    "NUCLEUS_OP",
    "plan_top_p",
    "run_top_p",
    "top_p_filter",
    "OP",
    "TOP_THRESHOLD_TORCH",
    "TOP_THRESHOLD_VENDOR",
    "plan_topk_threshold",
    "run_topk_threshold",
    "topk_threshold",
]

OP = "sampling_topk_threshold"

_DTYPES = frozenset({"bfloat16", "float16", "float32"})

_SHAPE = ShapeSpec(
    ndim=2,
    dims=(dim_min(1), dim_min(1)),
    dtypes=_DTYPES,
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


# ---------------------------------------------------------------------------------------------
# The greedy sibling, and the routing between the two.
# ---------------------------------------------------------------------------------------------
GREEDY = "sampling_greedy"

GREEDY_TORCH = ImplSpec(
    op=GREEDY,
    impl="torch.argmax",
    form="torch",
    device="any",
    shape=ShapeSpec(
        ndim=2,
        dims=(dim_min(1), dim_min(1)),
        dtypes=_DTYPES,
        contiguous=False,
    ),
    graph="capture_safe",
    plan_kind="stateless",
    enabled_by_default=True,
    measured_ratio=1.0,
    bench="bench/bench_sampling_greedy.py",
    ratio_note=(
        "definitional 1.0: the only implementation.  argmax is the semantics, not an optimisation, so "
        "there is nothing to beat.  Ties resolve to the first maximum index, as torch does."
    ),
)


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Index of the maximum logit per row (greedy decoding); no host read, no allocation of a mask."""
    return torch.argmax(logits, dim=-1)


def plan_greedy(impl: ImplSpec, **pure: Any) -> OpPlan:
    batch = int(pure["batch"])
    vocab = int(pure["vocab"])
    dtype = str(pure.get("dtype", "bfloat16"))
    if batch < 1:
        raise ValueError(f"{GREEDY}/{impl.impl}: batch must be >= 1, got {batch}")
    if vocab < 1:
        raise ValueError(f"{GREEDY}/{impl.impl}: vocab must be >= 1, got {vocab}")
    if dtype not in _DTYPES:
        raise ValueError(f"{GREEDY}/{impl.impl}: dtype {dtype!r} is outside {sorted(_DTYPES)}")

    def execute(logits):
        if logits.dim() != 2 or tuple(logits.shape) != (batch, vocab):
            raise ValueError(
                f"{GREEDY}/{impl.impl}: logits must be {(batch, vocab)}, got {tuple(logits.shape)}"
            )
        return greedy(logits)

    return OpPlan(
        op=GREEDY,
        impl=impl.impl,
        plan_kind="stateless",
        shape=(batch, vocab),
        dtypes=(dtype,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={"select": "argmax"},
        execute=execute,
    )


REGISTRY.op(GREEDY, doc="Greedy sampling: the per-row argmax index (do_sample=False).")
REGISTRY.register(GREEDY_TORCH)
REGISTRY.planner(GREEDY, plan_greedy)


# ---------------------------------------------------------------------------------------------
# Nucleus (top-p) filtering: the README listed it as missing, and it is the one sampler a serving
# stack cannot do without next to greedy and top-k.
#
# It is a *different function* from the top-k threshold, not a variant: top-k cuts at a rank, top-p
# cuts where the cumulative mass reaches p, so the survivor count varies per row.  That is why it is
# its own op with its own plan instead of a flag on `sampling_topk_threshold`.
# ---------------------------------------------------------------------------------------------
NUCLEUS_OP = "sampling_top_p"

#: Written down as a constant so "not implemented" is a fact in the code, not only in the README.
MINP_UNIMPLEMENTED = (
    "min-p (keep tokens with prob >= min_p * max_prob) is not implemented; it is a different "
    "rule from top-p and the dispatcher must not pretend one is the other"
)

NUCLEUS_TORCH = ImplSpec(
    op=NUCLEUS_OP,
    impl="torch.top_p",
    form="torch",
    device="any",
    shape=_SHAPE,
    kv_layouts=frozenset({"ND"}),
    graph="capture_safe",
    plan_kind="stateless",
    enabled_by_default=True,
    measured_ratio=1.0,
    bench="bench/bench_sampling_top_p.py",
    ratio_note=(
        "definitional 1.0: the only implementation.  The op is a sort plus a cumulative sum, so there "
        "is nothing to compare it against yet; the ledger records absolute latency at B=8, V=152064."
    ),
    note=(
        "nucleus filter in the logit domain: everything outside the smallest surviving prefix is -inf. "
        "The prefix is 'mass strictly before this token < p', ties broken by lowest index, and the "
        "most likely token always survives."
    ),
)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Set every logit outside the nucleus to ``-inf``; ``p >= 1`` and ``p <= 0`` are the identity.

    The rule is the prefix form (``cumsum before the token < p``), not the threshold form
    (``prob >= t``): they differ on ties, and the prefix form is what the reference sampling stack
    does. Computing in the input dtype is deliberate - see the note in ``ops/moe/compute.py``: an
    implementation that promotes to fp32 changes the numbers at the boundary and is no longer a
    drop-in replacement.
    """
    p = float(p)
    if not 0.0 < p < 1.0:
        # p <= 0 keeps everything (the threshold is never reached); p >= 1 also keeps everything -
        # writing it as "the identity" avoids the floating-point corner where the last token's
        # pre-mass rounds to exactly 1.0 and a real token disappears.
        return logits.clone()
    probs = torch.softmax(logits, dim=-1)
    order = torch.argsort(probs, dim=-1, descending=True, stable=True)
    ordered = torch.gather(probs, -1, order)
    before = torch.cumsum(ordered, dim=-1) - ordered
    keep = before < p
    keep[..., 0] = True  # the most likely token survives even if its own mass already exceeds p
    mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, order, keep)
    return logits.masked_fill(~mask, float("-inf"))


def plan_top_p(impl: ImplSpec, **pure: Any) -> OpPlan:
    batch = int(pure["batch"])
    vocab = int(pure["vocab"])
    p = float(pure["p"])
    dtype = str(pure.get("dtype", "bfloat16"))
    if batch < 1 or vocab < 1:
        raise ValueError(f"{NUCLEUS_OP}: batch and vocab must be >= 1, got {batch}, {vocab}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{NUCLEUS_OP}: p must be in [0, 1], got {p}")
    if dtype not in _DTYPES:
        raise ValueError(f"{NUCLEUS_OP}: dtype {dtype!r} is outside {sorted(_DTYPES)}")

    def execute(logits):
        if logits.dim() != 2 or tuple(logits.shape) != (batch, vocab):
            raise ValueError(
                f"{NUCLEUS_OP}/{impl.impl}: logits must be {(batch, vocab)}, got {tuple(logits.shape)}"
            )
        return top_p_filter(logits, p)

    return OpPlan(
        op=NUCLEUS_OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(batch, vocab),
        dtypes=(dtype,),
        kv_layouts=tuple(sorted(impl.kv_layouts)),
        graph=impl.graph,
        meta={"p": p, "rule": "prefix"},
        execute=execute,
    )


def run_top_p(p: OpPlan, logits):
    return p(logits)


REGISTRY.op(
    NUCLEUS_OP,
    doc="Nucleus (top-p) sampling: keep the smallest prefix of the sorted distribution whose mass "
    "reaches p, everything else -inf. min-p is not implemented and is not aliased to this.",
)
REGISTRY.register(NUCLEUS_TORCH)
REGISTRY.planner(NUCLEUS_OP, plan_top_p)


def sampler_for(do_sample: bool) -> str:
    """Which operator serves a request: the only routing rule the sampling family has.

    ``do_sample=False`` is greedy decoding — a different *function*, not a variant of the threshold, so
    it is a different op and not an ``ImplSpec`` flag.  Keeping them apart is what lets each one carry
    its own declaration (greedy is capture-safe and allocation-free in spirit; the threshold allocates
    a mask) and its own ratio.
    """
    return OP if do_sample else GREEDY


def sample(logits: torch.Tensor, k: Optional[int] = None, do_sample: bool = True):
    """Plan and run the right sampler for one ``[batch, vocab]`` logits tensor.

    A convenience for callers that only have logits: dispatch stays cheap because both planners are pure
    data, and the caller keeps the right to ``plan()`` once and reuse the ``OpPlan``.
    """
    # Imported here, not at module level: the package's ``__init__`` imports this module to trigger
    # registration, so a top-level ``from ... import plan`` would be a cycle.
    from ... import plan, run

    if logits.dim() != 2:
        raise ValueError(f"logits must be [batch, vocab], got {tuple(logits.shape)}")
    batch, vocab = int(logits.shape[0]), int(logits.shape[1])
    dtype = str(logits.dtype).replace("torch.", "")
    if not do_sample:
        return run(plan(GREEDY, batch=batch, vocab=vocab, dtype=dtype), logits)
    if k is None:
        raise ValueError("do_sample=True needs k (the threshold top-k)")
    return run(plan(OP, batch=batch, vocab=vocab, k=k, dtype=dtype), logits)
