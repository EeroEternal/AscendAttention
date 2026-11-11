"""MoE compute ops: routing, block alignment, and the unquantised expert MLP.

Why these live *here* and not in the platform layer: they are arithmetic with shape, dtype and
graph constraints - the same class of thing as ``silu_and_mul`` in any fused-op library.  A platform
layer owns device semantics and engine symbol binding; it must not own a GEMM.  The boundary review
(``docs/boundary.md``) records how the previous split left this path owned by nobody.

Signatures are the **algorithm's**, not an engine's:

* ``silu_and_mul(x)``                     -> ``silu(x[..., :d]) * x[..., d:]``
* ``topk_softmax(scores, top_k, ...)``    -> ``(weights, ids)``
* ``moe_align_block_size(topk_ids, ...)`` -> ``(sorted_ids, expert_ids, num_tokens_post_pad)``
* ``fused_experts(hidden, w1, w2, weights, ids, ...)`` -> routed MLP output
* ``unquantized_fused_moe(...)``          -> the same math under the "method" entry point

The platform's vocabulary maps its engine symbols onto these entries; nothing here imports or names
an engine.  Every function is pure torch, so it is also the fallback when no kernel is available.

Precision policy: every function computes in the **input dtype**, matching the engine's own native
fallback (``F.silu(x[..., :d]) * x[..., d:]``, ``softmax`` in the input dtype, per-expert matmuls in the
input dtype).  An earlier version promoted half inputs to float32, which is more accurate but *different*:
substituting one implementation for the other would then change the numbers by ~1 ulp per element, and
the whole point of these baselines is that the platform can swap them in.  A real kernel is a different
story - it may accumulate in fp32, and that is a claim it has to benchmark, not something to hide in a
baseline.  The fp64 oracles in ``tests/test_moe_compute.py`` are what keep the tolerance honest.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch

__all__ = [
    "align_block_size_reference",
    "fused_experts_reference",
    "moe_align_block_size",
    "silu_and_mul",
    "topk_softmax",
    "unquantized_fused_moe",
]

#: The padding sentinel ``moe_align_block_size`` writes into the tail of every expert's block run.
#: It equals ``num_tokens * top_k`` - an index one past the last real row - so a consumer that forgets
#: to mask reads row 0 of the gathered tile rather than out of bounds (the OOB-gather trap).
PAD_SENTINEL = "num_tokens * top_k"


def _activation(name: Any) -> str:
    """Normalise an activation selector.

    Engines pass an enum in one call path and its string name in another (the wrapper translates), so
    accepting both spellings is part of the *semantics* here, not an engine detail.  Only SiLU is
    implemented: claiming gelu while computing silu is exactly the silent-wrong-number failure this
    library exists to prevent, so anything else raises.
    """
    if name is None:
        return "silu"
    if isinstance(name, str):
        key = name.strip().lower()
    else:  # an enum or any object with a ``name``/``value``
        for attribute in ("name", "value"):
            value = getattr(name, attribute, None)
            if isinstance(value, str):
                key = value.strip().lower()
                break
        else:  # pragma: no cover - defensive
            raise TypeError(f"unsupported activation selector {name!r}")
    if key not in ("silu", "swish"):
        raise NotImplementedError(
            f"activation {key!r} is not implemented (only SiLU/swish); refusing rather than "
            "computing the wrong function"
        )
    return "silu"


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[..., :d]) * x[..., d:]`` - the SwiGLU gate applied to a fused gate/up projection.

    Input last dimension must be even and is halved.  Computed in the input dtype, on purpose: the
    engine's native fallback is ``F.silu(x[..., :d]) * x[..., d:]`` with no promotion, so promoting here
    would make this a *different* function at ~1 ulp per element (see the module docstring).
    """
    if x.shape[-1] % 2:
        raise ValueError(f"silu_and_mul needs an even last dimension, got {tuple(x.shape)}")
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]


def topk_softmax(
    scores: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax over experts, then keep the ``top_k`` largest and (optionally) renormalise.

    Returns ``(weights, ids)`` with ``weights`` in the input dtype and ``ids`` as int32 - the width the
    block-alignment op and every grouped-GEMM kernel index with.

    ``renormalize=True`` is the serving default (the kept weights sum to 1).  With it off, a token's
    weights sum to less than 1 because the discarded experts still consumed probability mass; that
    difference is observable in the output, so it is an argument rather than a fixed choice.

    Ties are broken by lowest expert id (``torch.topk``'s ``sorted=True`` on a stable sort), which is
    what makes a bf16 vocabulary with duplicate logits reproducible run to run.
    """
    if scores.dim() != 2:
        raise ValueError(f"topk_softmax expects [tokens, experts], got {tuple(scores.shape)}")
    experts = scores.shape[1]
    if not 0 < top_k <= experts:
        raise ValueError(f"top_k={top_k} must be in 1..{experts}")
    probs = torch.softmax(scores, dim=-1)
    # ``torch.topk`` does not promise which of several equal values it keeps, and duplicated bf16
    # logits are common at a 150k vocabulary, so the choice has to be pinned: a *stable* descending
    # sort keeps the lowest expert id first among equals, which makes a run reproducible.
    candidates, ids = torch.sort(probs, dim=-1, descending=True, stable=True)
    weights = candidates[..., :top_k]
    ids = ids[..., :top_k]
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(probs.dtype).tiny
        )
    return weights.to(scores.dtype), ids.to(torch.int32)


def align_block_size_reference(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """The loop form of :func:`moe_align_block_size`, kept separate as the oracle.

    Written as the definition (count -> pad each expert to whole blocks -> lay the runs out in expert
    order) so a vectorised implementation has something independent to be checked against.
    """
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be [tokens, top_k], got {tuple(topk_ids.shape)}")
    num_tokens, top_k = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int64)
    sentinel = num_tokens * top_k
    if flat.numel() and int(flat.max()) >= num_experts:
        raise ValueError(f"expert id {int(flat.max())} is outside num_experts={num_experts}")

    order: list = []
    expert_ids: list = []
    cursor = 0
    for expert in range(num_experts):
        positions = torch.nonzero(flat == expert, as_tuple=False).reshape(-1).tolist()
        blocks = -(-len(positions) // block_size)
        for block in range(blocks):
            chunk = positions[block * block_size : (block + 1) * block_size]
            order.extend(chunk)
            order.extend([sentinel] * (block_size - len(chunk)))
            expert_ids.append(expert)
            cursor += block_size

    total = cursor
    sorted_ids = torch.full((max(total, 1),), sentinel, dtype=torch.int32)
    if total:
        sorted_ids[:total] = torch.tensor(order, dtype=torch.int32)
    expert_ids_t = torch.full((max(len(expert_ids), 1),), -1, dtype=torch.int32)
    if expert_ids:
        expert_ids_t[: len(expert_ids)] = torch.tensor(expert_ids, dtype=torch.int32)
    return sorted_ids, expert_ids_t, total


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    *,
    num_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group ``(token, expert)`` pairs by expert and pad each expert's run up to ``block_size`` rows.

    This is the layout a grouped GEMM consumes: ``sorted_ids`` holds the flattened pair index
    (``token * top_k + slot``) in expert-major, block-padded order; ``expert_ids`` names the expert of
    each block; ``num_tokens_post_pad`` is the number of padded rows actually in use.

    Buffer sizes are the engine's fixed ones - ``pairs + num_experts * block_size`` for ``sorted_ids``
    and ``pairs // block_size + num_experts`` for ``expert_ids``, both sentinel-filled past the used
    prefix.  That is deliberate: a captured step must not allocate per call, and the tail is what a
    consumer masks off.  ``num_tokens`` only widens the sentinel value when the buffer is compiled for
    more tokens than this step has.

    Returns ``num_tokens_post_pad`` as a **0-dim device tensor** and never reads it on the host: every
    number the caller needs is already device-side, so the op is capturable and does not sync.  The
    loop form lives in :func:`align_block_size_reference` and the test compares the two.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be [tokens, top_k], got {tuple(topk_ids.shape)}")
    num_tokens = int(topk_ids.shape[0] if num_tokens is None else num_tokens)
    pairs = topk_ids.numel()
    # No host-side "ids are in range" check: reading ``topk_ids.max()`` would sync (and break graph
    # capture).  ``index_add_`` below raises on an out-of-range id on the device, so a bad routing
    # output still fails loudly - just without a round trip.

    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    # ``bincount`` is not implemented on every backend, and ``index_add_`` is: same count, no host sync.
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device).index_add_(
        0, flat, torch.ones_like(flat)
    )
    padded = ((counts + block_size - 1) // block_size) * block_size
    starts = torch.cumsum(padded, 0) - padded
    order = torch.argsort(flat, stable=True)
    rank = torch.empty_like(flat)
    run_start = torch.zeros(num_experts, dtype=torch.int64, device=device)
    if num_experts > 1:
        run_start[1:] = torch.cumsum(counts, 0)[:-1]
    rank[order] = torch.arange(pairs, dtype=torch.int64, device=device) - run_start[flat[order]]

    sentinel = max(num_tokens * int(topk_ids.shape[1]), pairs)
    buffer_len = max(pairs + num_experts * block_size, 1)
    sorted_ids = torch.full((buffer_len,), sentinel, dtype=torch.int32, device=device)
    sorted_ids[starts[flat] + rank] = torch.arange(pairs, dtype=torch.int32, device=device)

    block_count = pairs // block_size + num_experts
    expert_ids = torch.full((block_count,), -1, dtype=torch.int32, device=device)
    expert_ids[: padded.sum() // block_size] = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device), padded // block_size
    )
    return sorted_ids, expert_ids, padded.sum().to(torch.int32)


def _expert_weights(
    w1: torch.Tensor, w2: torch.Tensor, expert: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return w1[expert], w2[expert]


def fused_experts_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: Any = "silu",
    apply_router_weight_on_input: bool = False,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-token loop form of the routed expert MLP - the oracle for :func:`fused_experts`.

    ``w1``: ``[experts, 2*inter, hidden]`` (gate rows first, then up rows), ``w2``:
    ``[experts, hidden, inter]``, matching the serving layout so no weight reshaping is hidden here.
    """
    act = _activation(activation)
    if act != "silu":  # pragma: no cover - guarded by _activation
        raise NotImplementedError(act)
    tokens, hidden = hidden_states.shape
    experts = w1.shape[0]
    out = torch.zeros(tokens, hidden, dtype=torch.float64)
    hidden64 = hidden_states.to(torch.float64)
    for token in range(tokens):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            if expert < 0:
                continue
            if expert_map is not None:
                expert = int(expert_map[expert])
                if expert < 0:
                    continue
            w1e, w2e = w1[expert].to(torch.float64), w2[expert].to(torch.float64)
            weight = float(topk_weights[token, slot])
            source = hidden64[token] * weight if apply_router_weight_on_input else hidden64[token]
            projected = source @ w1e.T
            inter = projected.shape[-1] // 2
            gated = torch.nn.functional.silu(projected[:inter]) * projected[inter:]
            contribution = gated @ w2e.T
            if not apply_router_weight_on_input:
                contribution = contribution * weight
            out[token] += contribution
    assert experts > 0
    return out.to(hidden_states.dtype)


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: Any = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: Optional[int] = None,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Any = None,
) -> torch.Tensor:
    """Routed expert MLP: for each ``(token, expert)`` pair, ``w2 @ silu(w1 @ x) * weight``, summed.

    The positional order is the one the engine's Python wrapper uses, because that wrapper is where
    the platform hands the call over; the *math* is what this function owns.  ``quant_config`` is
    refused rather than ignored: this is the unquantised path, and silently computing bf16 while the
    caller asked for fp8 is the failure mode that looks like a numerics bug for a week.

    ``expert_map`` maps a global expert id to a local one (``-1`` = not on this device), which is how
    expert parallelism reaches this function.
    """
    _activation(activation)  # refuse an unimplemented activation instead of quietly computing SiLU
    if quant_config is not None:
        raise NotImplementedError(
            "fused_experts here is the unquantised path; quantised MoE is not implemented "
            "(refusing to compute in the wrong precision)"
        )
    if global_num_experts is not None and global_num_experts != w1.shape[0]:
        if expert_map is None:
            raise ValueError(
                f"global_num_experts={global_num_experts} but w1 holds {w1.shape[0]} experts; "
                "without expert_map the mapping is ambiguous"
            )
    if hidden_states.dim() != 2:
        raise ValueError(f"fused_experts expects [tokens, hidden], got {tuple(hidden_states.shape)}")
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"topk_ids has {topk_ids.shape[0]} rows but hidden_states has {hidden_states.shape[0]}"
        )
    tokens = hidden_states.shape[0]
    compute = hidden_states.dtype  # the engine's native path computes in the input dtype
    experts = w1.shape[0]
    out = torch.zeros(tokens, hidden_states.shape[1], dtype=compute, device=hidden_states.device)
    source = hidden_states

    # Map global expert ids to local ones *once*: selecting with ``topk_ids == expert`` would compare
    # global ids against local indices, which is silently wrong as soon as expert_map != identity
    # (it read the wrong expert's weights and disagreed with the reference by ~8.6 absolute).
    mapped_ids = (
        topk_ids.to(torch.int64) if expert_map is None else expert_map[topk_ids.to(torch.int64)]
    )
    # One pass over the *experts* (a static count), and every per-pair number stays on the device:
    # an earlier version looped over pairs and did ``int(...)``/``float(...)`` per pair, which is a
    # host round trip for every (token, expert) pair and makes the op impossible to capture.
    for expert in range(experts):
        selection = torch.nonzero(mapped_ids == expert, as_tuple=False)
        if selection.shape[0] == 0:
            continue
        tokens_sel = selection[:, 0]
        slots_sel = selection[:, 1]
        # row-major flat index of the (token, slot) pair; ``selection.reshape(-1)`` would interleave
        # token and slot indices and silently scale the wrong rows (a 2-vs-4 broadcast error caught it)
        pair_index = tokens_sel * mapped_ids.shape[1] + slots_sel
        source_rows = source[tokens_sel]
        weights = topk_weights.flatten()[pair_index].to(compute).unsqueeze(-1)
        if apply_router_weight_on_input:
            source_rows = source_rows * weights
        projected = source_rows @ w1[expert].T
        inter = projected.shape[-1] // 2
        gated = torch.nn.functional.silu(projected[:, :inter]) * projected[:, inter:]
        contribution = gated @ w2[expert].T
        if not apply_router_weight_on_input:
            contribution = contribution * weights
        out.index_add_(0, tokens_sel, contribution)
    return out.to(hidden_states.dtype)


def unquantized_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: Optional[int] = None,
    activation: Any = "silu",
    apply_router_weight_on_input: bool = False,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Any = None,
) -> torch.Tensor:
    """The unquantised "method" entry point: routing already done, expert MLP applied and combined.

    It exists as its own entry because the engine's method object is what a serving path constructs,
    and that object has to expose *one* callable with this shape.  The math is
    :func:`fused_experts`; this function only fixes the argument names and refuses the quantised
    configuration loudly instead of falling through to the bf16 path.
    """
    if quant_config is not None:
        raise NotImplementedError(
            "unquantized_fused_moe refuses a quant_config: quantised experts are not implemented"
        )
    if num_experts is not None and num_experts != w1.shape[0]:
        raise ValueError(f"num_experts={num_experts} but w1 holds {w1.shape[0]} experts")
    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        apply_router_weight_on_input,
        global_num_experts=num_experts,
        expert_map=expert_map,
    )
