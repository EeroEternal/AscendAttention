"""Semantic operator names: this library's op name -> the platform's vocabulary.

The platform binds engine symbols onto *semantic* ops (``moe.experts``), not onto our internal names
(``moe_fused_experts``): it must be able to say "I need the expert MLP" without knowing which library
implements it. This table is the mapping, it is deliberately small and hand-maintained, and it is
published inside every manifest entry (``platform_name``) so the platform never hardcodes our names.

Two rules, both enforced by tests rather than by convention:

* every registered op has a name here - an op without one cannot be advertised, and a new op with no
  semantic name is exactly the drift that made the platform's table unresolvable;
* names are ``family.snake_case``, lowercase ASCII, no engine vocabulary, no implementation detail
  (``moe.experts``, never ``moe_fused_experts_v2``). Renaming one is a breaking change for the
  platform, so the names are pinned in a test snapshot.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping

__all__ = ["PLATFORM_NAMES", "NAME_PATTERN", "platform_name", "op_for", "unknown_ops", "duplicates"]

#: ``family.member``: lowercase, digits and underscores, exactly one dot.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")

#: op name registered in this library -> semantic name the platform binds.
PLATFORM_NAMES: Mapping[str, str] = {
    # attention
    "paged_decode_attention": "attention.paged_decode",
    "paged_prefill_attention": "attention.paged_prefill",
    "quant_dequant_fp8": "quant.dequant_fp8",
    "quant_fp8_gemm": "quant.fp8_gemm",
    # gated delta net
    "gdn_gating": "gdn.gating",
    "gdn_conv_update": "gdn.conv_update",
    "gdn_chunk_wu": "gdn.chunk_wu",
    # mixture of experts: layout, routing, activation, compute
    "moe_slot_layout": "moe.slot_layout",
    "moe_prenz_pack": "moe.prenz_pack",
    "moe_topk_softmax": "moe.routing_topk_softmax",
    "moe_align_block_size": "moe.align_block_size",
    "moe_silu_and_mul": "moe.swiglu_gate",
    "moe_fused_experts": "moe.experts",
    "moe_unquantized_fused_moe": "moe.experts_unquantized",
    # sampling
    "sampling_topk_threshold": "sampling.topk_threshold",
    "sampling_top_p": "sampling.top_p",
    "sampling_min_p": "sampling.min_p",
    "sampling_greedy": "sampling.greedy",
}

_INVERSE: Dict[str, str] = {}


def platform_name(op: str) -> str:
    """The semantic name for one of our ops. Missing is a hard error, not an empty string."""
    try:
        return PLATFORM_NAMES[op]
    except KeyError:
        raise KeyError(
            f"op {op!r} has no semantic name; add it to vocabulary.PLATFORM_NAMES "
            f"(the platform cannot bind an op it cannot name)"
        ) from None


def op_for(semantic: str) -> str:
    """Inverse of :func:`platform_name`; ``KeyError`` lists what is actually bindable."""
    if not _INVERSE:
        for op, semantic_name in PLATFORM_NAMES.items():
            _INVERSE[semantic_name] = op
    try:
        return _INVERSE[semantic]
    except KeyError:
        raise KeyError(
            f"unknown semantic op {semantic!r}; this library provides: {sorted(_INVERSE)}"
        ) from None


def unknown_ops(registered: Mapping[str, object] | None = None) -> tuple:
    """Registered ops with no semantic name (a registration-time bug, not a runtime condition)."""
    if registered is None:
        from .spec import REGISTRY

        registered = {name: None for name in REGISTRY.names()}
    return tuple(sorted(name for name in registered if name not in PLATFORM_NAMES))


def duplicates() -> tuple:
    """Semantic names claimed twice; the platform would bind one of them arbitrarily."""
    seen: Dict[str, str] = {}
    clashes = []
    for op, semantic in PLATFORM_NAMES.items():
        if semantic in seen:
            clashes.append((seen[semantic], op, semantic))
        seen[semantic] = op
    return tuple(clashes)
