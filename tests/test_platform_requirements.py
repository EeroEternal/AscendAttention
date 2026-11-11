"""The platform's required-name table must resolve against this package, name by name.

This is the test that was missing when 8 of 13 required operators existed here and none could be bound:
the platform's family table imports every operator unconditionally, so one missing module wiped out the
whole table - and nothing in either repo asked "does this name exist *here*?".

Ownership is deliberate and lives in one dict: an engine-facing *name* (the platform's vocabulary) mapped
to a *call site* in this library.  It is data, not a generated table, so a rename here fails loudly
instead of silently unbinding.

The platform is an optional consumer, so this skips when it is not installed - but it never weakens into
"skip when the name is missing": that would be the original bug with a test around it.
"""

from __future__ import annotations

import importlib
from typing import Dict

import pytest

#: Engine-facing name -> ``package.module:attribute`` inside this library.
#: Every entry is an *algorithm* signature; the platform adapts its own ABI onto it.
REQUIRED_TO_TARGET: Dict[str, str] = {
    "chunk_gated_delta_rule": "ascend_attention.ops.fla.chunk:chunk_gated_delta_rule",
    "l2norm_fwd": "ascend_attention.ops.gdn.leaves:l2norm_fwd",
    "causal_conv1d_fn": "ascend_attention.ops.fla.causal_conv1d:causal_conv1d_fn",
    "causal_conv1d_update": "ascend_attention.ops.fla.causal_conv1d:causal_conv1d_update",
    "fused_sigmoid_gating_delta_rule_update": (
        "ascend_attention.ops.fla.gating:fused_sigmoid_gating_delta_rule_update"
    ),
    "fused_recurrent_gated_delta_rule_packed_decode": (
        "ascend_attention.ops.fla.recurrent:fused_recurrent_gated_delta_rule_packed_decode"
    ),
    "chunk_local_cumsum": "ascend_attention.ops.gdn.leaves:chunk_local_cumsum",
    "fused_post_conv_prep": "ascend_attention.ops.fla.post_conv:fused_post_conv_prep",
    "topk_softmax": "ascend_attention.ops.moe.compute:topk_softmax",
    "moe_align_block_size": "ascend_attention.ops.moe.compute:moe_align_block_size",
    "fused_experts": "ascend_attention.ops.moe.compute:fused_experts",
    "unquantized_fused_moe": "ascend_attention.ops.moe.compute:unquantized_fused_moe",
    "silu_and_mul": "ascend_attention.ops.moe.compute:silu_and_mul",
}


def _resolve(target: str):
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    missing = object()
    value = getattr(module, attribute, missing)
    assert value is not missing, f"{module_name} has no attribute {attribute!r}"
    return value


@pytest.mark.parametrize("name,target", sorted(REQUIRED_TO_TARGET.items()))
def test_each_engine_facing_name_resolves_inside_this_library(name, target):
    assert callable(_resolve(target)), name


def test_the_mapping_covers_every_name_the_platform_requires():
    try:
        from ascend_platform.ops.families.gdn_moe import required_names
    except Exception:  # pragma: no cover - platform not installed
        pytest.skip("AscendPlatform is not importable; the cross-repo half cannot be checked here")

    required = set(required_names())
    mapped = set(REQUIRED_TO_TARGET)
    assert not required - mapped, (
        f"the platform requires names this library does not map: {sorted(required - mapped)}"
    )
    assert not mapped - required, (
        f"this library maps names the platform no longer requires: {sorted(mapped - required)}"
    )


def test_the_platform_resolver_returns_our_functions_for_the_semantic_keys_it_asks():
    """The strongest form of this test: ask the way *their* table asks, and get *our* callable back.

    Their resolver tries the provider first (by the key written in ``gdn_moe.py``) and falls back to the
    layout path. Pinning the returned object - not just "something resolved" - is what catches a rename on
    either side: a fallback that happens to import the same function under the old layout would look
    identical in a "did it bind?" test and different here.
    """
    try:
        from ascend_platform.ops.families.gdn_moe import _Impl
    except Exception:  # pragma: no cover - platform not installed
        pytest.skip("AscendPlatform is not importable; the cross-repo half cannot be checked here")

    import ascend_attention.ops.moe.compute as compute

    resolver = _Impl()
    for key, target, function in [
        ("moe.routing_topk_softmax", "topk_softmax", compute.topk_softmax),
        ("moe.align_block_size", "moe_align_block_size", compute.moe_align_block_size),
        ("moe.experts", "fused_experts", compute.fused_experts),
        ("moe.swiglu_gate", "silu_and_mul", compute.silu_and_mul),
        ("unquantized_fused_moe", "unquantized_fused_moe", compute.unquantized_fused_moe),
    ]:
        value, note = resolver.impl(key, "moe", target)
        assert value is function, f"{key}: got {value!r} ({note})"


def test_the_platform_parser_accepts_the_descriptor_we_publish():
    """Our document has to survive *their* parser, including the fields they ignore.

    Their ``parse_descriptor`` is the gate at discovery time, and it is stricter than a dict shape check:
    it validates ``target`` as ``module:attribute`` and rejects unknown *structure* while ignoring unknown
    keys. Now that the descriptor carries 19 ops and per-op notes (which mention punctuation, quotes and
    parentheses), a note that broke the schema would fail at their start-up rather than here.
    """
    try:
        from ascend_platform.contracts.provider import parse_descriptor
    except Exception:  # pragma: no cover - platform not installed
        pytest.skip("AscendPlatform is not importable; the parser half cannot be checked here")

    import ascend_attention

    parsed = parse_descriptor(ascend_attention.describe())
    ours = set(ascend_attention.describe()["ops"])
    theirs = set(parsed.ops)
    assert theirs == ours, f"parser dropped or invented ops: {sorted(ours ^ theirs)}"
    assert parsed.name == "ascend_attention"


def test_the_mapping_itself_is_free_of_engine_imports():
    """The mapping is a *name* table, not an import of the engine: a target must be ours."""
    for name, target in REQUIRED_TO_TARGET.items():
        assert target.startswith("ascend_attention."), (name, target)
