"""Coverage of TileInfer's operator bindings, checked against the real ``bind()`` table.

The P5 acceptance line is "the manifest can drive TileInfer's bindings — 100% coverage or a note for
each gap".  A hand-written coverage table rots the day someone adds a binding, so this test reads
TileInfer's own ``bind()`` records when the checkout is present and asserts that every one of them is
either covered by this repo or listed in :data:`GAPS` with a reason.

The mapping is deliberately *explicit* rather than clever: a name-based guess ("both are called
l2norm_fwd, so they must match") is how a wrong pairing survives a review.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

TILEINFER_SRC = pathlib.Path(__file__).resolve().parents[2] / "TileInfer"

#: TileInfer binding name -> the symbol in this repo that serves it (``None`` = not migrated).
COVERED = {
    "chunk_gated_delta_rule": "ascend_attention.ops.fla.chunk_gated_delta_rule",
    "l2norm_fwd": "ascend_attention.ops.gdn.l2norm_fwd",
    "causal_conv1d_fn": "ascend_attention.ops.fla.causal_conv1d_fn",
    "causal_conv1d_update": "ascend_attention.ops.fla.causal_conv1d_update",
    "fused_sigmoid_gating_delta_rule_update": "ascend_attention.ops.fla.fused_sigmoid_gating_delta_rule_update",
    "fused_recurrent_gated_delta_rule_packed_decode": (
        "ascend_attention.ops.fla.fused_recurrent_gated_delta_rule_packed_decode"
    ),
    "chunk_local_cumsum": "ascend_attention.ops.gdn.chunk_local_cumsum",
    "fused_post_conv_prep": "ascend_attention.ops.fla.post_conv.fused_post_conv_prep",
}

#: Binding name -> why it is not here.  Every gap must have a one-line reason, not just "no".
GAPS = {
    "topk_softmax": "MoE routing arithmetic: engine-side (needs expert grouping + vLLM's ABI), not a layout op",
    "moe_align_block_size": "archived: measured below the gate (docs/archive-moe.md)",
    "fused_experts": "MoE expert compute: grouped-GEMM territory, and the 0.86x negative lives there",
    "unquantized_fused_moe": "a vLLM CustomOp subclass (custom_op_oot), i.e. platform ABI, not an operator",
    "silu_and_mul": "dense family, outside this migration's four families",
}


def _tileinfer_bindings():
    src = TILEINFER_SRC / "src"
    if not src.exists():
        src = TILEINFER_SRC
    if not (src / "tileinfer").exists():
        pytest.skip(f"TileInfer checkout not present at {TILEINFER_SRC}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        module = importlib.import_module("tileinfer.vllm.ops.families.gdn_moe")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot import TileInfer bindings: {exc}")
    return [record.op for record in module.bind()]


def _resolve(dotted: str):
    module_name, _, attribute = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), attribute)


def test_every_binding_is_covered_or_explained():
    names = _tileinfer_bindings()
    assert names, "the bind() table came back empty - the test would be vacuous"
    unexplained = [n for n in names if n not in COVERED and n not in GAPS]
    assert not unexplained, f"bindings with neither a mapping nor a reason: {unexplained}"


def test_every_claimed_coverage_actually_resolves():
    for binding, dotted in COVERED.items():
        assert callable(_resolve(dotted)), f"{binding} -> {dotted} is not callable"


def test_the_gap_list_has_no_stale_entries():
    """A gap whose binding disappeared from TileInfer must be deleted here, not kept as folklore."""
    names = set(_tileinfer_bindings())
    stale = [name for name in GAPS if name not in names]
    assert not stale, f"stale gap entries (binding no longer exists): {stale}"
