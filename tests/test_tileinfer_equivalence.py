"""The ported operators must be *identical* to the ones they came from, not merely similar.

Every module here was copied from ``TileInfer`` with import paths rewritten and nothing else.  That is
a claim, and until it is checked, "we ported it" only means "we copied a file".  This test imports both
copies in one process and compares outputs bitwise (CPU, no device needed).

Skipped cleanly when the source checkout is absent — the point is provenance, not a hard dependency, and
this repo must never import TileInfer at runtime.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest
import torch

TILEINFER_SRC = pathlib.Path(__file__).resolve().parents[2] / "TileInfer"


def _import_source(module: str):
    src = TILEINFER_SRC / "src"
    if not src.exists():  # some layouts keep the package at the root
        src = TILEINFER_SRC
    if not (src / module.split(".")[0]).exists():
        pytest.skip(f"TileInfer checkout not present at {TILEINFER_SRC}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - a broken source env is not this repo's failure
        pytest.skip(f"cannot import {module} from TileInfer: {exc}")


def test_gating_matches_the_source_implementation():
    source_mod = _import_source("tileinfer.kernels.gdn")
    ours = importlib.import_module("ascend_attention.ops.gdn.gating")

    g = torch.Generator().manual_seed(0)
    A_log = torch.randn(4, generator=g, dtype=torch.float32)
    a = torch.randn(3, 5, 4, generator=g) * 0.5
    b = torch.randn(3, 5, 4, generator=g) * 0.5
    dt_bias = torch.randn(4, generator=g)

    for name in ("fused_gdn_gating", "reference_gdn_gating"):
        want = getattr(source_mod, name)(A_log, a, b, dt_bias)
        got = getattr(ours, name)(A_log, a, b, dt_bias)
        if isinstance(want, (tuple, list)):
            assert all(torch.equal(x, y) for x, y in zip(got, want)), name
        else:
            assert torch.equal(got, want), name


def test_conv_and_chunk_chain_match_the_source():
    source_fla = _import_source("tileinfer.vllm.ops.fla")
    ours = importlib.import_module("ascend_attention.ops.fla")

    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 8, generator=g) * 0.3  # [batch, dim] for the decode form
    state = torch.zeros(4, 8, 3)
    weight = torch.randn(8, 4, generator=g) * 0.3
    bias = torch.randn(8, generator=g) * 0.2
    idx = torch.tensor([1], dtype=torch.int32)
    want = source_fla.causal_conv1d_update(
        x, state.clone(), weight, bias, "silu", conv_state_indices=idx, null_block_id=None
    )
    got = ours.causal_conv1d_update(
        x, state.clone(), weight, bias, "silu", conv_state_indices=idx, null_block_id=None
    )
    assert torch.equal(got, want)

    # chunk chain: same inputs, same outputs, bit for bit
    q = torch.randn(1, 128, 2, 8, generator=g) * 0.3
    k = torch.randn(1, 128, 2, 8, generator=g) * 0.3
    v = torch.randn(1, 128, 2, 8, generator=g) * 0.3
    beta = torch.rand(1, 128, 2, generator=g) * 0.5
    gate = (-torch.rand(1, 128, 2, generator=g) * 0.1)
    s0 = torch.zeros(1, 2, 8, 8)
    out_w, state_w = source_fla.chunk_gated_delta_rule(
        q, k, v, gate, beta, initial_state=s0.clone(), output_final_state=True
    )
    out_g, state_g = ours.chunk_gated_delta_rule(
        q, k, v, gate, beta, initial_state=s0.clone(), output_final_state=True
    )
    assert torch.equal(out_g, out_w)
    assert torch.equal(state_g, state_w)


def test_sampling_reference_matches_the_source():
    try:
        source_ref = _import_source("tileinfer.testing.sampling_reference")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"source sampling reference unavailable: {exc}")
    ours = importlib.import_module("ascend_attention.testing.sampling_reference")
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(4, 64, generator=g)
    for name in ("mask_top_k_by_logits", "mask_top_k_by_probs"):
        assert torch.equal(getattr(ours, name)(logits, 5), getattr(source_ref, name)(logits, 5)), name


def test_moe_slot_layout_matches_the_source():
    source = _import_source("tileinfer.kernels.moe.slot_layout")
    ours = importlib.import_module("ascend_attention.ops.moe.slot_layout")
    group_list = torch.tensor([3, 5, 5, 9, 12], dtype=torch.int32)
    kw = dict(num_rows=12, rows_static=8 * 32, em_static=8, num_experts=5, block_m=32)
    want = source.slot_layout_from_group_list(group_list, **kw)
    got = ours.slot_layout_from_group_list(group_list, **kw)
    assert set(got) == set(want)
    for key in want:
        assert torch.equal(got[key], want[key]), key
