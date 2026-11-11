"""The prefill kernel must be compiled per **bucket**, not per prompt length.

``num_tiles`` is a linear function of the query length, and ``_cache_key`` puts the whole spec in the
key (deliberately: anything that changes the emitted program has to be there). The two facts together
meant every new sequence length compiled a new variant - measured on the reference machine at 129.7 s
of GDN prefill inside one 192-token run, against 0.8 s for the same step with the artifact already
built. The plan stays per step (that is the contract); the *compiled kernel* is shared.

These tests run on CPU: a fake builder records the spec, so the bucketing and the packing/unpacking of
padded tiles are exercised without the toolchain. Numerics of a padded tile are the kernel's own P3
contract, already covered by the device suite.
"""

from __future__ import annotations

import types

import pytest
import torch

from ascend_attention.abi import _TILE_BUCKETS, quantise_tiles
from ascend_attention.kernels.attention import paged_decode_ascend950 as k
from ascend_attention.metadata import RaggedMetadata
from ascend_attention.plan import plan_query_tiles_from_host

GROUP, KV_HEADS, DIM, PAGE = 8, 2, 128, 64


def _step(length: int, *, pages: int = 4, fill: float = 3.0, monkeypatch=None):
    """One ``forward_prefill`` call whose kernel is a dict-recording stub."""
    q = torch.zeros(1, KV_HEADS * GROUP, length, DIM, dtype=torch.bfloat16)
    k_cache = torch.zeros(pages, PAGE, KV_HEADS, DIM, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    meta = RaggedMetadata(
        kv_indptr=torch.tensor([0, pages], dtype=torch.int32),
        kv_indices=torch.arange(pages, dtype=torch.int32),
        kv_last_page_len=torch.tensor([PAGE], dtype=torch.int32),
        page_size=PAGE,
        qo_indptr=torch.tensor([0, length], dtype=torch.int32),
    )
    schedule = plan_query_tiles_from_host(
        [length * GROUP], [pages], device=torch.device("cpu"), block_q=128
    )
    recorded = {}

    def fake_build(**spec):
        recorded.update(spec)

        def kernel(*args):
            out = args[-1]
            out.fill_(fill)
            return out

        return kernel

    monkeypatch.setattr(k, "build_prefill_kernel", fake_build)
    out = k.forward_prefill(
        q,
        k_cache,
        v_cache,
        meta,
        schedule,
        torch.tensor([length], dtype=torch.int32),
        group=GROUP,
        block_q=128,
    )
    return recorded, schedule.num_tiles, out


def test_quantise_tiles_is_monotone_and_bucketed():
    assert [quantise_tiles(n) for n in (0, 1, 2, 3, 17, 128, 129)] == [1, 1, 2, 4, 32, 128, 256]
    assert quantise_tiles(512) == 512
    with pytest.raises(ValueError):
        quantise_tiles(_TILE_BUCKETS[-1] + 1)
    with pytest.raises(ValueError):
        quantise_tiles(-1)


def test_two_lengths_in_one_bucket_compile_one_variant(monkeypatch):
    spec_a, live_a, out_a = _step(200, monkeypatch=monkeypatch)
    spec_b, live_b, out_b = _step(230, monkeypatch=monkeypatch)
    assert live_a != live_b, "the two lengths must really produce different live tile counts"
    assert quantise_tiles(live_a) == quantise_tiles(live_b)
    assert spec_a["num_tiles"] == spec_b["num_tiles"] == quantise_tiles(live_a)
    # and the whole key is identical, not just that one field
    assert k._cache_key("prefill", spec_a) == k._cache_key("prefill", spec_b)
    # the *result* still has one row per query position: sharing a kernel does not share the output
    assert out_a.shape[2] == 200 and out_b.shape[2] == 230


def test_a_length_in_another_bucket_does_compile_a_different_variant(monkeypatch):
    spec_small, _, _ = _step(16, monkeypatch=monkeypatch)
    spec_large, _, _ = _step(4096, pages=64, monkeypatch=monkeypatch)
    assert spec_small["num_tiles"] < spec_large["num_tiles"]
    assert k._cache_key("prefill", spec_small) != k._cache_key("prefill", spec_large)


def test_padded_tiles_are_inert():
    schedule = plan_query_tiles_from_host([8 * GROUP], [2], device=torch.device("cpu"), block_q=128)
    padded = k._pad_schedule(schedule, 8, torch.device("cpu"))
    assert padded.num_tiles == 8
    assert padded.q_lens.tolist()[: schedule.num_tiles] == schedule.q_lens.tolist()
    assert padded.q_lens.tolist()[schedule.num_tiles :] == [0] * (
        8 - schedule.num_tiles
    ), "an appended tile must carry q_len == 0, the state the kernel masks out"
    assert padded.seq_ids.tolist()[schedule.num_tiles :] == [0] * (8 - schedule.num_tiles)
