"""MLA decode: the latent-attention definition, checked against a straight-line oracle.

MLA is the op where an implementation is *plausible* and wrong in ways ordinary attention tests miss:

* the softmax scale is ``1/sqrt(D_c + D_r)`` (two concatenated dot products), not ``1/sqrt(D_c)``;
* the value side is the shared latent ``c_kv``, not a per-head V;
* the paged gather must take exactly ``kv_len`` positions and ignore the rest of the last page;
* a zero-length request returns zeros rather than ``nan`` from an empty softmax.

The oracle below is written from the definition with explicit Python loops in fp64, so it shares no code
with the implementation beyond ``torch.softmax``.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import pytest
import torch

from ascend_attention import plan, run
from ascend_attention.ops.mla import OP, kv_len_from_pages, mla_decode_reference

ROOT = Path(__file__).resolve().parents[1]


def _gather(c_kv, k_pe, kv_indices, indptr, b, page_size, length):
    pages = kv_indices[int(indptr[b]) : int(indptr[b + 1])].tolist()
    flat_c = c_kv.reshape(-1, c_kv.shape[-1])[torch.tensor([p * page_size + i for p in pages for i in range(page_size)])]
    flat_pe = k_pe.reshape(-1, k_pe.shape[-1])[torch.tensor([p * page_size + i for p in pages for i in range(page_size)])]
    return flat_c[:length], flat_pe[:length]


def _oracle(q_nope, q_pe, c_kv, k_pe, indptr, kv_indices, last_page_len, page_size):
    """Definition, in fp64, one request and one head at a time."""
    batch, heads, _, latent = q_nope.shape
    rope = q_pe.shape[-1]
    scale = 1.0 / math.sqrt(latent + rope)
    out = torch.zeros(batch, heads, 1, latent, dtype=torch.float64)
    for b in range(batch):
        pages = int(indptr[b + 1] - indptr[b])
        length = (pages - 1) * page_size + int(last_page_len[b]) if pages else 0
        if length <= 0:
            continue
        c, pe = _gather(c_kv, k_pe, kv_indices, indptr, b, page_size, length)
        for h in range(heads):
            # the two dot products are summed *before* the softmax, each contributing its own score
            s = torch.zeros(length, dtype=torch.float64)
            for j in range(length):
                s[j] = (q_nope[b, h, 0].to(torch.float64) * c[j]).sum() * scale + (
                    q_pe[b, h, 0].to(torch.float64) * pe[j]
                ).sum() * scale
            p = torch.softmax(s, dim=-1)
            acc = torch.zeros(latent, dtype=torch.float64)
            for j in range(length):
                acc += p[j] * c[j]
            out[b, h, 0] = acc
    return out


def _case(batch=2, heads=3, latent=128, rope=64, page_size=8, seed=0):
    torch.manual_seed(seed)
    pages = [1, 2] if batch == 2 else [1] * batch
    indptr = torch.tensor([0, *torch.cumsum(torch.tensor(pages), 0).tolist()], dtype=torch.int32)
    total = int(indptr[-1])
    kv_indices = torch.randperm(total, dtype=torch.long)
    last = torch.tensor([page_size] * batch, dtype=torch.int32)
    c_kv = torch.randn(total, page_size, latent, dtype=torch.float64)
    k_pe = torch.randn(total, page_size, rope, dtype=torch.float64)
    q_nope = torch.randn(batch, heads, 1, latent, dtype=torch.float64) / math.sqrt(latent)
    q_pe = torch.randn(batch, heads, 1, rope, dtype=torch.float64) / math.sqrt(rope)
    return q_nope, q_pe, c_kv, k_pe, indptr, kv_indices, last, page_size


def test_matches_the_definition():
    q_nope, q_pe, c_kv, k_pe, indptr, kv_indices, last, page_size = _case()
    got = mla_decode_reference(
        q_nope, q_pe, c_kv, k_pe,
        kv_indptr=indptr, kv_indices=kv_indices, kv_last_page_len=last, num_heads=3,
    )
    expected = _oracle(q_nope, q_pe, c_kv, k_pe, indptr, kv_indices, last, page_size)
    assert torch.allclose(got.double(), expected, atol=1e-10)


def test_the_scale_is_over_the_sum_of_both_widths():
    """Using 1/sqrt(D_c) instead is a ~1.4x temperature error that no shape check catches."""
    q_nope, q_pe, c_kv, k_pe, indptr, kv_indices, last, page_size = _case(latent=128, rope=64)
    got = mla_decode_reference(
        q_nope, q_pe, c_kv, k_pe, kv_indptr=indptr, kv_indices=kv_indices,
        kv_last_page_len=last, num_heads=3,
    )
    wrong_scale = mla_decode_reference(
        q_nope, q_pe, c_kv, k_pe, kv_indptr=indptr, kv_indices=kv_indices,
        kv_last_page_len=last, num_heads=3, sm_scale=1.0 / math.sqrt(128),
    )
    assert not torch.allclose(got, wrong_scale, atol=1e-3)


def test_the_last_page_is_partial_and_the_tail_is_ignored():
    """A wrong gather reads padding rows and the softmax silently attends to noise."""
    torch.manual_seed(1)
    latent, rope, heads, page_size = 128, 64, 2, 8
    c_kv = torch.randn(1, page_size, latent, dtype=torch.float64)
    k_pe = torch.randn(1, page_size, rope, dtype=torch.float64)
    q_nope = torch.randn(1, heads, 1, latent, dtype=torch.float64)
    q_pe = torch.randn(1, heads, 1, rope, dtype=torch.float64)
    indptr = torch.tensor([0, 1], dtype=torch.int32)
    live = 3
    last = torch.tensor([live], dtype=torch.int32)
    got = mla_decode_reference(
        q_nope, q_pe, c_kv, k_pe, kv_indptr=indptr, kv_indices=torch.tensor([0]),
        kv_last_page_len=last, num_heads=heads,
    )
    full = mla_decode_reference(
        q_nope, q_pe, c_kv, k_pe, kv_indptr=indptr, kv_indices=torch.tensor([0]),
        kv_last_page_len=torch.tensor([page_size], dtype=torch.int32), num_heads=heads,
    )
    assert not torch.allclose(got, full), "the padding rows must not contribute"
    # the partial-length result equals the definition over the first `live` rows only
    scores = (q_nope[0, :, 0] @ c_kv[0, :live].T + q_pe[0, :, 0] @ k_pe[0, :live].T) / math.sqrt(
        latent + rope
    )
    expected = torch.softmax(scores, dim=-1) @ c_kv[0, :live]
    assert torch.allclose(got[0, :, 0], expected, atol=1e-10)


def test_a_zero_length_request_returns_zeros_not_nan():
    q_nope, q_pe, c_kv, k_pe, indptr, kv_indices, last, page_size = _case(batch=2)
    # request 0 has no pages at all: indptr[0] == indptr[1]
    indptr = torch.tensor([0, 0, int(kv_indices.numel())], dtype=torch.int32)
    last = torch.tensor([0, page_size], dtype=torch.int32)
    got = mla_decode_reference(
        q_nope, q_pe, c_kv, k_pe, kv_indptr=indptr, kv_indices=kv_indices,
        kv_last_page_len=last, num_heads=3,
    )
    assert torch.count_nonzero(got[0]) == 0
    assert torch.isfinite(got).all()


def test_kv_len_from_pages_handles_a_partial_last_page():
    indptr = torch.tensor([0, 2, 5], dtype=torch.int32)
    last = torch.tensor([7, 16], dtype=torch.int32)
    lengths = kv_len_from_pages(indptr, last, page_size=16)
    assert lengths.tolist() == [1 * 16 + 7, 2 * 16 + 16]


def test_plan_and_run_and_the_declared_contract():
    q_nope, q_pe, c_kv, k_pe, indptr, kv_indices, last, page_size = _case(
        batch=1, heads=2, latent=128, rope=64
    )
    # The declared contract is the serving dtypes (bf16/fp16/fp32); fp64 is only used for the oracles,
    # which call the function directly rather than through a plan.
    q_nope, q_pe, c_kv, k_pe = (t.float() for t in (q_nope, q_pe, c_kv, k_pe))
    p = plan(
        OP, impl="torch.reference", num_heads=2, kv_indptr=indptr, kv_indices=kv_indices,
        kv_last_page_len=last, c_kv=c_kv, k_pe=k_pe, dtype="float32", device="cpu",
    )
    out = run(p, q_nope, q_pe)
    assert tuple(out.shape) == (1, 2, 1, 128)
    with pytest.raises(ValueError, match="outside the declared set"):
        plan(
            OP, num_heads=2, kv_indptr=indptr, kv_indices=kv_indices, kv_last_page_len=last,
            c_kv=torch.randn(1, page_size, 100), k_pe=k_pe, dtype="float32", device="cpu",
        )
    with pytest.raises(ValueError, match="q_pe must be"):
        run(p, q_nope, torch.randn(1, 2, 1, 32, dtype=torch.float32))


def test_manifest_entry_bench_and_ledger_exist():
    from ascend_attention import manifest as manifest_module

    entry = next(op for op in manifest_module.manifest()["ops"] if op["name"] == OP)
    assert entry["platform_name"] == "attention.mla_decode"
    assert entry["measured_ratio"] == 1.0
    assert (ROOT / entry["bench"]).is_file()
    ledger = ROOT / "bench" / "results" / "mla_decode_b8.json"
    assert ledger.is_file()
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["ms"]["mla_decode"]["median_ms"] > 0
