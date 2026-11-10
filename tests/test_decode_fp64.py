"""Decode against an fp64 oracle written straight from the definition.

The portable backend is already checked against ``testing/reference.py``, but that oracle computes in
fp32 and uses the same page-flattening helper as the implementation -- a shared mistake would cancel out.
This file recomputes the answer in fp64 from the formula (softmax over the request's own keys, GQA head
mapping, ``kv_last_page_len`` trimming) with no shared code, at a tolerance the difference between fp32
and fp64 makes visible.
"""

from __future__ import annotations

import math

import pytest
import torch

from ascend_attention import plan, run


def _oracle_decode(q, k_cache, v_cache, meta, page_size):
    """``q``: ``[batch, h_q, 1, d]`` -> fp64 attention over each request's real KV length."""
    batch, h_q, _, d = q.shape
    _, _, h_kv, _ = k_cache.shape
    group = h_q // h_kv
    indptr = meta.kv_indptr.tolist()
    pages = meta.kv_indices.tolist()
    last = meta.kv_last_page_len.tolist()
    flat_k = k_cache.reshape(-1, h_kv, d).to(torch.float64)
    flat_v = v_cache.reshape(-1, h_kv, d).to(torch.float64)
    out = torch.zeros(batch, h_q, d, dtype=torch.float64)
    for b in range(batch):
        page_ids = pages[indptr[b] : indptr[b + 1]]
        kv_len = (len(page_ids) - 1) * page_size + last[b]
        slots = []
        for j in range(kv_len):
            page_index, offset = divmod(j, page_size)
            slots.append(page_ids[page_index] * page_size + offset)
        k = flat_k[torch.tensor(slots)]
        v = flat_v[torch.tensor(slots)]
        for h in range(h_q):
            hk = h // group
            logits = (k[:, hk] @ q[b, h, 0].to(torch.float64)) / math.sqrt(d)
            out[b, h] = torch.softmax(logits, dim=-1) @ v[:, hk]
    return out


def _case(kv_lens=(5, 9), page_size=4, h_q=4, h_kv=2, d=8, dtype=torch.float32):
    batch = len(kv_lens)
    indptr = [0]
    last = []
    for length in kv_lens:
        indptr.append(indptr[-1] + -(-length // page_size))
        last.append(page_size if length % page_size == 0 else length % page_size)
    total_pages = indptr[-1]
    torch.manual_seed(0)
    k_cache = torch.randn(total_pages, page_size, h_kv, d, dtype=dtype)
    v_cache = torch.randn(total_pages, page_size, h_kv, d, dtype=dtype)
    q = torch.randn(batch, h_q, 1, d, dtype=dtype) * 0.3
    from ascend_attention.metadata import RaggedMetadata

    meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32),
        kv_indices=torch.arange(total_pages, dtype=torch.int32),
        kv_last_page_len=torch.tensor(last, dtype=torch.int32),
        page_size=page_size,
    )
    return q, k_cache, v_cache, meta


def test_decode_matches_a_standalone_fp64_oracle():
    q, k_cache, v_cache, meta = _case()
    p = plan(
        "paged_decode_attention",
        impl="torch.reference",
        k_cache=k_cache,
        v_cache=v_cache,
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=meta.page_size,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=8,
        batch=2,
        dtype="float32",
        device="cpu",
    )
    got = run(p, q)
    expected = _oracle_decode(q, k_cache, v_cache, meta, meta.page_size)
    assert got.shape[:2] == expected.shape[:2]
    diff = (got.to(torch.float64).reshape(2, 4, 8) - expected).abs().max().item()
    assert diff < 1e-6, f"fp64 oracle disagrees by {diff:.3e}"


def test_the_oracle_notices_a_wrong_kv_length():
    """Positive control: feed the oracle one token too many and the comparison must fail.

    Without this, a case where every request happened to be page-aligned would let an off-by-one in the
    trimming survive unnoticed.
    """
    q, k_cache, v_cache, meta = _case(kv_lens=(5, 9))
    expected = _oracle_decode(q, k_cache, v_cache, meta, meta.page_size)
    wrong = meta.kv_last_page_len.clone()
    wrong[0] = meta.page_size  # pretend the tail page is full
    meta_wrong = type(meta)(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=wrong,
        page_size=meta.page_size,
    )
    loosened = _oracle_decode(q, k_cache, v_cache, meta_wrong, meta.page_size)
    assert not torch.allclose(expected, loosened, atol=1e-6)
