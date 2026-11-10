"""Prefill through the same three verbs: an fp64 oracle over a paged cache with a real past.

The prefill *kernel* is not migrated in this phase, so what is verified here is the interface and the
semantics: a cache that already holds ``past`` tokens, ``qo_len`` new tokens per request, causal masking
that must not look past the query position, and a query-side shape that differs from decode.

The oracle is written from the definition (not by calling the backend): for request ``b``, token ``t`` in
the new segment, the output is softmax over ``past_b + t + 1`` keys - the *new* segment's keys are already
in the cache at their slots, which is what makes an append step different from a decode step.
"""

from __future__ import annotations

import math

import pytest
import torch

from ascend_attention import plan, run
from ascend_attention.testing.cache import build_paged_cache


def _oracle(q, k_cache, v_cache, meta, page_size, h_kv, d, causal=True):
    """fp64 attention over the cache, expanding ``kv_indices`` (page ids) to token slots.

    ``kv_indptr`` indexes *pages*, not tokens - the one convention that is easy to read wrong and then
    "verify" a wrong implementation against a wrong oracle.
    """
    batch, h_q, qo_len, _ = q.shape
    kv_indptr = meta.kv_indptr.tolist()
    kv_last = meta.kv_last_page_len.tolist()
    pages = meta.kv_indices.tolist()
    out = torch.zeros(batch, h_q, qo_len, d, dtype=torch.float64)
    group = h_q // h_kv
    flat_k = k_cache.reshape(-1, h_kv, d).to(torch.float64)
    flat_v = v_cache.reshape(-1, h_kv, d).to(torch.float64)
    for b in range(batch):
        page_ids = pages[kv_indptr[b] : kv_indptr[b + 1]]
        kv_len = (len(page_ids) - 1) * page_size + kv_last[b]
        slots = []
        for j in range(kv_len):
            page_index, offset = divmod(j, page_size)
            slots.append(page_ids[page_index] * page_size + offset)
        k = flat_k[torch.tensor(slots)]  # [kv_len, h_kv, d]
        v = flat_v[torch.tensor(slots)]
        past = kv_len - qo_len
        for t in range(qo_len):
            limit = past + t + 1 if causal else kv_len
            for h in range(h_q):
                hk = h // group
                logits = (k[:limit, hk] @ q[b, h, t].to(torch.float64)) / math.sqrt(d)
                probs = torch.softmax(logits, dim=-1)
                out[b, h, t] = probs @ v[:limit, hk]
    return out


def _case(batch=2, h_q=4, h_kv=2, past=48, qo_len=5, page_size=16, d=32):
    """Two requests, one shared past length.

    The cache builder produces a *rectangular* ``[batch, h_kv, kv_len, d]`` cache, so per-request past
    lengths need a ragged build (``qo_indptr`` + a non-uniform cache).  Keeping this case rectangular is
    deliberate: the ragged variant is what ``kv_indptr`` expresses, and mixing the two in one test hides
    which one is being exercised.
    """
    kv_len = past + qo_len
    k = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float32)
    v = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float32)
    k_cache, v_cache, meta = build_paged_cache(k, v, page_size=page_size, shuffle_pages=True, seed=7)
    q = torch.randn(batch, h_q, qo_len, d, dtype=torch.float32) * 0.3
    return q, k_cache, v_cache, meta, kv_len


def test_prefill_matches_the_fp64_oracle():
    q, k_cache, v_cache, meta, kv_len = _case()
    p = plan(
        "paged_prefill_attention",
        k_cache=k_cache,
        v_cache=v_cache,
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=meta.page_size,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=32,
        qo_indptr=torch.tensor([0, 5, 10], dtype=torch.int32),
        dtype="float32",
        device="cpu",
    )
    out = run(p, q)
    assert out.shape == q.shape
    # the oracle needs full-length caches; keep the request's own kv_len per row
    expected = _oracle(q, k_cache, v_cache, meta, meta.page_size, 2, 32)
    assert torch.allclose(out.to(torch.float64), expected, rtol=1e-4, atol=1e-4), (
        (out.to(torch.float64) - expected).abs().max().item()
    )


def test_prefill_plan_is_cheap_and_pure_data():
    q, k_cache, v_cache, meta, _ = _case()
    kwargs = dict(
        k_cache=k_cache,
        v_cache=v_cache,
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=meta.page_size,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=32,
        qo_indptr=torch.tensor([0, 5, 10], dtype=torch.int32),
        dtype="float32",
        device="cpu",
    )
    p = plan("paged_prefill_attention", **kwargs)
    assert p.meta["mode"] == "prefill"
    assert p.shape == (2, 4, 5, 32)
    assert p.graph == "eager_only"


def test_prefill_requires_a_query_length():
    _, k_cache, v_cache, meta, _ = _case()
    with pytest.raises(ValueError, match="qo_indptr or"):
        plan(
            "paged_prefill_attention",
            k_cache=k_cache,
            v_cache=v_cache,
            kv_indptr=meta.kv_indptr,
            kv_indices=meta.kv_indices,
            page_size=meta.page_size,
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim=32,
            batch=2,
            dtype="float32",
            device="cpu",
        )


def test_prefill_rejects_illegal_head_dim_at_plan_time():
    _, k_cache, v_cache, meta, _ = _case()
    with pytest.raises(ValueError, match="outside the declared contract"):
        plan(
            "paged_prefill_attention",
            k_cache=k_cache,
            v_cache=v_cache,
            kv_indptr=meta.kv_indptr,
            kv_indices=meta.kv_indices,
            page_size=meta.page_size,
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim=48,
            qo_len=5,
            batch=2,
            dtype="float32",
            device="cpu",
        )


def test_the_unwired_kernel_is_declared_and_refuses_clearly():
    """The inventory must not imply the prefill kernel is missing - it exists, it is just not wired."""
    from ascend_attention import capabilities
    from ascend_attention.spec import REGISTRY

    op = next(o for o in capabilities()["ops"] if o["op"] == "paged_prefill_attention")
    impls = {i["impl"]: i for i in op["impls"]}
    kernel = impls["tilelang.prefill950"]
    assert kernel["enabled_by_default"] is False
    assert kernel["measured_ratio"] is None
    assert "orchestration deliberately not migrated" in kernel["note"]
    assert "paged_prefill_ascend950" in kernel["provenance"]

    _, k_cache, v_cache, meta, _ = _case()
    with pytest.raises(NotImplementedError, match="not plan-backed"):
        plan(
            "paged_prefill_attention",
            impl="tilelang.prefill950",
            k_cache=k_cache,
            v_cache=v_cache,
            kv_indptr=meta.kv_indptr,
            kv_indices=meta.kv_indices,
            page_size=meta.page_size,
            num_qo_heads=4,
            num_kv_heads=2,
            qo_indptr=torch.tensor([0, 5, 10], dtype=torch.int32),
            head_dim=128,
            dtype="bfloat16",
            device="cpu",
        )
