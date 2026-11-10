"""Device numerics for the Ascend 950 paged decode kernel (``npu`` marker).

The CPU suite proves the *contract* (plan/run shape, dtype, gate state).  This file proves the
*numbers*, because a kernel that is byte-plausible and silently wrong is the failure mode the whole
library is built to prevent: bf16 against a fp32 oracle of the gathered KV, ``TOL = 3e-2`` — the same
tolerance TileInfer uses for this kernel, so the two repos' results are comparable.

Ragged batch (one full page, one tail page, one single-token request) and a shuffled page table are
in the test on purpose: the tail-page and page-order paths are where this kernel historically broke.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention.metadata import RaggedMetadata
from ascend_attention.plan import OpPlan
from ascend_attention.testing.reference import reference_attention

try:  # pragma: no cover - environment dependent
    import tilelang.ascend  # noqa: F401

    _HAVE_ASCEND950 = True
except Exception:  # pragma: no cover
    _HAVE_ASCEND950 = False

from ascend_attention.utils import is_npu_available

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
    pytest.mark.skipif(not _HAVE_ASCEND950, reason="tilelang.ascend (>=0.1.15) not importable"),
]

DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"
TOL = 3e-2


def _build(batch, kv_heads, group, page_size, dim, kv_lens, seed=0):
    device = torch.device("npu", 0)
    torch.manual_seed(seed)

    indptr = [0]
    last_page_len = []
    for length in kv_lens:
        indptr.append(indptr[-1] + -(-length // page_size))
        last_page_len.append(page_size if length % page_size == 0 else length % page_size)
    total_pages = indptr[-1]

    k_cache = torch.randn(total_pages, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    v_cache = torch.randn(total_pages, page_size, kv_heads, dim, dtype=DTYPE, device=device)

    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for b in range(batch):
        start, end = indptr[b], indptr[b + 1]
        perm = torch.randperm(end - start, generator=gen)
        indices[start:end] = indices[start:end][perm.to(device)]

    meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        kv_indices=indices,
        kv_last_page_len=torch.tensor(last_page_len, dtype=torch.int32, device=device),
        page_size=page_size,
    )
    q = torch.randn(batch, kv_heads * group, 1, dim, dtype=DTYPE, device=device) * 0.2
    return q, k_cache, v_cache, meta, indptr, last_page_len


def _decode_plan(q, k_cache, v_cache, meta, batch, kv_heads, group, dim, page_size):
    from ascend_attention import plan

    return plan(
        "paged_decode_attention",
        impl="tilelang.decode950",
        batch=batch,
        k_cache=k_cache,
        v_cache=v_cache,
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=page_size,
        num_qo_heads=kv_heads * group,
        num_kv_heads=kv_heads,
        head_dim=dim,
        causal=False,
        dtype=DTYPE_NAME,
        device=torch.device("npu", 0),
    )


def _oracle(q, k_cache, v_cache, meta, indptr, last_page_len, page_size, batch):
    """Host-side fp32 oracle over the *same* page table (shuffled pages included)."""
    host_meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32),
        kv_indices=meta.kv_indices.cpu(),
        kv_last_page_len=torch.tensor(last_page_len, dtype=torch.int32),
        page_size=page_size,
    )
    host_meta.qo_indptr = torch.arange(batch + 1, dtype=torch.int32)
    return reference_attention(
        q.cpu().float(), k_cache.cpu().float(), v_cache.cpu().float(), host_meta
    ).float()


def test_decode_plan_is_data_only_and_carries_no_kernel_until_run():
    q, k_cache, v_cache, meta, _, _ = _build(3, 2, 4, 64, 128, [128, 100, 1])
    p = _decode_plan(q, k_cache, v_cache, meta, 3, 2, 4, 128, 64)
    assert isinstance(p, OpPlan)
    data = p.describe()
    assert data["impl"] == "tilelang.decode950"
    assert data["has_handle"] is True  # finalize_plan compiled during plan()
    assert data["meta"]["head_dim"] == 128


def test_decode_matches_fp32_oracle_ragged():
    """One full page, one tail page, one single-token request; shuffled page table."""
    batch, kv_heads, group, page_size, dim = 3, 2, 4, 64, 128
    q, k_cache, v_cache, meta, indptr, last_page_len = _build(
        batch, kv_heads, group, page_size, dim, [128, 100, 1]
    )
    p = _decode_plan(q, k_cache, v_cache, meta, batch, kv_heads, group, dim, page_size)
    out = p(q)
    assert tuple(out.shape) == tuple(q.shape)

    expected = _oracle(q, k_cache, v_cache, meta, indptr, last_page_len, page_size, batch)
    err = (out.cpu().float() - expected).abs().max().item()
    assert err < TOL, f"max abs error {err:.4f} >= {TOL}"


def test_decode_head_dim_256_matches_oracle():
    """The other compiled head_dim; 256 was added later and has its own tiling."""
    batch, kv_heads, group, page_size, dim = 2, 1, 8, 128, 256
    q, k_cache, v_cache, meta, indptr, last_page_len = _build(
        batch, kv_heads, group, page_size, dim, [256, 129]
    )
    p = _decode_plan(q, k_cache, v_cache, meta, batch, kv_heads, group, dim, page_size)
    out = p(q)
    expected = _oracle(q, k_cache, v_cache, meta, indptr, last_page_len, page_size, batch)
    err = (out.cpu().float() - expected).abs().max().item()
    assert err < TOL, f"max abs error {err:.4f} >= {TOL}"
