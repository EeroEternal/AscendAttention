"""The three acceptance mechanisms that were still prose: padding poison, pool offsets, prewarm keys.

Each one corresponds to a line in ``docs/plan-migrate-tileinfer.md`` §7 that says "writing it in a doc is
not solving it; a test is".  They are grouped because they share a property: they only fail *silently*
in production, so the test is the only place the mistake is ever visible.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention import plan, run
from ascend_attention.abi import capacity_view
from ascend_attention.testing.cache import build_paged_cache


def _cache(batch=1, h_kv=2, kv_len=53, d=16, page_size=16, poison=0.0):
    torch.manual_seed(11)  # the poison and the clean cache must differ in exactly one thing
    k = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float32)
    v = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float32)
    k_cache, v_cache, meta = build_paged_cache(k, v, page_size=page_size, seed=11)
    if poison:
        # slots of the last page that the request does not own: exactly what a kernel would read if it
        # trusted ``page_size`` instead of ``kv_last_page_len``
        last_len = int(meta.kv_last_page_len[0])
        page_ids = meta.kv_indices[
            int(meta.kv_indptr[0]) : int(meta.kv_indptr[0 + 1])
        ].tolist()
        last_page = page_ids[-1]
        for slot in range(last_len, page_size):
            k_cache[last_page, slot] = poison
            v_cache[last_page, slot] = poison
    return k_cache, v_cache, meta


def test_partial_last_page_padding_does_not_change_the_result():
    """Poison the unused tail of the last page with 1e4; the output must be bitwise identical.

    A kernel (or an oracle) that attends to ``page_size`` tokens per page instead of ``kv_last_page_len``
    reads that garbage.  The score is huge but finite, so the only visible symptom is a slightly wrong
    answer - which is why this is a *bitwise* comparison of two runs, not a tolerance check.
    """
    q = torch.randn(1, 4, 5, 16, dtype=torch.float32) * 0.3
    k_clean, v_clean, meta = _cache(poison=0.0)
    k_poison, v_poison, meta_p = _cache(poison=1e4)
    assert torch.equal(meta.kv_indptr, meta_p.kv_indptr)

    def _run(k_cache, v_cache, meta):
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
            head_dim=16,
            qo_len=5,
            batch=1,
            dtype="float32",
            device="cpu",
        )
        return run(p, q)

    clean = _run(k_clean, v_clean, meta)
    poisoned = _run(k_poison, v_poison, meta_p)
    assert torch.equal(clean, poisoned), (
        "padding leaked into the result: "
        f"{(clean - poisoned).abs().max().item():.3e}"
    )


def test_capacity_view_keeps_the_storage_offset():
    """A pool view must stay at the original offset: V lives right after K in a fused pool."""
    pool = torch.arange(4 * 4 * 2 * 8, dtype=torch.float32).reshape(4, 4, 2, 8)  # [pages, slots, h, d]
    v_half = pool[2:]  # storage_offset == 128 elements
    assert v_half.storage_offset() == 128
    view = capacity_view(v_half, 2)
    assert view.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
    assert view.storage_offset() == 128
    assert torch.equal(view, v_half[:2])
    # the failure this guards: rebinding at offset 0 would return K's rows instead of V's
    assert not torch.equal(view, pool[:2])


def test_capacity_view_pads_without_moving_the_offset():
    pool = torch.ones(4, 4, 2, 8, dtype=torch.float32)
    view = capacity_view(pool, 6)  # declared capacity larger than the pool: a bigger view, no copy
    assert view.shape == (6, 4, 2, 8)
    assert view.storage_offset() == 0


def test_prewarm_lists_the_compile_keys_without_compiling():
    """``prewarm`` is pure data: the platform decides *when* to pay the compile, not the library."""
    from ascend_attention import prewarm

    keys = prewarm(["paged_decode_attention:decode950:b8:d128", "paged_decode_attention:decode950:b8:d128"])
    assert keys == ("paged_decode_attention:decode950:b8:d128",)
    with pytest.raises(TypeError):
        prewarm(None)  # type: ignore[arg-type]
