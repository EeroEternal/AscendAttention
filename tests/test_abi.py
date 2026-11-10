"""The kernel ABI arithmetic — small functions, two history-grade traps each."""

from __future__ import annotations

import pytest
import torch

from ascend_attention import abi


@pytest.mark.parametrize("group,expected", [(1, 32), (8, 32), (16, 32), (31, 32), (32, 32), (33, 64), (64, 64)])
def test_padded_rows(group, expected):
    assert abi.padded_rows(group) == expected


def test_padded_rows_rejects_non_positive():
    with pytest.raises(ValueError):
        abi.padded_rows(0)


@pytest.mark.parametrize("pages,expected", [(1, 1), (2, 2), (3, 4), (9425, 16384), (9899, 16384), (16384, 16384)])
def test_declared_pool_cap_rounds_up_to_power_of_two(pages, expected):
    assert abi.declared_pool_cap(pages) == expected


def test_declared_pool_cap_honours_explicit_pin():
    assert abi.declared_pool_cap(100, pin=4096) == 4096


def test_declared_pool_cap_rejects_pin_below_pool():
    with pytest.raises(ValueError, match="below the real pool"):
        abi.declared_pool_cap(5000, pin=4096)


def test_declared_pool_cap_reads_environment(monkeypatch):
    monkeypatch.delenv(abi.ENV_POOL_CAP, raising=False)
    monkeypatch.delenv(abi.ENV_POOL_CAP_LEGACY, raising=False)
    monkeypatch.setenv(abi.ENV_POOL_CAP, "2048")
    assert abi.declared_pool_cap(100) == 2048
    # explicit pin wins over the environment
    assert abi.declared_pool_cap(100, pin=512) == 512


def test_legacy_environment_name_is_still_honoured(monkeypatch):
    monkeypatch.delenv(abi.ENV_POOL_CAP, raising=False)
    monkeypatch.setenv(abi.ENV_POOL_CAP_LEGACY, "1024")
    assert abi.declared_pool_cap(37) == 1024


def test_declared_pool_cap_rejects_zero_pages():
    with pytest.raises(ValueError):
        abi.declared_pool_cap(0)


def _pool(pages=8, page_size=16, heads=2, dim=4, device="cpu"):
    return torch.arange(pages * page_size * heads * dim, dtype=torch.float16, device=device).reshape(
        pages, page_size, heads, dim
    )


def test_capacity_view_inflates_without_copying():
    pool = _pool(pages=8)
    view = abi.capacity_view(pool, 16)
    assert tuple(view.shape) == (16, 16, 2, 4)
    assert view.data_ptr() == pool.data_ptr()
    torch.testing.assert_close(view[:8], pool)


def test_capacity_view_at_same_capacity_is_identity():
    pool = _pool(pages=8)
    assert abi.capacity_view(pool, 8) is pool


def test_capacity_view_carries_the_storage_offset():
    """A fused pool's V half lives at a non-zero offset; re-binding at 0 silently reads K as V."""
    big = _pool(pages=16)
    half = big[8:]
    assert half.storage_offset() > 0
    view = abi.capacity_view(half, 16)
    assert view.storage_offset() == half.storage_offset()
    assert view.data_ptr() == half.data_ptr()
    torch.testing.assert_close(view[:8], half)


def test_capacity_view_rejects_shrinking():
    with pytest.raises(ValueError, match="grew past"):
        abi.capacity_view(_pool(pages=8), 4)


def test_capacity_view_requires_4d():
    with pytest.raises(ValueError, match=r"\[pages, page_size, heads, dim\]"):
        abi.capacity_view(torch.zeros(8, 16), 16)


def test_capacity_view_materialises_a_strided_pool():
    pool = torch.randn(8, 16, 2, 8, dtype=torch.float16)[..., ::2]  # non-contiguous last dim
    assert not pool.is_contiguous()
    view = abi.capacity_view(pool, 16)
    assert view.is_contiguous()
    torch.testing.assert_close(view[:8], pool.contiguous())


def test_pool_view_is_the_kernel_spelling():
    pool = _pool(pages=8)
    assert abi.pool_view(pool, 16).data_ptr() == pool.data_ptr()
