"""Weight-shaping contracts for the GDN input projections (CPU, exact).

Two independent derivations of the same row mapping must agree, and the permutation must be a
bitwise operation rather than a near-identity: the fused GEMM accumulates over ``K`` in the same order
for every output column, so reordering columns cannot change a single bit on a real device either.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention.ops.gdn.weights import (
    apply_row_order,
    fuse_inproj_weight,
    reorder_rows_plan,
    split_interleaved,
)

CONFIGS = [
    (1, 1, 4, 4),
    (2, 2, 4, 4),
    (2, 4, 8, 8),
    (4, 8, 16, 16),
    (2, 6, 8, 8),
    (3, 12, 4, 4),
]


@pytest.mark.parametrize("k_heads,v_heads,dk,dv", CONFIGS)
def test_loop_form_and_reshape_form_agree(k_heads, v_heads, dk, dv):
    plan = reorder_rows_plan(k_heads, v_heads, dk, dv)
    expected = split_interleaved(k_heads, v_heads, dk, dv)
    assert torch.equal(plan, expected), f"row mapping differs for {k_heads/k_heads}"


@pytest.mark.parametrize("k_heads,v_heads,dk,dv", CONFIGS)
def test_plan_is_a_bijection(k_heads, v_heads, dk, dv):
    plan = reorder_rows_plan(k_heads, v_heads, dk, dv)
    assert torch.equal(torch.sort(plan).values, torch.arange(plan.numel()))


@pytest.mark.parametrize("k_heads,v_heads,dk,dv", CONFIGS)
def test_reorder_is_bitwise_for_a_gemm(k_heads, v_heads, dk, dv):
    plan = reorder_rows_plan(k_heads, v_heads, dk, dv)
    rows = plan.numel()
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(rows, 16, generator=gen, dtype=torch.float64)
    x = torch.randn(16, 4, generator=gen, dtype=torch.float64)

    permuted = apply_row_order(weight, plan)
    # every output column of the permuted GEMM equals the corresponding source column, bit for bit
    assert torch.equal(permuted @ x, (weight @ x)[plan])
    assert torch.equal(permuted, weight[plan])


def test_fuse_cats_qkvz_then_ba():
    gen = torch.Generator().manual_seed(1)
    qkvz = torch.randn(24, 8, generator=gen)
    ba = torch.randn(4, 8, generator=gen)
    fused = fuse_inproj_weight(qkvz, ba)
    assert fused.shape == (28, 8)
    assert torch.equal(fused[:24], qkvz) and torch.equal(fused[24:], ba)


def test_fuse_rejects_mismatched_hidden_and_dtype():
    with pytest.raises(ValueError, match="hidden size mismatch"):
        fuse_inproj_weight(torch.zeros(4, 8), torch.zeros(2, 6))
    with pytest.raises(ValueError, match="dtype mismatch"):
        fuse_inproj_weight(torch.zeros(4, 8), torch.zeros(2, 8, dtype=torch.float64))
    with pytest.raises(ValueError, match="2-D"):
        fuse_inproj_weight(torch.zeros(4), torch.zeros(2, 8))


def test_apply_row_order_rejects_a_truncated_plan():
    with pytest.raises(ValueError, match="permutation"):
        apply_row_order(torch.zeros(4, 3), torch.tensor([0, 1, 1, 2]))
    with pytest.raises(ValueError, match="plan has"):
        apply_row_order(torch.zeros(4, 3), torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError, match="2-D"):
        apply_row_order(torch.zeros(4), torch.tensor([0]))


def test_reorder_matches_the_documented_qwen36_layout():
    """A concrete, hand-checkable case: 2 k-heads, 4 v-heads, dk=2, dv=1.

    ``vpg = 2``, so one key-head group occupies ``[q q k k v v z z]`` (8 rows) and the ``ba`` tail has
    ``[b b a a]`` (4 rows) per group.  Consumption order is group-major flat inside each part.
    """
    k_heads, v_heads, dk, dv = 2, 4, 2, 1
    vpg = v_heads // k_heads
    group = 2 * dk + 2 * vpg * dv  # 8
    n_qkvz = k_heads * group  # 16

    # q_all / k_all: one dk-wide slice per group, groups flattened
    q_all = [g * group + j for g in range(k_heads) for j in range(dk)]
    k_all = [g * group + dk + j for g in range(k_heads) for j in range(dk)]
    v_all = [g * group + 2 * dk + j for g in range(k_heads) for j in range(vpg * dv)]
    z_all = [g * group + 2 * dk + vpg * dv + j for g in range(k_heads) for j in range(vpg * dv)]
    b_all = [n_qkvz + g * 2 * vpg + j for g in range(k_heads) for j in range(vpg)]
    a_all = [n_qkvz + g * 2 * vpg + vpg + j for g in range(k_heads) for j in range(vpg)]

    expected = q_all + k_all + v_all + z_all + b_all + a_all
    assert reorder_rows_plan(k_heads, v_heads, dk, dv).tolist() == expected
    # and the same mapping stated the other way round, for the reader
    assert q_all == [0, 1, 8, 9] and k_all == [2, 3, 10, 11]
    assert v_all == [4, 5, 12, 13] and z_all == [6, 7, 14, 15]
    assert b_all == [16, 17, 20, 21] and a_all == [18, 19, 22, 23]
