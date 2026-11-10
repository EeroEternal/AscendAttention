"""MoE layout ops: two independent implementations must agree (CPU).

``ops/moe/slot_layout.py`` builds the padded slot layout with ``searchsorted``/``cumsum``/``scatter``
on the device; ``testing/moe_reference.build_slot_layout`` builds the same layout with a per-expert
Python loop.  They were written from the same contract by different routes, which is exactly the
situation where a test is worth more than a review: whichever one has the index bug disagrees.

Two conventions differ on purpose and are asserted as such:

* pad entries of ``sorted_ids`` are ``rows_static`` here (compile-time bound, the kernel tests
  ``< R``) versus ``num_expanded`` in the reference (its own R).  Both mean "inactive";
* ``expert_ids`` tail beyond the actual tile count is clamped to ``E - 1`` because the buffer is
  static, and those tiles are inactive anyway.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention import plan
from ascend_attention.ops.moe import em_upper_bound
from ascend_attention.ops.moe.prenz import compact_nz_c0
from ascend_attention.testing.moe_reference import build_slot_layout

BLOCK_M = 32


def _case(num_tokens=37, top_k=4, num_experts=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), generator=g, dtype=torch.int32)
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts)
    group_list = torch.cumsum(counts, 0).to(torch.int32)
    return topk_ids, order, counts, group_list, num_tokens * top_k


def _run_slot_layout(group_list, num_rows, num_experts, block_m=BLOCK_M):
    # the slot buffer is em_static x block_m; em depends on the static row bound, which is num_rows
    # in these tests (the engine may over-provision it)
    em_static = em_upper_bound(num_rows, num_experts, block_m)
    rows_static = em_static * block_m
    p = plan(
        "moe_slot_layout",
        rows_static=rows_static,
        num_rows=num_rows,
        em_static=em_static,
        num_experts=num_experts,
        block_m=block_m,
    )
    return p(group_list), em_static


def test_slot_layout_matches_the_reference_oracle():
    topk_ids, order, counts, group_list, num_rows = _case()
    num_experts = counts.numel()
    got, em_static = _run_slot_layout(group_list, num_rows, num_experts)
    ref = build_slot_layout(topk_ids, BLOCK_M, num_experts)

    assert int(got["tiles_actual"][0]) == int(ref["num_tiles"])
    nt = int(ref["num_tiles"])
    assert torch.equal(got["expert_ids"][:nt], ref["expert_ids"][:nt])
    # every compact row lands on a *distinct* slot whose sorted_ids entry points back at the row
    slot_pos = got["slot_pos"]
    assert slot_pos.shape == (num_rows,)
    assert torch.unique(slot_pos).numel() == num_rows, "two compact rows share a slot"
    assert int(slot_pos.max()) < int(got["sorted_ids"].numel())
    # this port's sorted_ids is an *activity mask* (active slots carry their own position, pad slots
    # carry rows_static), whereas the oracle's carries the expanded row to gather - so compare the
    # compact->slot mapping itself: for compact row i, the oracle's slot is where it stores order[i].
    active = torch.zeros_like(got["sorted_ids"], dtype=torch.bool)
    active[slot_pos.long()] = True
    assert torch.equal(got["sorted_ids"][active], slot_pos.sort().values)
    ref_slot_of_expanded = {
        int(ref["sorted_ids"][s]): s for s in range(ref["sorted_ids"].numel())
        if int(ref["sorted_ids"][s]) != int(ref["num_expanded"])
    }
    for i in range(num_rows):
        assert int(slot_pos[i]) == ref_slot_of_expanded[int(order[i])], f"compact row {i} -> wrong slot"
    rows_static = em_static * BLOCK_M
    assert torch.equal(
        got["sorted_ids"][~active], torch.full_like(got["sorted_ids"][~active], rows_static)
    )


def test_slot_layout_keeps_tiles_inside_the_static_bound():
    topk_ids, _, counts, group_list, num_rows = _case(num_tokens=64, seed=1)
    got, em_static = _run_slot_layout(group_list, num_rows, counts.numel())
    nt = int(got["tiles_actual"][0])
    assert nt <= em_static
    assert got["expert_ids"].shape == (em_static,)
    # tail tiles are clamped, never out of range
    assert int(got["expert_ids"].max()) <= counts.numel() - 1


def test_slot_layout_skips_empty_experts_and_still_indexes_correctly():
    num_experts = 6
    # experts 1 and 4 are unused
    topk_ids = torch.tensor([[0, 2], [2, 3], [5, 0]], dtype=torch.int32)
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts)
    assert counts[1].item() == 0 and counts[4].item() == 0
    group_list = torch.cumsum(counts, 0).to(torch.int32)
    got, _ = _run_slot_layout(group_list, num_rows=6, num_experts=num_experts)

    ref = build_slot_layout(topk_ids, BLOCK_M, num_experts)
    nt = int(ref["num_tiles"])
    assert int(got["tiles_actual"][0]) == nt == 4  # one tile per used expert: 0, 2, 3, 5
    used = sorted({int(e) for e in ref["expert_ids"][:nt]})
    assert used == [0, 2, 3, 5]
    assert 1 not in used and 4 not in used


def test_slot_layout_plan_rejects_an_undersized_static_bound():
    with pytest.raises(ValueError, match="below the static upper bound"):
        plan(
            "moe_slot_layout",
            num_rows=64,
            rows_static=64,
            em_static=2,
            num_experts=8,
            block_m=BLOCK_M,
        )
    with pytest.raises(ValueError, match="rows_static"):
        plan(
            "moe_slot_layout",
            num_rows=64,
            rows_static=32,
            em_static=em_upper_bound(64, 8, BLOCK_M),
            # rows_static must equal em_static*block_m, so this one fails on the row check first
            num_experts=8,
            block_m=BLOCK_M,
        )
    with pytest.raises(ValueError, match="must equal em_static"):
        plan(
            "moe_slot_layout",
            num_rows=64,
            rows_static=64,
            em_static=4,
            num_experts=8,
            block_m=BLOCK_M,
        )


# --------------------------------------------------------------------------- prenz
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prenz_round_trip(dtype):
    p = plan("moe_prenz_pack", rows=8, cols=64, dtype=str(dtype).split(".")[-1])
    tile = torch.randn(8, 64, generator=torch.Generator().manual_seed(0)).to(dtype)
    packed = p(tile)
    assert packed.dtype == tile.dtype and packed.shape == tile.shape
    assert not torch.equal(packed, tile), "packing must actually permute memory"
    back = plan(
        "moe_prenz_pack", rows=8, cols=64, dtype=str(dtype).split(".")[-1], unpack=True
    )(packed)
    assert torch.equal(back, tile)


def test_prenz_physical_order_matches_the_documented_formula():
    """``phys[col//C0, row, col%C0]`` order, checked against the reshape-based implementation."""
    rows, cols = 4, 32
    c0 = compact_nz_c0(16)
    p = plan("moe_prenz_pack", rows=rows, cols=cols, dtype="bfloat16")
    tile = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols).to(torch.bfloat16)

    packed = p(tile)
    want = torch.empty(rows * cols, dtype=torch.float32)
    for col in range(cols):
        for row in range(rows):
            want[(col // c0) * (rows * c0) + row * c0 + (col % c0)] = tile[row, col].float()
    assert torch.equal(packed.reshape(-1).float(), want)


def test_prenz_rejects_a_cols_count_that_is_not_a_multiple_of_c0():
    with pytest.raises(ValueError, match="divisible by C0"):
        plan("moe_prenz_pack", rows=4, cols=12, dtype="bfloat16")


def test_prenz_pack_rejects_non_contiguous_input():
    from ascend_attention.ops.moe.prenz import pack_nd_to_compact_nz

    with pytest.raises(ValueError, match="contiguous"):
        pack_nd_to_compact_nz(torch.zeros(4, 64, 2).transpose(0, 1))


def test_compact_nz_c0_rejects_odd_element_sizes():
    assert compact_nz_c0(16) == 16 and compact_nz_c0(32) == 8
    with pytest.raises(ValueError, match="unsupported element bits"):
        compact_nz_c0(24)
