"""The five MoE compute ops, checked against definitions written independently of them.

Why this file matters more than the layout tests: the boundary review found these ops owned by nobody,
and the failure they cause is not a crash - the platform binds an engine symbol to a torch function, the
service starts, and the first MoE step returns plausible numbers. So every op here is compared to a
loop written from the definition, in fp64, with the tie/edge cases that a vectorised implementation
gets wrong: duplicate logits, an expert with zero tokens, a weight applied on the input instead of the
output, and an expert_map that skips a global expert.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention import plan, run
from ascend_attention.ops.moe import compute

DEVICE = "cpu"
TOL = 1e-5


def _plan(op, impl, **pure):
    return plan(op, impl=impl, device=DEVICE, **pure)


# ------------------------------------------------------------------ silu_and_mul


def test_silu_and_mul_matches_the_definition():
    torch.manual_seed(0)
    x = torch.randn(5, 16, dtype=torch.float64)
    gate, up = x[..., :8], x[..., 8:]
    expected = torch.nn.functional.silu(gate) * up
    got = compute.silu_and_mul(x)
    assert torch.allclose(got, expected, atol=1e-12)


def test_silu_and_mul_computes_in_the_input_dtype_like_the_engine_fallback():
    """bf16 in, bf16 out, **no f32 promotion**: the op must be substitutable for the engine's own path.

    A promoted version is more accurate and still fails this test - that is the point.  Promoting by ~1
    ulp per element makes "swap the implementation" change the numbers, which is exactly what the
    platform's substitution contract forbids.
    """
    torch.manual_seed(0)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    got = compute.silu_and_mul(x)
    assert got.dtype == torch.bfloat16
    reference = torch.nn.functional.silu(x[..., :4]) * x[..., 4:]
    assert torch.equal(got, reference)
    promoted = (torch.nn.functional.silu(x[..., :4].float()) * x[..., 4:].float()).to(torch.bfloat16)
    # if these ever become equal for every seed, this test has stopped proving anything
    assert not torch.equal(promoted, reference)


def test_silu_and_mul_refuses_an_odd_width():
    with pytest.raises(ValueError, match="even last dimension"):
        compute.silu_and_mul(torch.randn(2, 7))
    with pytest.raises(ValueError, match="even fused width"):
        _plan("moe_silu_and_mul", "torch.silu_and_mul", width=7, dtype="float32")


# ------------------------------------------------------------------ topk_softmax


def _topk_reference(scores, top_k, renormalize):
    probs = torch.softmax(scores.to(torch.float64), dim=-1)
    values, ids = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    return values, ids


def test_topk_softmax_matches_the_definition():
    torch.manual_seed(1)
    scores = torch.randn(6, 8, dtype=torch.float64)
    weights, ids = compute.topk_softmax(scores, 3)
    expected_w, expected_i = _topk_reference(scores, 3, True)
    assert torch.equal(ids.to(torch.int64), expected_i)
    assert torch.allclose(weights, expected_w, atol=1e-12)
    assert weights.sum(-1).sub(1.0).abs().max() < 1e-6


def test_renormalize_false_keeps_the_discarded_mass_visible():
    """The switch must change the numbers, otherwise it is decoration."""
    torch.manual_seed(2)
    scores = torch.randn(4, 16, dtype=torch.float64)
    full, _ = compute.topk_softmax(scores, 2, renormalize=True)
    partial, _ = compute.topk_softmax(scores, 2, renormalize=False)
    assert torch.allclose(partial, _topk_reference(scores, 2, False)[0], atol=1e-12)
    assert partial.sum(-1).max() < 0.999, "top-k of 16 experts should not already sum to 1"
    assert not torch.allclose(full, partial)


def test_ties_are_broken_by_the_lowest_expert_id():
    """Duplicated bf16 logits are common at 150k vocabulary; the order must be reproducible."""
    scores = torch.full((1, 4), 1.0, dtype=torch.float32)
    _, ids = compute.topk_softmax(scores, 2)
    assert ids.tolist() == [[0, 1]]


def test_topk_softmax_ids_are_int32_and_in_range():
    weights, ids = compute.topk_softmax(torch.randn(3, 5), 2)
    assert ids.dtype == torch.int32
    assert int(ids.max()) < 5
    assert weights.dtype == torch.float32


def test_topk_soffmax_rejects_k_beyond_the_expert_count():
    with pytest.raises(ValueError, match="top_k"):
        compute.topk_softmax(torch.randn(2, 4), 5)
    with pytest.raises(ValueError, match="top_k"):
        _plan("moe_topk_softmax", "torch.topk_softmax", num_experts=4, top_k=9, dtype="float32")


# ------------------------------------------------------------------ moe_align_block_size


def test_align_block_size_matches_its_own_loop_form():
    torch.manual_seed(3)
    ids = torch.randint(0, 5, (7, 2), dtype=torch.int32)
    sorted_ids, expert_ids, total = compute.moe_align_block_size(ids, 5, 4)
    ref_sorted, ref_experts, ref_total = compute.align_block_size_reference(ids, 5, 4)
    assert int(total) == ref_total
    assert torch.equal(sorted_ids[:ref_total], ref_sorted[:ref_total])
    assert torch.equal(expert_ids[: ref_total // 4], ref_experts[: ref_total // 4])


def test_align_block_size_handles_an_expert_with_no_tokens():
    """An empty expert still occupies its own (zero-length) run; the block count must not shift."""
    ids = torch.tensor([[0], [0], [3]], dtype=torch.int32)
    sorted_ids, expert_ids, total = compute.moe_align_block_size(ids, 4, 2)
    ref_sorted, ref_experts, ref_total = compute.align_block_size_reference(ids, 4, 2)
    assert int(total) == ref_total == 4
    assert torch.equal(sorted_ids[:4], ref_sorted[:4])
    assert torch.equal(expert_ids[:2], ref_experts[:2])
    # rows 0..1 are expert 0, rows 2..3 are expert 3 -> no block belongs to experts 1 or 2
    assert expert_ids[0].item() == 0 and expert_ids[1].item() == 3


def test_align_block_size_tail_is_the_sentinel_not_a_real_pair():
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    sorted_ids, _, total = compute.moe_align_block_size(ids, 2, 4)
    assert int(total) == 8, "two experts x one block of four"
    assert sorted_ids[int(total) :].eq(2).all(), "everything past the used prefix is num_tokens*top_k"


def test_align_block_size_is_sync_free_and_buffers_are_the_fixed_size():
    ids = torch.randint(0, 3, (5, 2), dtype=torch.int32)
    sorted_ids, expert_ids, total = compute.moe_align_block_size(ids, 3, 2)
    assert sorted_ids.numel() == 5 * 2 + 3 * 2
    assert expert_ids.numel() == (5 * 2) // 2 + 3
    assert total.dim() == 0 and total.dtype == torch.int32


def test_align_block_size_refuses_an_out_of_range_expert():
    """No host-side range check (that would sync), so the device-side index_add must be the guard."""
    with pytest.raises(Exception):
        compute.moe_align_block_size(torch.tensor([[9]], dtype=torch.int32), 4, 2)


# ------------------------------------------------------------------ fused_experts


def _weights(experts, hidden, inter, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(experts, 2 * inter, hidden, dtype=dtype),
        torch.randn(experts, hidden, inter, dtype=dtype),
    )


def test_fused_experts_matches_the_per_token_loop():
    hidden, inter, experts, tokens, top_k = 6, 5, 4, 5, 2
    w1, w2 = _weights(experts, hidden, inter)
    torch.manual_seed(4)
    x = torch.randn(tokens, hidden, dtype=torch.float64)
    ids = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32)
    weights = torch.softmax(torch.randn(tokens, top_k, dtype=torch.float64), dim=-1)

    got = compute.fused_experts(x, w1, w2, weights, ids)
    expected = compute.fused_experts_reference(x, w1, w2, weights, ids)
    assert torch.allclose(got, expected, atol=1e-10)


def test_router_weight_on_input_is_a_different_computation():
    hidden, inter, experts = 4, 3, 3
    w1, w2 = _weights(experts, hidden, inter)
    x = torch.randn(3, hidden, dtype=torch.float64)
    ids = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int32)
    weights = torch.rand(3, 2, dtype=torch.float64)

    out = compute.fused_experts(x, w1, w2, weights, ids, apply_router_weight_on_input=True)
    ref = compute.fused_experts_reference(
        x, w1, w2, weights, ids, apply_router_weight_on_input=True
    )
    assert torch.allclose(out, ref, atol=1e-10)
    plain = compute.fused_experts(x, w1, w2, weights, ids)
    assert not torch.allclose(out, plain), "scaling before vs after SiLU is not the same function"


def test_expert_map_skips_a_global_expert():
    hidden, inter, experts = 4, 3, 5
    w1, w2 = _weights(experts, hidden, inter)
    x = torch.randn(3, hidden, dtype=torch.float64)
    ids = torch.tensor([[2], [2], [2]], dtype=torch.int32)  # every token wants global expert 2
    weights = torch.ones(3, 1, dtype=torch.float64)
    skipping = torch.tensor([0, 1, -1, 3, 4], dtype=torch.int64)

    out = compute.fused_experts(x, w1, w2, weights, ids, expert_map=skipping)
    assert torch.count_nonzero(out) == 0, "the only requested expert is not on this device"
    ref = compute.fused_experts_reference(x, w1, w2, weights, ids, expert_map=skipping)
    assert torch.allclose(out, ref, atol=1e-12)


def test_global_num_experts_without_a_map_is_ambiguous_and_refused():
    w1, w2 = _weights(2, 4, 3)
    x = torch.randn(1, 4, dtype=torch.float64)
    ids = torch.tensor([[0]], dtype=torch.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        compute.fused_experts(x, w1, w2, torch.ones(1, 1), ids, global_num_experts=8)


def test_quant_config_is_refused_instead_of_ignored():
    w1, w2 = _weights(2, 4, 3)
    x = torch.randn(1, 4)
    ids = torch.tensor([[0]], dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="unquantised"):
        compute.fused_experts(x, w1, w2, torch.ones(1, 1), ids, quant_config=object())
    with pytest.raises(NotImplementedError, match="quant_config"):
        compute.unquantized_fused_moe(x, w1, w2, torch.ones(1, 1), ids, quant_config=object())


def test_unsupported_activation_is_refused():
    w1, w2 = _weights(2, 4, 3)
    x = torch.randn(1, 4)
    ids = torch.tensor([[0]], dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="not implemented"):
        compute.fused_experts(x, w1, w2, torch.ones(1, 1), ids, activation="gelu")


def test_activation_accepts_an_enum_like_selector():
    """Engines pass an enum in one path and a string in another; both spellings must mean SiLU."""

    class _Act:
        name = "silu"

    w1, w2 = _weights(2, 4, 3)
    x = torch.randn(1, 4, dtype=torch.float64)
    ids = torch.tensor([[0]], dtype=torch.int32)
    weights = torch.ones(1, 1, dtype=torch.float64)
    enum_like = compute.fused_experts(x, w1, w2, weights, ids, activation=_Act())
    text = compute.fused_experts(x, w1, w2, weights, ids, activation="silu")
    assert torch.equal(enum_like, text)


# ------------------------------------------------------------------ the three verbs


def test_all_five_ops_are_reachable_through_plan_and_run():
    torch.manual_seed(5)
    hidden, inter, experts, tokens, top_k = 4, 3, 3, 4, 1
    w1, w2 = _weights(experts, hidden, inter, dtype=torch.float32)
    x = torch.randn(tokens, hidden)
    scores = torch.randn(tokens, experts)
    weights, ids = run(
        _plan("moe_topk_softmax", "torch.topk_softmax", num_experts=experts, top_k=top_k, dtype="float32"),
        scores,
    )
    got = run(
        _plan(
            "moe_fused_experts",
            "torch.fused_experts",
            num_tokens=tokens,
            hidden=hidden,
            inter=inter,
            num_experts=experts,
            top_k=top_k,
            dtype="float32",
        ),
        x,
        w1,
        w2,
        weights,
        ids,
    )
    assert got.shape == (tokens, hidden)
    assert torch.allclose(got, compute.fused_experts_reference(x, w1, w2, weights, ids).float(), atol=TOL)

    silu = run(_plan("moe_silu_and_mul", "torch.silu_and_mul", width=8, dtype="float32"), torch.randn(2, 8))
    assert silu.shape == (2, 4)
    sorted_ids, expert_ids, total = run(
        _plan(
            "moe_align_block_size",
            "torch.align_block_size",
            num_tokens=tokens,
            top_k=top_k,
            num_experts=experts,
            block_size=2,
        ),
        ids,
    )
    assert int(total) > 0 and sorted_ids.numel() and expert_ids.numel()
    unquant = run(
        _plan(
            "moe_unquantized_fused_moe",
            "torch.unquantized_fused_moe",
            num_tokens=tokens,
            hidden=hidden,
            inter=inter,
            num_experts=experts,
            top_k=top_k,
            dtype="float32",
        ),
        x,
        w1,
        w2,
        weights,
        ids,
    )
    assert torch.allclose(unquant, got)
