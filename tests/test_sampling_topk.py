"""Sampling top-k threshold: the contract stated three ways must agree (CPU).

The op's sentence is "keep every token whose logit is at least the k-th largest".  ``ops/sampling``
implements it with a ``topk``-derived cutoff; ``testing/sampling_reference`` states it in the logit
domain and again in the prob domain.  All three are compared here, and the *tie* behaviour is asserted
explicitly, because a sampler that drops tied tokens changes the distribution without failing.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention import plan
from ascend_attention.ops.sampling import TOP_THRESHOLD_VENDOR, _vendor_window_ok, topk_threshold
from ascend_attention.testing.sampling_reference import mask_top_k_by_logits, mask_top_k_by_probs


def _plan(dtype="float32", batch=4, vocab=64, k=5):
    return plan(
        "sampling_topk_threshold",
        impl="torch.topk_threshold",
        batch=batch,
        vocab=vocab,
        k=k,
        dtype=dtype,
    )


def test_baseline_equals_the_reference_in_both_domains():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 64, generator=g)
    got = _plan()(logits)
    assert torch.equal(got, mask_top_k_by_logits(logits, 5))
    assert torch.equal(got, mask_top_k_by_probs(logits, 5))


def test_exactly_k_tokens_survive_when_logits_are_distinct():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(8, 128, generator=g)
    got = _plan(batch=8, vocab=128, k=7)(logits)
    kept = (got > float("-inf")).sum(dim=-1)
    assert torch.equal(kept, torch.full_like(kept, 7))


def test_ties_at_the_cutoff_all_survive():
    """Threshold form keeps ties; a sort-and-take-k form would not.  This is the semantic difference."""
    logits = torch.tensor([[3.0, 3.0, 3.0, 1.0, 0.0]], dtype=torch.float32)
    got = _plan(batch=1, vocab=5, k=2)(logits)
    assert int((got > float("-inf")).sum()) == 3  # all three 3.0 survive, not just two


def test_input_is_not_mutated_and_dtype_is_preserved():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(4, 64, generator=g, dtype=torch.bfloat16)
    before = logits.clone()
    got = _plan(dtype="bfloat16")(logits)
    assert got.dtype == logits.dtype
    assert torch.equal(logits, before)
    assert not torch.equal(got, logits)


def test_k_out_of_range_is_rejected_at_plan_time():
    with pytest.raises(ValueError, match="must be in"):
        plan("sampling_topk_threshold", impl="torch.topk_threshold", batch=2, vocab=16, k=16)
    with pytest.raises(ValueError, match="must be in"):
        plan("sampling_topk_threshold", impl="torch.topk_threshold", batch=2, vocab=16, k=0)


def test_vendor_impl_is_enabled_only_on_its_own_device():
    """Enabled (3.84x measured) but device-gated, so a CPU box never routes to it."""
    assert TOP_THRESHOLD_VENDOR.enabled_by_default is True
    assert TOP_THRESHOLD_VENDOR.measured_ratio > 1.0
    assert TOP_THRESHOLD_VENDOR.device == "ascend950"
    assert TOP_THRESHOLD_VENDOR.form == "vendor_call"
    assert "deep_select" in TOP_THRESHOLD_VENDOR.requires
    # on this machine the gate either rejects it (no 950) or the vendor lib is absent; both are fine
    from ascend_attention import describe

    decision = describe("sampling_topk_threshold")
    assert decision["impls"][0]["impl"] == "torch.topk_threshold"
    # on CPU every window check short-circuits before touching the vendor library
    cpu = torch.zeros(2, 64, dtype=torch.bfloat16)
    assert _vendor_window_ok(cpu, 4) is False  # not an NPU tensor
    assert _vendor_window_ok(torch.zeros(2, 64, dtype=torch.float32), 4) is False  # bf16-only
    assert _vendor_window_ok(torch.zeros(2, 64, dtype=torch.bfloat16), 64) is False  # k < vocab


def test_module_level_helper_matches_the_plan():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(3, 32, generator=g)
    assert torch.equal(topk_threshold(logits, 4), _plan(batch=3, vocab=32, k=4)(logits))
