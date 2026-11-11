"""``capture_guard`` must catch the cheap mistakes and confirm the declared tiers.

The interesting part is not that a guard can see allocations; it is that the ``graph`` field in
:class:`ascend_attention.spec.ImplSpec` is a *claim about behaviour*.  Two plan-level cases are checked
here, one in each direction:

* ``sampling_topk_threshold/torch.topk_threshold`` declares ``capture_safe`` → zero host reads;
* ``gdn_conv_update/torch.causal_conv1d_update`` declares ``eager_only`` → at least one host read
  (it reads the cache-line index back to the host), i.e. the declaration is honest.

If a future edit makes the conv baseline capture-safe, this test fails and someone gets to *raise* the
tier intentionally instead of leaving a stale ``eager_only`` in the registry.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention import plan, run
from ascend_attention.capture import CaptureGuard, capture_guard, check_capture


def test_allocation_counts_but_does_not_condemn():
    with capture_guard() as guard:
        _ = torch.empty(4, 4)
    summary = guard.summary()
    assert summary["allocs"] >= 1
    assert summary["syncs"] == 0
    assert summary["capture_safe"] is True


def test_item_is_a_host_read():
    x = torch.randn(8)
    with capture_guard() as guard:
        _ = float(x.sum())
    summary = guard.summary()
    assert summary["syncs"] == 1
    assert summary["capture_safe"] is False
    assert "aten::_local_scalar_dense" in summary["sync_ops"]
    assert "host read" in guard.explain()


def test_scalar_conversion_is_a_host_read():
    """``bool(t)``/``float(t)`` go through the same op as ``.item()``.

    ``.tolist()`` and ``.cpu()`` on an *already CPU* tensor do not reach the dispatcher at all, so they
    are invisible here - a property of the box, not a hole to paper over.  Device operators are checked
    on the device (see ``test_eager_only_declaration_is_honest_for_the_conv_baseline``).
    """
    x = torch.randn(1)
    with capture_guard() as guard:
        _ = bool(x)
    assert guard.summary()["syncs"] >= 1


def test_disabled_guard_records_nothing():
    with capture_guard(enabled=False) as guard:
        _ = float(torch.randn(2).sum())
    assert guard.summary()["syncs"] == 0


def test_capture_safe_declaration_holds_for_the_sampling_baseline():
    logits = torch.randn(2, 32)
    p = plan(
        "sampling_topk_threshold",
        impl="torch.topk_threshold",
        batch=2,
        vocab=32,
        k=3,
        dtype="float32",
    )
    result = check_capture(p, logits)
    assert result["syncs"] == 0, result["detail"]
    assert result["capture_safe"] is True


def test_eager_only_declaration_is_honest_for_the_conv_baseline():
    """``eager_only`` because it reads the cache-line index back to the host; the guard sees it.

    Must run on the device: the read it performs (``conv_state_indices`` -> host list) only moves data
    when the tensor lives on an accelerator, and a CPU-only box cannot observe it.
    """
    channels, lines, width = 64, 4, 4
    x = torch.randn(channels)
    state = torch.randn(lines, channels, width - 1)
    weight = torch.randn(channels, width)
    bias = torch.randn(channels)
    p = plan(
        "gdn_conv_update",
        impl="torch.causal_conv1d_update",
        channels=channels,
        width=width,
        lines=lines,
        dtype="bfloat16",
    )
    from ascend_attention.utils import is_npu_available

    if not is_npu_available():
        pytest.skip("the host read is only observable with device tensors")
    device = torch.device("npu", 0)
    result = check_capture(
        p,
        x.to(device),
        state.to(device),
        weight.to(device),
        bias.to(device),
        1,
    )
    assert result["syncs"] >= 1
    assert result["capture_safe"] is False


def test_moe_slot_layout_allocates_but_never_reads_back():
    from ascend_attention import plan

    group_list = torch.tensor([3, 5, 5, 9], dtype=torch.int32)
    p = plan(
        "moe_slot_layout",
        rows_static=8 * 32,
        num_rows=9,
        em_static=8,
        num_experts=4,
        block_m=32,
    )
    result = check_capture(p, group_list)
    assert result["syncs"] == 0
    assert result["allocs"] >= 1  # it builds fresh index tensors: allowed, and counted


def test_guard_reports_distinct_ops_for_a_real_call():
    guard = CaptureGuard()
    with guard:
        torch.relu(torch.randn(4, 4))
    assert guard.summary()["distinct_ops"] >= 1
    assert any(op.startswith("aten.") for op in guard.ops)


# -- the MoE family: two ops that used to claim capture_safe and cannot ------------------------
#
# A reviewer from the platform read the torch references and pointed out the claim was wrong: the block
# alignment reads the pair count back to the host, and the routed MLP selects pairs with ``nonzero``,
# whose *output shape* depends on the routing data. The second one was invisible to this guard (its op
# list had no data-dependent-shape entry), which is why the registry said ``capture_safe`` for months
# without anything contradicting it. Both directions are pinned now.


def _moe_inputs():
    torch.manual_seed(0)
    pairs = torch.tensor([[0, 1], [2, 1], [1, 2]], dtype=torch.int32)
    hidden, inter, experts = 8, 4, 3
    h = torch.randn(3, hidden)
    w1 = torch.randn(experts, 2 * inter, hidden)
    w2 = torch.randn(experts, hidden, inter)
    weights = torch.full((3, 2), 0.5)
    return pairs, h, w1, w2, weights


def test_block_alignment_is_eager_only_because_it_reads_the_pair_count_back():
    from ascend_attention.ops.moe import compute

    pairs, *_ = _moe_inputs()
    with capture_guard() as guard:
        compute.moe_align_block_size(pairs, 3, 2)
    summary = guard.summary()
    assert summary["capture_safe"] is False
    assert "aten::_local_scalar_dense" in summary["sync_ops"], (
        "the op returns the padded pair count as a tensor *and* uses it for the expert map"
    )


def test_routed_experts_is_eager_only_because_nonzero_has_a_data_dependent_shape():
    from ascend_attention.ops.moe import compute

    pairs, h, w1, w2, weights = _moe_inputs()
    with capture_guard() as guard:
        compute.fused_experts(h, w1, w2, weights, pairs)
    summary = guard.summary()
    assert summary["capture_safe"] is False
    assert "aten::nonzero" in summary["shape_ops"], (
        "one nonzero per expert: the tile size depends on the routing, so it cannot enter a graph"
    )
    assert "data-dependent shape" in guard.explain()


def test_router_topk_is_capture_safe_and_the_guard_agrees():
    """``topk``/``sort``/``softmax`` are static-shape: the k is a plan-time integer."""
    from ascend_attention.ops.moe import compute

    torch.manual_seed(0)
    scores = torch.randn(6, 4)
    with capture_guard() as guard:
        compute.topk_softmax(scores, 2)
    summary = guard.summary()
    assert summary["capture_safe"] is True
    assert summary["syncs"] == 0
    # the reference sorts and gathers at plan-time-known k; no allocation claim is made here because
    # torch's topk/sort may write into caller-visible buffers instead of going through aten::empty
    assert summary["distinct_ops"] >= 3
