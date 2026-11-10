"""Vendor DeepSelect vs torch top-k on the device (``npu`` marker).

Tie-robust by construction: bf16 logits over a 152k vocabulary collide, and the two implementations
may then pick different *indices* for the same value.  Correctness here is therefore "same selected
values, same count", not "same index set" — the threshold form keeps ties and a set comparison would
report a false failure (it did, on the first run).
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention.utils import is_npu_available

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
]

BATCH, VOCAB, K = 4, 8192, 16


def _require_vendor():
    try:
        import deep_select  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"deep_select not installed: {exc}")


def test_vendor_selection_matches_torch_on_selected_values():
    _require_vendor()
    from ascend_attention import plan

    device = torch.device("npu", 0)
    torch.manual_seed(0)
    logits = torch.randn(BATCH, VOCAB, dtype=torch.bfloat16, device=device) * 2.0

    p_vendor = plan(
        "sampling_topk_threshold", impl="vendor.deep_select",
        batch=BATCH, vocab=VOCAB, k=K, dtype="bfloat16",
    )
    values, indices = p_vendor(logits)
    torch.npu.synchronize()

    expected = torch.topk(logits.float(), K, dim=-1).values
    got = torch.sort(values.float(), dim=-1).values
    assert got.shape == expected.shape
    assert torch.equal(got, torch.sort(expected, dim=-1).values), "selected values differ from torch"
    assert indices.shape == (BATCH, K) and indices.dtype == torch.int32
    # every reported index points at a real vocabulary entry
    assert int(indices.min()) >= 0 and int(indices.max()) < VOCAB


def test_vendor_selected_values_are_all_at_least_the_torch_cutoff():
    _require_vendor()
    from ascend_attention import plan

    device = torch.device("npu", 0)
    torch.manual_seed(1)
    logits = torch.randn(BATCH, VOCAB, dtype=torch.bfloat16, device=device)
    values, _ = plan(
        "sampling_topk_threshold", impl="vendor.deep_select",
        batch=BATCH, vocab=VOCAB, k=K, dtype="bfloat16",
    )(logits)
    torch.npu.synchronize()

    cutoff = torch.topk(logits.float(), K, dim=-1).values.min(dim=-1, keepdim=True).values
    assert bool((values.float() >= cutoff.to(torch.float32) - 1e-3).all())


def test_dispatch_prefers_the_vendor_impl_on_this_device():
    """bf16 + 2-D + 950 + a 3.84x record: the gate should pick it without a pin."""
    _require_vendor()
    from ascend_attention import plan

    device = torch.device("npu", 0)
    p = plan(
        "sampling_topk_threshold",
        batch=BATCH,
        vocab=VOCAB,
        k=K,
        dtype="bfloat16",
        device=device,
    )
    assert p.impl == "vendor.deep_select", p.impl
