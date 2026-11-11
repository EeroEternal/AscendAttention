"""The in-tree prefill kernel, graded against the fp64 reference that the API already ships.

Two things are being checked, and only the second needs a device:

* **structural** (CPU): the kernel plan refuses to be built without query lengths. That refusal is the
  whole contract - a prefill plan *is* this step's tile count - and a "nice default" here would silently
  truncate a longer prompt;
* **numerical** (NPU): append/prefill with a real past. Row 0 of the new segment must see the cached
  prefix and nothing after its own position, which is the ``causal_shift = kv_len - qo_len`` rule the
  kernel's per-row thresholds implement.

The reference is ``impl="torch.reference"`` for the same inputs, compared in fp32; both are exercised
through the public ``plan``/``run`` verbs so this test cannot pass by calling an internal helper.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention import plan, run
from ascend_attention.utils import is_npu_available

PREFILL = "paged_prefill_attention"


def test_the_kernel_plan_needs_query_lengths():
    """No qo_lens and no qo_indptr: there is no honest tile count, so refuse instead of guessing 1."""
    with pytest.raises(ValueError, match="needs qo_lens"):
        plan(
            PREFILL,
            impl="tilelang.prefill950",
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim=128,
            qo_len=64,
            dtype="bfloat16",
            k_cache=torch.zeros(2, 16, 2, 128, dtype=torch.bfloat16),
            v_cache=torch.zeros(2, 16, 2, 128, dtype=torch.bfloat16),
            kv_indptr=torch.tensor([0, 2], dtype=torch.int32),
            kv_indices=torch.tensor([0, 1], dtype=torch.int32),
            kv_last_page_len=torch.tensor([16], dtype=torch.int32),
        )


@pytest.mark.npu
@pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available")
def test_append_matches_the_reference_with_a_real_past():
    torch.manual_seed(0)
    device = torch.device("npu", 0)
    batch, h_q, h_kv, d = 2, 4, 2, 128
    past, qo_len, page_size = 32, 64, 16
    kv_len = past + qo_len
    dtype = torch.bfloat16

    from ascend_attention.testing.cache import build_paged_cache

    k = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float32) / 8
    v = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float32) / 8
    q = torch.randn(batch, h_q, qo_len, d, dtype=torch.float32) / 8
    # the new segment is the last qo_len slots: an append step, so causal_shift = past
    k_cache, v_cache, meta = build_paged_cache(
        k.to(dtype).to(device), v.to(dtype).to(device), page_size=page_size, device=device
    )

    common = dict(
        num_qo_heads=h_q,
        num_kv_heads=h_kv,
        head_dim=d,
        page_size=page_size,
        qo_len=qo_len,
        qo_lens=torch.full((batch,), qo_len, dtype=torch.int32, device=device),
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        k_cache=k_cache,
        v_cache=v_cache,
        dtype="bfloat16",
        device=device,
    )
    kernel_plan = plan(PREFILL, impl="tilelang.prefill950", **common)
    assert kernel_plan.meta["num_tiles"] > 0
    assert kernel_plan.graph == "eager_only"

    ref_plan = plan(PREFILL, impl="torch.reference", **common)
    q_npu = q.to(dtype).to(device)
    got = run(kernel_plan, q_npu)
    expected = run(ref_plan, q_npu)

    assert tuple(got.shape) == (batch, h_q, qo_len, d)
    # bf16 both sides: 2e-2 absolute is well inside one bf16 ulp at these magnitudes
    assert torch.allclose(got.float(), expected.float(), atol=2e-2, rtol=2e-2)

    # causal_shift is the append rule; without it row 0 of the segment would see its own suffix.
    # Moving the cache content of the *future* slots must not change the output.
    k2 = k.clone()
    k2[:, :, past + qo_len - 1] += 3.0
    k2_cache, v2_cache, meta2 = build_paged_cache(
        k2.to(dtype).to(device), v.to(dtype).to(device), page_size=page_size, device=device
    )
    tweaked = plan(
        PREFILL,
        impl="tilelang.prefill950",
        **{**common, "k_cache": k2_cache, "v_cache": v2_cache, "kv_indptr": meta2.kv_indptr,
           "kv_indices": meta2.kv_indices, "kv_last_page_len": meta2.kv_last_page_len},
    )
    got2 = run(tweaked, q_npu)
    first_row = 40  # a token whose window ends before the tweaked slot
    assert torch.allclose(
        got2[:, :, first_row].float(), got[:, :, first_row].float(), atol=2e-2, rtol=2e-2
    ), "a token saw a KV position after its own query position"
