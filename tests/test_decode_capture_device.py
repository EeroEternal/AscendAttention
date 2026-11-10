"""Capture safety of the 950 decode kernel, verified two ways (``npu`` marker).

A declaration is not evidence, so this file re-derives the tier on the device:

* ``capture_guard`` must report **zero host reads** over a run - an allocation is fine (static shapes),
  a ``.item()``/``.cpu()``/sync is not;
* an actual ``torch.npu.NPUGraph`` capture + replay must reproduce the eager output **bitwise** - the
  guard can only see dispatched ops, and a compiled kernel's host-side staging is exactly where that
  blindness matters.

If either half fails, ``ImplSpec.graph`` for this kernel is wrong and the manifest is lying to the
platform, which is the failure this pair of tests exists to prevent.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention import plan, run
from ascend_attention.capture import capture_guard
from ascend_attention.utils import is_npu_available

try:  # pragma: no cover - environment dependent
    import tilelang.ascend  # noqa: F401

    _HAVE_ASCEND950 = True
except Exception:  # pragma: no cover
    _HAVE_ASCEND950 = False

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
    pytest.mark.skipif(not _HAVE_ASCEND950, reason="tilelang.ascend (>=0.1.15) not importable"),
]

BATCH, KV_HEADS, GROUP, PAGE, DIM = 4, 2, 4, 128, 128


def _plan():
    device = torch.device("npu", 0)
    torch.manual_seed(0)
    k_cache = torch.randn(8, PAGE, KV_HEADS, DIM, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(8, PAGE, KV_HEADS, DIM, dtype=torch.bfloat16, device=device)
    indices = torch.arange(8, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32)
    last = torch.tensor([PAGE, PAGE // 2, PAGE, PAGE // 4], dtype=torch.int32)
    q = torch.randn(BATCH, KV_HEADS * GROUP, 1, DIM, dtype=torch.bfloat16, device=device) * 0.2
    p = plan(
        "paged_decode_attention",
        impl="tilelang.decode950",
        k_cache=k_cache,
        v_cache=v_cache,
        kv_indptr=indptr,
        kv_indices=indices,
        kv_last_page_len=last,
        page_size=PAGE,
        num_qo_heads=KV_HEADS * GROUP,
        num_kv_heads=KV_HEADS,
        head_dim=DIM,
        batch=BATCH,
        dtype="bfloat16",
        device=device,
    )
    return p, q


def test_decode_run_has_no_host_reads():
    p, q = _plan()
    with capture_guard() as guard:
        out = run(p, q)
    torch.npu.synchronize()
    summary = guard.summary()
    assert torch.isfinite(out).all()
    assert summary["syncs"] == 0, guard.explain()
    # allocations are allowed and expected (the refreshed metadata + the output)
    assert summary["allocs"] >= 0


def test_decode_survives_a_real_graph_capture_and_replays_identically():
    p, q = _plan()
    eager = run(p, q).clone()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run(p, q)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(captured, eager), (captured.float() - eager.float()).abs().max().item()


def test_manifest_declares_what_this_file_verifies():
    """The declared tier and the evidence must not drift apart."""
    from ascend_attention.spec import REGISTRY

    entry = REGISTRY.entry("paged_decode_attention")
    impls = {impl.impl: impl for impl in entry.impls}
    assert impls["tilelang.decode950"].graph == "capture_safe"
