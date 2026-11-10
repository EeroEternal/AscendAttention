"""Device numerics for the fused GDN decode conv (``npu`` marker).

The baseline is the torch port the engine runs when no kernel exists, so this test is an A/B in the
only sense that matters here: same inputs, both implementations, output **and state** compared.  The
state comparison is not optional — the kernel shifts the conv window in place, and a wrong window
silently corrupts every later step.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention.ops.gdn import CONV_KERNEL
from ascend_attention.plan import OpPlan

from ascend_attention.utils import is_npu_available

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
]

DT = torch.bfloat16
CHANNELS, LINES, WIDTH, LINE = 4096, 8, 4, 3


def _inputs(seed=0):
    device = torch.device("npu", 0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(CHANNELS, generator=g, dtype=torch.float32).to(DT).to(device) * 0.2
    state = (torch.randn(LINES, CHANNELS, WIDTH - 1, generator=g, dtype=torch.float32) * 0.2).to(DT).to(device)
    weight = (torch.randn(CHANNELS, WIDTH, generator=g, dtype=torch.float32) * 0.1).to(DT).to(device)
    bias = (torch.randn(CHANNELS, generator=g, dtype=torch.float32) * 0.1).to(device)
    return x, state, weight, bias


def _plans():
    from ascend_attention import plan

    kwargs = dict(channels=CHANNELS, width=WIDTH, lines=LINES, dtype="bfloat16")
    return (
        plan("gdn_conv_update", impl="tilelang.conv950", **kwargs),
        plan("gdn_conv_update", impl="torch.causal_conv1d_update", **kwargs),
    )


def test_conv_kernel_matches_torch_on_output_and_state():
    pytest.importorskip("tilelang")
    p_kernel, p_torch = _plans()
    assert isinstance(p_kernel, OpPlan) and p_kernel.handle is not None
    x, state0, weight, bias = _inputs()

    state_k = state0.clone()
    state_t = state0.clone()
    out_k = p_kernel(x, state_k, weight, bias, LINE)
    out_t = p_torch(x, state_t, weight, bias, LINE)
    torch.npu.synchronize()

    scale = float(out_t.float().abs().max()) or 1.0
    err = float((out_k.float() - out_t.float()).abs().max())
    assert err <= 2e-2 * scale, f"out max_abs={err:.3e} (scale {scale:.3e})"
    # the shifted window must agree too: a wrong window corrupts the next 3 steps silently
    assert torch.equal(state_k[LINE, :, -1], x.to(DT)), "last tap of the window is not X"
    state_err = float((state_k.float() - state_t.float()).abs().max())
    assert state_err <= 2e-2 * max(scale, 1e-3), f"state max_abs={state_err:.3e}"


def test_conv_kernel_rejects_width_other_than_4():
    from ascend_attention import plan

    with pytest.raises(ValueError, match="width == 4"):
        plan(
            "gdn_conv_update",
            impl="tilelang.conv950",
            channels=CHANNELS,
            width=3,
            lines=LINES,
            dtype="bfloat16",
        )


def test_conv_kernel_is_not_default():
    """14.15x op-level, 0.995x end-to-end: measured, recorded, still opt-in."""
    assert CONV_KERNEL.enabled_by_default is False
    assert CONV_KERNEL.measured_ratio is not None
    from ascend_attention import describe

    data = describe("gdn_conv_update")
    assert data["impls"][0]["impl"] == "torch.causal_conv1d_update"
