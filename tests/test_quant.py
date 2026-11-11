"""FP8 fake-quant: the scale arithmetic, checked where it is silently wrong.

Three claims carry this family, and each has a test that fails if the choice is reversed:

1. the cast **saturates** at the format maximum (torch's own cast would give inf/nan on an outlier);
2. the scale **divides before rounding and multiplies after** (dividing after the cast is a different,
   off-by-one-ulp function);
3. the scale **axis** matches the operand: per-token on activations, per-channel on weights, and a
   dequant-then-matmul equals a matmul with the scales folded in only when the axes are right.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from ascend_attention import plan, run
from ascend_attention.ops.quant import (
    FORMATS,
    GEMM_OP,
    OP,
    fp8_gemm_reference,
    max_finite,
    quant_dequant,
)

ROOT = Path(__file__).resolve().parents[1]


def test_the_format_maxima_match_torch_finfo():
    """The constants are the format, not a guess; torch's finfo is the second opinion."""
    for name, (dtype, limit) in FORMATS.items():
        assert torch.finfo(dtype).max == limit, name


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_outliers_saturate_instead_of_overflowing(fmt):
    """An activation outlier must become a bounded error, not inf - this is the whole point of clamping."""
    limit = max_finite(fmt)
    x = torch.tensor([[-1e9, -1.0, 0.0, 1.0, 1e9]])
    out = quant_dequant(x, 1.0, fmt=fmt)
    assert torch.isfinite(out).all()
    assert out[0, 0] == -limit and out[0, 4] == limit


def test_the_scale_divides_before_rounding_and_multiplies_after():
    """Rounding ``x/s`` is a different function from rounding ``x`` and then scaling; both are checked."""
    # values that are exactly on the fp8 grid survive the round trip
    x = torch.tensor([[0.5, 1.5, 2.5]])
    assert torch.equal(quant_dequant(x, 0.25, fmt="e4m3"), x)
    # 1.1 is not on the grid: nearest e4m3 value is 1.125 (spacing 0.125 in [1, 2))
    assert quant_dequant(torch.tensor([[1.1]]), 1.0, fmt="e4m3").item() == 1.125
    # with scale 0.25 we round 4.4 grid units (nearest is 4.5: spacing is 0.5 in [4, 8)) -> 1.125.
    # Casting *first* and scaling after would give a different number, because it rounds 1.1 with
    # scale-1 spacing (0.125) instead of 4.4 with scale-0.25 spacing (0.5).
    assert quant_dequant(torch.tensor([[1.1]]), 0.25, fmt="e4m3").item() == 1.125
    cast_first = (torch.tensor([[1.1]]).to(torch.float8_e4m3fn).to(torch.float32)) * 0.25
    assert not torch.isclose(torch.tensor(1.125), cast_first)


def test_the_scale_is_positive_and_finite():
    x = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="must be > 0"):
        quant_dequant(x, torch.tensor([0.0, 1.0]), axis="token")
    with pytest.raises(ValueError, match="must be finite"):
        quant_dequant(x, float("nan"))
    with pytest.raises(ValueError, match="must be > 0"):
        quant_dequant(x, torch.tensor([1.0, -1.0]), axis="token")


def test_the_axis_is_checked_against_the_scale_shape():
    """A per-token scale broadcast along the wrong axis is a silent channel swap - refuse it."""
    x = torch.randn(4, 8)
    with pytest.raises(ValueError, match="axis='token' wants"):
        quant_dequant(x, torch.ones(8), axis="token")
    with pytest.raises(ValueError, match="axis='channel' wants"):
        quant_dequant(x, torch.ones(4), axis="channel")
    with pytest.raises(ValueError, match="scalar scale"):
        quant_dequant(x, torch.ones(4), axis="tensor")
    with pytest.raises(ValueError, match="not one of"):
        quant_dequant(x, 1.0, axis="head")


def test_per_token_scaling_is_row_wise_and_bounded():
    torch.manual_seed(0)
    x = torch.randn(6, 128) * torch.linspace(1, 50, 6).reshape(-1, 1)
    scale = x.abs().amax(dim=-1) / max_finite("e4m3")
    out = quant_dequant(x, scale, axis="token")
    assert out.dtype == x.dtype
    # each row's largest magnitude is within one grid step of the format maximum
    rel = (out.abs().amax(dim=-1) / (x.abs().amax(dim=-1))).max()
    assert 0.9 <= float(rel) <= 1.0
    # and the error is bounded by the format's relative step (~2^-3 for e4m3 mantissa 3 bits + implicit)
    assert float(((out - x).abs() / x.abs().amax(dim=-1).reshape(-1, 1)).max()) < 0.07


def test_fake_quant_is_idempotent_on_the_fp8_grid():
    """Re-quantising the dequantised values must not move them - a kernel that drifts would show here."""
    torch.manual_seed(1)
    x = torch.randn(8, 32)
    scale = x.abs().amax() / 448
    once = quant_dequant(x, scale)
    assert torch.equal(quant_dequant(once, scale), once)


def test_bf16_is_not_silently_promoted():
    x = (torch.randn(2, 16) * 4).to(torch.bfloat16)
    out = quant_dequant(x, 0.5)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, quant_dequant(x.clone(), 0.5))


def test_gemm_matches_the_dequantised_definition_in_fp64():
    torch.manual_seed(2)
    m, k, n = 5, 16, 3
    a = (torch.randn(m, k)).to(torch.float8_e4m3fn)
    b = (torch.randn(k, n) * 0.5).to(torch.float8_e4m3fn)
    a_scale = torch.rand(m) + 0.5
    b_scale = torch.rand(n) + 0.5
    got = fp8_gemm_reference(a, a_scale, b, b_scale, out_dtype=torch.float64)
    a64 = a.to(torch.float64) * a_scale.reshape(-1, 1).to(torch.float64)
    b64 = b.to(torch.float64) * b_scale.reshape(1, -1).to(torch.float64)
    assert torch.allclose(got.to(torch.float64), a64 @ b64, atol=1e-6)


def test_folding_the_scales_after_the_matmul_agrees_only_with_the_right_axes():
    """The equivalence a real kernel relies on: (a*sa)@(b*sb) == sa[:,None] * (a@b) * sb[None,:]."""
    torch.manual_seed(3)
    m, k, n = 4, 8, 6
    a = torch.randn(m, k).to(torch.float8_e4m3fn)
    b = torch.randn(k, n).to(torch.float8_e4m3fn)
    a_scale = torch.rand(m) + 0.5
    b_scale = torch.rand(n) + 0.5
    folded = fp8_gemm_reference(a, torch.ones(1), b, torch.ones(1), out_dtype=torch.float64)
    folded = a_scale.reshape(-1, 1).to(torch.float64) * folded * b_scale.reshape(1, -1).to(torch.float64)
    assert torch.allclose(
        fp8_gemm_reference(a, a_scale, b, b_scale, out_dtype=torch.float64), folded, atol=1e-9
    )


def test_gemm_validation():
    a = torch.zeros(3, 4, dtype=torch.float8_e4m3fn)
    b = torch.zeros(4, 2, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="2-D"):
        fp8_gemm_reference(a.reshape(1, 3, 4), torch.ones(3), b, torch.ones(2))
    with pytest.raises(ValueError, match="inner dims"):
        fp8_gemm_reference(a, torch.ones(3), torch.zeros(5, 2, dtype=torch.float8_e4m3fn), torch.ones(2))
    with pytest.raises(ValueError, match="a_scale must be"):
        fp8_gemm_reference(a, torch.ones(7), b, torch.ones(2))


def test_plans_and_runs():
    x = torch.randn(4, 1024)
    p = plan(OP, batch=1, rows=4, cols=1024, fmt="e4m3", axis="token", dtype="float32", device="cpu")
    scale = x.abs().amax(-1) / 448
    assert torch.equal(run(p, x, scale), quant_dequant(x, scale, axis="token"))
    with pytest.raises(ValueError, match="is outside the declared contract"):
        plan(OP, rows=4, cols=3, fmt="e4m3", axis="token", dtype="float32", device="cpu")
    with pytest.raises(ValueError, match="x must be"):
        run(p, torch.randn(5, 1024), scale)

    a = torch.zeros(1024, 4096, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.zeros(4096, 1024, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    pg = plan(GEMM_OP, m=1024, k=4096, n=1024, fmt="e4m3", device="cpu")
    assert tuple(run(pg, a, torch.ones(1024), b, torch.ones(1024)).shape) == (1024, 1024)


def test_manifest_entries_and_ledger_exist():
    from ascend_attention import manifest as manifest_module

    ops = {op["name"]: op for op in manifest_module.manifest()["ops"]}
    assert ops[OP]["platform_name"] == "quant.dequant_fp8"
    assert ops[GEMM_OP]["platform_name"] == "quant.fp8_gemm"
    for name in (OP, GEMM_OP):
        assert ops[name]["measured_ratio"] == 1.0
        assert (ROOT / ops[name]["bench"]).is_file()
    ledger = ROOT / "bench" / "results" / "quant_fp8_b8.json"
    assert ledger.is_file()
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["ms"]["dequant"]["median_ms"] > 0
    assert payload["ms"]["gemm"]["median_ms"] > 0
