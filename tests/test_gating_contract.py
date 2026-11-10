"""``gdn_gating`` through the public spine: supports -> plan -> run, plus the contract guards.

The numerical oracle tests live in ``test_gdn_gating.py`` (migrated from TileInfer, unchanged).  This
file tests the *library* behaviour around it: that the gate answers before a plan exists, that the
plan refuses a shape it did not declare, and that ``run`` only accepts what the plan promised.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention import OpPlan, capabilities, plan, run, supports
from ascend_attention.ops.gdn import gating


def _tensors(t=5, h=16, dtype=torch.bfloat16):
    torch.manual_seed(0)
    A_log = torch.randn(h, dtype=torch.float32) * 0.3
    dt_bias = torch.randn(h, dtype=torch.float32) * 0.1
    a = torch.randn(t, h, dtype=dtype)
    b = torch.randn(t, h, dtype=dtype)
    return A_log, a, b, dt_bias


def test_supports_answers_without_building_a_plan():
    assert supports("gdn_gating", shape=(5, 16), dtype="bfloat16") is True
    assert supports("gdn_gating", shape=(5, 16), dtype="float32") is True
    assert supports("gdn_gating", shape=(5, 16, 4), dtype="bfloat16") is False
    assert supports("gdn_gating", shape=(5, 0), dtype="bfloat16") is False


def test_plan_run_matches_the_direct_call():
    A_log, a, b, dt_bias = _tensors()
    p = plan("gdn_gating", shape=tuple(a.shape), dtype="bfloat16")
    assert isinstance(p, OpPlan)
    assert p.op == "gdn_gating" and p.impl == "torch.gating"
    got_g, got_beta = run(p, A_log, a, b, dt_bias)
    ref_g, ref_beta = gating.fused_gdn_gating(A_log, a, b, dt_bias)
    torch.testing.assert_close(got_g, ref_g)
    torch.testing.assert_close(got_beta, ref_beta)


def test_plan_describe_is_jsonable():
    p = plan("gdn_gating", shape=(7, 8), dtype="float16")
    data = p.describe()
    assert data["op"] == "gdn_gating"
    assert data["shape"] == [7, 8]
    assert data["dtypes"] == ["float16"]
    assert data["has_handle"] is False
    assert data["spec_key"] == "gdn_gating/torch.gating/float16/ND/7/8"


def test_plan_rejects_a_shape_it_did_not_declare():
    with pytest.raises(ValueError, match="2-D"):
        plan("gdn_gating", shape=(2, 3, 4), dtype="bfloat16")
    p = plan("gdn_gating", shape=(5, 16), dtype="bfloat16")
    A_log, a, b, dt_bias = _tensors(t=6)
    with pytest.raises(ValueError, match="shape"):
        run(p, A_log, a, b, dt_bias)


def test_run_rejects_a_non_contiguous_tensor():
    p = plan("gdn_gating", shape=(5, 16), dtype="bfloat16")
    A_log, a, b, dt_bias = _tensors()
    transposed = a.t()  # shape (16, 5) is wrong too, so build a wrong-strided (5,16)
    bad = torch.randn(5, 32, dtype=torch.bfloat16)[:, ::2]
    assert tuple(transposed.shape) != (5, 16)
    with pytest.raises(ValueError, match="contiguous|shape"):
        run(p, A_log, bad, b, dt_bias)


def test_run_requires_a_plan_object():
    with pytest.raises(TypeError, match="OpPlan"):
        run("gdn_gating", torch.zeros(1))


def test_unknown_impl_is_a_key_error():
    with pytest.raises(KeyError, match="no impl"):
        plan("gdn_gating", impl="does.not.exist", shape=(5, 16), dtype="bfloat16")


def test_capabilities_reports_the_gate_state():
    data = capabilities("gdn_gating")
    assert data["op"] == "gdn_gating"
    assert data["baseline"] == "torch.gating"
    assert data["impls"][0]["impl"] == "torch.gating"
    assert data["impls"][0]["enabled_by_default"] is True


def test_capabilities_without_argument_is_jsonable():
    import json

    data = capabilities()
    assert data["library"] == "ascend-attention"
    assert "gdn_gating" in [op["op"] for op in data["ops"]]
    json.dumps(data, allow_nan=False)


def test_bfloat16_round_trip_is_inside_the_contract():
    """``beta`` is computed in fp32 and returned in ``b``'s dtype — the bf16 round-trip is the spec."""
    A_log, a, b, dt_bias = _tensors(t=3, h=4)
    p = plan("gdn_gating", shape=(3, 4), dtype="bfloat16")
    g, beta = run(p, A_log, a, b, dt_bias)
    assert g.dtype == torch.float32 and g.shape == (1, 3, 4)
    assert beta.dtype == torch.bfloat16 and beta.shape == (1, 3, 4)
