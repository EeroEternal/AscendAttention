"""The four gates, in order, and the promise that a gate failure is *cheap* and *visible*."""

from __future__ import annotations

import pytest

from ascend_attention import dispatch
from ascend_attention.spec import ImplSpec, Registry, ShapeSpec, dim_min


def _registry(*specs):
    reg = Registry()
    reg.op("demo", "a demo op")
    for spec in specs:
        reg.register(spec)
    return reg.entry("demo")


BASE = dict(
    op="demo",
    form="torch",
    shape=ShapeSpec(ndim=2, dims=(None, dim_min(1)), dtypes=frozenset({"bfloat16"})),
    kv_layouts=frozenset({"ND"}),
    enabled_by_default=True,
    measured_ratio=1.0,
    bench="bench/bench_demo.py",
)


def _baseline():
    return ImplSpec(impl="torch.demo", **BASE)


def _candidate(ratio=1.42, **kw):
    data = dict(BASE)
    data.update(
        impl="kernel.demo",
        form="tilelang",
        measured_ratio=ratio,
        bench="bench/bench_demo.py",
    )
    data.update(kw)
    return ImplSpec(**data)


def test_candidate_is_preferred_when_all_gates_pass(monkeypatch):
    monkeypatch.delenv(dispatch.ENV_CANDIDATES, raising=False)
    entry = _registry(_candidate(), _baseline())
    decision = dispatch.select(entry, shape=(4, 8), dtype="bfloat16")
    assert decision.impl.impl == "kernel.demo"
    assert decision.reason == "eligible"


def test_unset_switch_keeps_candidates_on():
    assert dispatch.candidates_enabled() is True


@pytest.mark.parametrize("value", ["off", "0", "false", "OFF", " no "])
def test_kill_switch_disarms_candidates(monkeypatch, value):
    monkeypatch.setenv(dispatch.ENV_CANDIDATES, value)
    entry = _registry(_candidate(), _baseline())
    decision = dispatch.select(entry, shape=(4, 8), dtype="bfloat16")
    assert decision.impl.impl == "torch.demo"
    assert any(reason == "ASCEND_ATTN_CANDIDATES=off" for _, reason in decision.rejected)


def test_measured_below_parity_is_ineligible_even_if_enabled(monkeypatch):
    """The 0.86x grouped-GEMM lesson: a recorded loss never becomes eligible."""
    monkeypatch.delenv(dispatch.ENV_CANDIDATES, raising=False)
    entry = _registry(_candidate(ratio=0.86), _baseline())
    decision = dispatch.select(entry, shape=(4, 8), dtype="bfloat16")
    assert decision.impl.impl == "torch.demo"
    assert any("0.86x" in reason for _, reason in decision.rejected)


def test_shape_mismatch_falls_back_to_baseline(monkeypatch):
    monkeypatch.delenv(dispatch.ENV_CANDIDATES, raising=False)
    entry = _registry(_candidate(), _baseline())
    decision = dispatch.select(entry, shape=(4,), dtype="bfloat16")  # wrong ndim
    assert decision.impl.impl == "torch.demo"
    assert any("does not match" in reason for _, reason in decision.rejected)


def test_dtype_mismatch_falls_back_to_baseline(monkeypatch):
    monkeypatch.delenv(dispatch.ENV_CANDIDATES, raising=False)
    entry = _registry(_candidate(), _baseline())
    decision = dispatch.select(entry, shape=(4, 8), dtype="float32")
    assert decision.impl.impl == "torch.demo"


def test_capture_request_rejects_eager_only_candidate(monkeypatch):
    monkeypatch.delenv(dispatch.ENV_CANDIDATES, raising=False)
    entry = _registry(_candidate(graph="eager_only"), _baseline())
    decision = dispatch.select(entry, shape=(4, 8), dtype="bfloat16", graph=True)
    assert decision.impl.impl == "torch.demo"


def test_pin_overrides_gates(monkeypatch):
    monkeypatch.setenv(dispatch.ENV_CANDIDATES, "off")
    monkeypatch.setenv(dispatch.ENV_IMPL, "demo=kernel.demo")
    entry = _registry(_candidate(), _baseline())
    decision = dispatch.select(entry, shape=(4, 8), dtype="bfloat16")
    assert decision.impl.impl == "kernel.demo"
    assert "pinned" in decision.reason


def test_malformed_pin_is_ignored_with_a_warning(monkeypatch):
    monkeypatch.setenv(dispatch.ENV_IMPL, "demo")
    assert dispatch.pin_impl("demo") is None
    monkeypatch.setenv(dispatch.ENV_IMPL, "demo=kernel.demo, other=torch.x")
    assert dispatch.pin_impl("demo") == "kernel.demo"
    assert dispatch.pin_impl("other") == "torch.x"


def test_no_baseline_and_no_candidate_is_explicit():
    entry = _registry(_candidate(enabled_by_default=False, measured_ratio=None, bench=None))
    decision = dispatch.select(entry, shape=(4, 8), dtype="bfloat16")
    assert decision.impl is None
    assert "no torch baseline" in decision.reason


def test_decision_is_jsonable(monkeypatch):
    monkeypatch.delenv(dispatch.ENV_CANDIDATES, raising=False)
    entry = _registry(_candidate(), _baseline())
    data = dispatch.select(entry, shape=(4, 8), dtype="bfloat16").as_dict()
    assert data["impl"] == "kernel.demo"
    assert isinstance(data["rejected"], list)


def test_describe_reports_switch_and_status(monkeypatch):
    monkeypatch.delenv(dispatch.ENV_CANDIDATES, raising=False)
    from ascend_attention.spec import REGISTRY

    data = dispatch.describe("gdn_gating")
    assert data["op"] == "gdn_gating"
    assert data["switch"]["enabled"] is True
    assert data["baseline"] == "torch.gating"
    assert data["impls"][0]["advertised"] is True
