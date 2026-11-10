"""The declarative spec is the constitution: every rule it states must fail loudly when broken."""

from __future__ import annotations

import pytest

from ascend_attention.spec import (
    ImplSpec,
    ShapeSpec,
    dim_in,
    dim_min,
    dim_mul,
)


def _impl(**kw):
    base = dict(
        op="demo",
        impl="torch.demo",
        form="torch",
        shape=ShapeSpec(ndim=2, dims=(None, dim_min(1)), dtypes=frozenset({"bfloat16"})),
        measured_ratio=1.0,
        bench="bench/bench_demo.py",
    )
    base.update(kw)
    return ImplSpec(**base)


# --------------------------------------------------------------------- registration assertions


def test_enabled_without_ratio_is_rejected():
    with pytest.raises(ValueError, match="measured_ratio"):
        _impl(enabled_by_default=True, measured_ratio=None, bench=None)


def test_enabled_without_bench_is_rejected():
    with pytest.raises(ValueError, match="bench"):
        _impl(enabled_by_default=True, bench=None)


def test_unmeasured_and_disabled_is_allowed():
    impl = _impl(enabled_by_default=False, measured_ratio=None, bench=None)
    assert impl.advertised is False


def test_unknown_form_is_rejected():
    with pytest.raises(ValueError, match="form"):
        _impl(form="magic")


def test_unknown_graph_tier_is_rejected():
    with pytest.raises(ValueError, match="graph"):
        _impl(graph="maybe")


def test_unknown_plan_kind_is_rejected():
    with pytest.raises(ValueError, match="plan_kind"):
        _impl(plan_kind="sometimes")


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="device"):
        _impl(device="ascend999")


def test_empty_kv_layouts_is_rejected():
    with pytest.raises(ValueError, match="kv_layouts"):
        _impl(kv_layouts=frozenset())


def test_unknown_kv_layout_is_rejected():
    with pytest.raises(ValueError, match="kv_layouts"):
        _impl(kv_layouts=frozenset({"NHD"}))  # NHD is an op parameter, not a memory format


def test_non_positive_ratio_is_rejected():
    with pytest.raises(ValueError, match="> 0"):
        _impl(measured_ratio=0.0)


# --------------------------------------------------------------------- shape language


@pytest.mark.parametrize(
    "constraint,value,ok",
    [
        (dim_in(128, 256), 128, True),
        (dim_in(128, 256), 192, False),
        (dim_mul(16), 32, True),
        (dim_mul(16), 33, False),
        (dim_min(2), 1, False),
        (dim_min(2), 2, True),
        (7, 7, True),
        (7, 8, False),
        (None, 12345, True),
    ],
)
def test_dimension_language(constraint, value, ok):
    spec = ShapeSpec(ndim=1, dims=(constraint,), dtypes=frozenset({"bfloat16"}))
    assert spec.matches((value,), "bfloat16") is ok


def test_ndim_mismatch_fails_closed():
    spec = ShapeSpec(ndim=2, dims=(None, None), dtypes=frozenset({"bfloat16"}))
    assert spec.matches((4, 8, 16), "bfloat16") is False


def test_dtype_aliases_normalise():
    spec = ShapeSpec(ndim=1, dims=(None,), dtypes=frozenset({"float16"}))
    assert spec.matches((8,), "fp16") is True
    assert spec.matches((8,), "half") is True
    assert spec.matches((8,), "bfloat16") is False


def test_contiguity_is_part_of_the_contract():
    spec = ShapeSpec(ndim=1, dims=(None,), dtypes=frozenset({"bfloat16"}), contiguous=True)
    assert spec.matches((8,), "bfloat16", contiguous=False) is False


def test_dimensions_must_match_ndim():
    with pytest.raises(ValueError, match="dimension constraints"):
        ShapeSpec(ndim=2, dims=(None,), dtypes=frozenset({"bfloat16"}))


def test_empty_dtype_set_is_rejected():
    with pytest.raises(ValueError, match="dtype"):
        ShapeSpec(ndim=1, dims=(None,), dtypes=frozenset())


# --------------------------------------------------------------------- advertising


def test_advertised_needs_ratio_and_bench():
    assert _impl().advertised is True
    assert _impl(bench=None, enabled_by_default=False).advertised is False


def test_capture_graph_rejects_non_capture_safe():
    impl = _impl(graph="eager_only")
    assert impl.accepts_shape(graph=True) is False
    assert impl.accepts_shape(graph=False) is True


def test_accepts_shape_honours_extra_predicates():
    impl = _impl(shapes={"head_dim": dim_in(128, 256)})
    assert impl.accepts_shape(head_dim=128) is True
    assert impl.accepts_shape(head_dim=192) is False
    assert impl.accepts_shape() is True  # unconstrained query is not a rejection
