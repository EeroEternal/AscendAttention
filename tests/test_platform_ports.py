"""The symbols ``AscendPlatform.contracts.ports`` resolves by name.

The platform never imports this package directly: it looks up a *logical* module under the operator
root (``attention`` -> ``api``, ``metadata``, ``plan``, ``decode_kernel``, ``testing``, ``ops``) and
then a symbol inside it.  Those names are a contract; renaming a module or a function here breaks the
platform at start-up, on a machine where nobody is looking at this repo.

This test pins the exact set (taken from ``AscendPlatform/src/ascend_platform/contracts/ports.py``
``_LAYOUTS["ascend_attention"]``), so a rename fails here first.
"""

from __future__ import annotations

import importlib

import pytest

#: logical name -> (dotted module under the package, required symbols)
PORTS = {
    "attention": ("api", ("BatchAttention",)),
    "metadata": ("metadata", ("RaggedMetadata", "PageTable")),
    "plan": ("plan", ("TileSchedule", "plan_query_tiles_from_host", "AttentionPlan", "OpPlan")),
    "decode_kernel": (
        "kernels.attention.paged_decode_ascend950",
        ("forward_prefill", "_pack_indices", "forward", "declared_pool_cap", "pool_view"),
    ),
    "testing": ("testing.moe_reference", ("reference_moe",)),
    "ops": ("ops", ("gdn", "fla", "attention")),
}


@pytest.mark.parametrize("logical", sorted(PORTS))
def test_platform_visible_symbols_exist(logical):
    module_name, symbols = PORTS[logical]
    module = importlib.import_module(f"ascend_attention.{module_name}")
    for symbol in symbols:
        assert hasattr(module, symbol), f"ascend_attention.{module_name} lost {symbol!r}"


def test_ops_submodules_are_importable_by_name():
    """``ports.ops_module("moe")`` does ``import_module(f"{ops.__name__}.{name}")``."""
    import ascend_attention.ops as ops

    for name in ("gdn", "fla"):
        assert importlib.import_module(f"{ops.__name__}.{name}") is not None


def test_platform_check_range_accepts_our_declared_range():
    """The manifest says ``requires_platform = ">=0.1,<0.2"``; the platform must agree it is valid."""
    import pathlib
    import sys

    platform_src = pathlib.Path(__file__).resolve().parents[2] / "AscendPlatform" / "src"
    if not platform_src.exists():
        pytest.skip("AscendPlatform checkout not present")
    if str(platform_src) not in sys.path:
        sys.path.insert(0, str(platform_src))
    try:
        from ascend_platform.contracts.version import VersionRange
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ascend_platform not importable: {exc}")

    from ascend_attention.manifest import REQUIRES_PLATFORM

    parsed = VersionRange.parse(REQUIRES_PLATFORM)
    assert parsed.contains("0.1.0")
    assert not parsed.contains("0.2.0")
