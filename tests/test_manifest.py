"""The manifest: committed snapshot, cross-repo parseability, and the "no ratio, no mainline" rule.

The cross-repo half is the important one.  ``AscendPlatform`` reads this file at start-up and
refuses to boot on a malformed entry; a test here means a schema drift is caught in *this* repo's CI
instead of on the 102 box.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from ascend_attention.manifest import (
    LIBRARY_NAME,
    MANIFEST_SCHEMA_VERSION,
    REQUIRES_PLATFORM,
    dumps,
    manifest as build_manifest,
)


def test_committed_snapshot_is_up_to_date():
    """Regenerate with ``python -m ascend_attention``; CI fails if the diff is non-empty."""
    committed = pathlib.Path(__file__).resolve().parents[1] / "src" / "ascend_attention" / "manifest.json"
    assert committed.exists(), "manifest.json is missing; run python -m ascend_attention"
    assert json.loads(committed.read_text(encoding="utf-8")) == json.loads(dumps()), (
        "manifest.json is stale; run python -m ascend_attention"
    )


def test_manifest_header_declares_platform_range():
    data = build_manifest()
    assert data["schema"] == MANIFEST_SCHEMA_VERSION
    assert data["library"]["name"] == LIBRARY_NAME
    assert data["library"]["version"]
    assert data["library"]["requires_platform"] == REQUIRES_PLATFORM


def test_every_advertised_op_has_ratio_and_bench():
    for op in build_manifest()["ops"]:
        assert isinstance(op["graph_capturable"], bool), op["name"]
        assert op["measured_ratio"] > 0, op["name"]
        assert op["bench"], op["name"]
        assert op["dtypes"], op["name"]
        assert op["kv_layouts"], op["name"]


def test_unmeasured_implementations_never_reach_the_manifest():
    """``ops.attention.REFERENCE`` is registered but must stay invisible to the platform."""
    names = {op["name"] for op in build_manifest()["ops"]}
    assert "paged_decode_attention" not in names
    from ascend_attention.ops.attention import REFERENCE

    assert REFERENCE.advertised is False


def test_manifest_is_pure_json():
    text = dumps()
    assert json.loads(text) == build_manifest()
    # no callables, no tensors, no reprs that only round-trip in-process
    json.dumps(build_manifest(), allow_nan=False)


def _platform_manifest_module():
    platform_src = pathlib.Path(__file__).resolve().parents[2] / "AscendPlatform" / "src"
    if not platform_src.exists():
        pytest.skip("AscendPlatform checkout not present")
    if str(platform_src) not in sys.path:
        sys.path.insert(0, str(platform_src))
    try:
        from ascend_platform.contracts import manifest as platform_manifest
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ascend_platform not importable: {exc}")
    return platform_manifest


def test_platform_parser_accepts_our_manifest():
    platform_manifest = _platform_manifest_module()
    parsed = platform_manifest.parse_manifest(json.loads(dumps()))
    assert parsed.library == LIBRARY_NAME
    assert parsed.version
    assert "gdn_gating" in parsed.names()
    entry = parsed.op("gdn_gating")
    assert entry.measured_ratio == 1.0
    assert entry.graph_capturable is False
    assert entry.dtypes == ("bfloat16", "float16", "float32")
    assert entry.kv_layouts == ("ND",)
    assert entry.bench == "bench/bench_gdn_gating.py"


def test_platform_rejects_a_manifest_we_would_never_emit():
    platform_manifest = _platform_manifest_module()
    broken = json.loads(dumps())
    broken["ops"][0].pop("measured_ratio")
    with pytest.raises(platform_manifest.ManifestError, match="measured_ratio"):
        platform_manifest.parse_manifest(broken)
