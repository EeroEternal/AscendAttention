"""The two artefacts must answer their own question and never each other's.

``manifest.json`` is the *usable* subset; ``capabilities()`` is the *full inventory*.  A test that only
looked at one of them would let the other drift: either the platform is told it can call something it
cannot, or it cannot discover an implementation that exists (which is the current state of decode, and is
a decision rather than an omission - see ``docs/contracts/exposure.md``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import pathlib

from ascend_attention import capabilities
from ascend_attention.manifest import manifest
from ascend_attention.spec import REGISTRY

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def test_manifest_is_a_subset_of_the_inventory():
    caps = capabilities()
    inventory = {(op["op"], impl["impl"]) for op in caps["ops"] for impl in op["impls"]}
    for op in manifest()["ops"]:
        assert (op["name"], op["impl"]) in inventory, (op["name"], op["impl"])


def test_every_registered_implementation_appears_in_the_inventory():
    caps = capabilities()
    seen = {(op["op"], impl["impl"]) for op in caps["ops"] for impl in op["impls"]}
    for name in REGISTRY.names():
        for impl in REGISTRY.entry(name).impls:
            assert (name, impl.impl) in seen, (name, impl.impl)


def test_disabled_implementations_are_visible_with_a_reason():
    """Decode is the canonical case: present, device-verified, capture-safe, and deliberately not enabled."""
    caps = capabilities()
    op = next(o for o in caps["ops"] if o["op"] == "paged_decode_attention")
    impls = {i["impl"]: i for i in op["impls"]}
    kernel = impls["tilelang.decode950"]
    assert kernel["enabled_by_default"] is False
    assert kernel["graph"] == "capture_safe"
    assert kernel["measured_ratio"] is None
    assert kernel["note"], "a disabled implementation must say why"
    # and it is genuinely absent from the manifest, because no impl of that op is enabled
    assert "paged_decode_attention" not in {o["name"] for o in manifest()["ops"]}


def test_the_cli_prints_both_artefacts():
    caps = subprocess.run(
        [sys.executable, "-m", "ascend_attention", "--capabilities", "--compact"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
        timeout=300,
    )
    assert caps.returncode == 0, caps.stderr
    parsed = json.loads(caps.stdout)
    assert parsed["library"] == "ascend-attention"
    assert any(op["op"] == "paged_decode_attention" for op in parsed["ops"])
