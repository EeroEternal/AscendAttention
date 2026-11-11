"""The packaging metadata is part of the contract, so it gets tested like code.

Three failure modes this catches, all of them silent in normal use:

* `dependencies` drifting into an exact pin - the library must stay importable on a machine that is not
  ours (torch on CPU is the whole point of the "no NPU, no toolchain" contract);
* `constraints.txt` disagreeing with what the extras claim, so "install the pinned set" installs something
  else;
* a package-data or metadata key going missing while the wheel still builds.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

tomllib = pytest.importorskip("tomllib") if sys.version_info >= (3, 11) else None


def _pyproject():
    if tomllib is None:  # pragma: no cover - py3.10
        pytest.skip("tomllib is 3.11+")
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_base_dependency_is_a_floor_not_a_pin():
    deps = _pyproject()["project"]["dependencies"]
    assert deps, "a library with no declared dependency cannot be installed"
    for dep in deps:
        assert "==" not in dep, f"{dep!r}: exact pins belong in constraints.txt, not dependencies"


def test_constraints_carry_the_reference_versions():
    text = (ROOT / "constraints.txt").read_text(encoding="utf-8")
    pinned = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", text, re.MULTILINE))
    # the three that change codegen and therefore invalidate a measured ratio
    assert pinned["torch"].startswith("2.13"), pinned
    assert pinned["torch-npu"] == "2.13.0rc1", pinned
    assert pinned["tilelang"] == "0.1.15", pinned


def test_the_pinned_extra_agrees_with_constraints():
    text = (ROOT / "constraints.txt").read_text(encoding="utf-8")
    pinned = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", text, re.MULTILINE))
    extra = _pyproject()["project"]["optional-dependencies"]["pinned"]
    for dep in extra:
        name, _, version = dep.partition("==")
        assert version, dep
        assert pinned[name] == version, f"{name}: extra says {version}, constraints say {pinned[name]}"


def test_extras_exist_for_the_two_optional_toolchains():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "npu" in extras and "tilelang" in extras and "dev" in extras
    joined = " ".join(extras["npu"] + extras["tilelang"])
    assert "torch_npu" in joined or "torch-npu" in joined
    assert "tilelang" in joined
