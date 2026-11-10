"""The kill switch must mean the same thing here and in TileInfer.

The two libraries live in one venv, so an operator writes one word in a launcher and expects both to
obey.  They did not: TileInfer treated ``=none``/``=disable``/``=disabled`` as off, this repo treated them
as **on** -- a typo-class divergence that silently keeps candidates live in one process and not the other.
Both sides are now pinned to the same set, and this test reads TileInfer's parser directly rather than
duplicating a copy of the list (a copy would drift and still pass).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from ascend_attention.dispatch import candidates_enabled

TILEINFER_SRC = pathlib.Path(__file__).resolve().parents[2] / "TileInfer"

SPELLINGS = ["", "0", "1", "off", "on", "false", "true", "no", "yes", "none", "disable", "disabled",
             "OFF", " Disable ", "garbage", "all", "enable"]


def _tileinfer_enabled(monkeypatch, raw: str) -> bool:
    monkeypatch.setenv("TILEINFER_OPS_CANDIDATES", raw)
    if str(TILEINFER_SRC) not in sys.path:
        sys.path.insert(0, str(TILEINFER_SRC))
    from tileinfer.vllm.ops.candidates import candidates_enabled as theirs

    return bool(theirs(spec=raw))


@pytest.mark.parametrize("raw", SPELLINGS)
def test_both_libraries_read_the_switch_the_same_way(monkeypatch, raw):
    if not TILEINFER_SRC.exists():
        pytest.skip("TileInfer checkout not present")
    monkeypatch.setenv("ASCEND_ATTN_CANDIDATES", raw)
    ours = candidates_enabled()
    theirs = _tileinfer_enabled(monkeypatch, raw)
    assert ours == theirs, f"{raw!r}: this repo says {ours}, TileInfer says {theirs}"


def test_garbage_disarms_rather_than_leaving_candidates_live(monkeypatch, capsys):
    """The direction of the fail-safe is the part that matters, so it gets its own assertion.

    "On by default" and "off on a typo" are compatible: the first is about an *unset* variable, the second
    about a value nobody meant.  Getting this backwards is how a typo silently ships an unmeasured kernel.
    """
    monkeypatch.setenv("ASCEND_ATTN_CANDIDATES", "ture")
    assert candidates_enabled() is False
