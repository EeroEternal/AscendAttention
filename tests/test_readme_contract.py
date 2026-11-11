"""README is a contract, not a poster: every claim it makes must be checkable in the code.

The failure mode is specific and has already happened once -- the README advertised MoE routing, expert
compute and top-p sampling while the ops for them did not exist.  A reader who trusts it writes an
integration against nothing.  So the bullets are pinned here: the op list must match the registry, and the
features that are *not* implemented must stay labelled as such.
"""

from __future__ import annotations

import pathlib
import re

from ascend_attention import capabilities

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def _text() -> str:
    return README.read_text(encoding="utf-8")


def test_every_registered_op_is_named_in_the_readme():
    names = {op["op"] for op in capabilities()["ops"]}
    text = _text()
    missing = sorted(name for name in names if name not in text)
    assert not missing, f"README does not mention: {missing}"


def test_the_readme_does_not_advertise_an_op_that_does_not_exist():
    """Any `\\`snake_case\\`` token in the ops table must be a registered op (or a known non-op noun)."""
    registered = {op["op"] for op in capabilities()["ops"]}
    # words that look like op names but are documents/files, not ops
    allowed = {"manifest_json", "graph", "requires"}
    table_rows = [line for line in _text().splitlines() if line.startswith("| `")]
    claimed = set()
    for row in table_rows:
        for token in re.findall(r"`([a-z][a-z0-9_]{3,})`", row):
            if token in allowed or "." in token:
                continue
            claimed.add(token)
    unknown = sorted(token for token in claimed if token not in registered and "_" in token)
    assert not unknown, f"README table names things that are not ops: {unknown}"


def test_unimplemented_features_are_labelled():
    """The three that were over-claimed before must carry an explicit 'not implemented' marker."""
    text = _text()
    for phrase, marker in [
        ("路由与专家计算", "未实现"),
        ("top-p", "未实现"),
        ("AscendC", "未实现"),
        ("vLLM 适配器", "未实现"),
    ]:
        assert phrase in text, f"README lost the note about {phrase!r}"
        window = text[text.index(phrase) : text.index(phrase) + 220]
        assert marker in window, f"{phrase!r} is claimed without a {marker!r} marker nearby"


def test_the_install_section_names_both_files():
    text = _text()
    assert "## 安装" in text
    assert "constraints.txt" in text
    assert "pyproject.toml" in text
