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
    """Every capability this README talks about must say whether it exists.

    Two rounds of history are pinned here.  The first three were over-claimed once (routing/expert
    compute, top-p sampling, the AscendC form) and the labels are what stopped the repeat.  The last
    three come from the boundary review: MLA and quantisation were filed under "platform" (they are
    operators, and they are not written yet), and the engine adapter was described as "not done
    because it would break the zero-vllm rule" -- it is not ours to write at all, so the note has to
    say *whose* it is.  MLA and quantisation have since been dropped outright rather than left half
    written (the target model needs neither), so those two entries now require the reason, not a
    "missing kernel" apology.  A README that drops any of these five sentences is advertising again.
    """
    text = _text()
    for phrase, marker in [
        ("MoE 的内核化", "没有加速内核"),
        ("AscendC", "未实现"),
        ("引擎适配不在本仓", "平台"),
        ("MLA", "不做"),
        ("量化（fp8/fp4）", "不做"),
        ("供给契约两仓都已落地", "实测 10 绑定"),
    ]:
        assert phrase in text, f"README lost the note about {phrase!r}"
        window = text[text.index(phrase) : text.index(phrase) + 320]
        assert marker in window, f"{phrase!r} is claimed without a {marker!r} marker nearby"


def test_the_manifest_count_in_the_readme_matches_the_file():
    """A hand-written count is a claim; the manifest is the fact. This is the cheapest way to keep them
    equal - the number drifted twice while this batch was being written."""
    import json
    import pathlib
    import re

    text = _text()
    match = re.search(r"只收\"有实测比值且默认启用\"的实现（(\d+) 条）", text)
    assert match, "README lost the sentence that states the manifest size"
    manifest = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "src/ascend_attention/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert int(match.group(1)) == len(manifest["ops"]), (
        f"README says {match.group(1)} manifest entries, the file has {len(manifest['ops'])}"
    )


def test_the_install_section_names_both_files():
    text = _text()
    assert "## 安装" in text
    assert "constraints.txt" in text
    assert "pyproject.toml" in text
