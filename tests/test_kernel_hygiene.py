"""Hygiene rules for the migrated TileLang kernels, expressed as AST ratchets.

Three rules from ``docs/plan-migrate-tileinfer.md`` §7 are enforceable only here, because the failure
mode is "the file imports fine and computes the wrong thing later":

* ``from __future__ import annotations`` breaks TileLang's parser;
* a kernel must not read ``os.environ`` **at all** - the only seam is
  :func:`ascend_attention.utils.kernel_flag`, which centralises the leak, logs it once, and is where the
  flags will become explicit compile-spec arguments;
* the flag *names* a kernel asks for are pinned, so a new one cannot appear without a review.

The AST walk (not grep) is deliberate: these files document the rules in their docstrings, and a textual
check would either flag the documentation or be "fixed" by deleting it.
"""

from __future__ import annotations

import ast
import pathlib

KERNELS = pathlib.Path(__file__).resolve().parents[1] / "src" / "ascend_attention" / "kernels"

#: file -> the flag names it is allowed to ask the seam for (the ratchet, only shrinks or moves).
ALLOWED_FLAGS = {
    "attention/paged_decode.py": {"ASCEND_TARGET"},
    "attention/paged_decode_ascend950.py": {
        "POOL_CAP",
        "SIMD_SOFTMAX",
        "PAGE_PIPELINE",
        "SW_PIPELINE",
        "SPLIT_DEBUG",
        "PREFILL_NOSCAN_OFF",
        "PREFILL_SPLIT_MS",
    },
}


def _python_files():
    return sorted(p for p in KERNELS.rglob("*.py") if p.name != "__init__.py")


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_no_future_annotations_import_in_kernels():
    offenders = []
    for path in _python_files():
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                if any(alias.name == "annotations" for alias in node.names):
                    offenders.append(str(path.relative_to(KERNELS)))
    assert not offenders, f"TileLang cannot parse these: {offenders}"


def test_no_direct_environment_reads_in_kernels():
    """``os.environ`` must not appear; :func:`ascend_attention.utils.kernel_flag` is the one seam."""
    offenders = []
    for path in _python_files():
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                offenders.append(f"{path.relative_to(KERNELS)}:{node.lineno}")
    assert not offenders, (
        f"direct environment reads: {offenders}; call utils.kernel_flag(...) instead"
    )


def test_flag_names_are_pinned():
    observed = {}
    for path in _python_files():
        names = set()
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("kernel_flag", "kernel_flag_bool"):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    names.add(node.args[0].value)
        relative = str(path.relative_to(KERNELS))
        allowed = ALLOWED_FLAGS.get(relative, set())
        extra = names - allowed
        assert not extra, f"{relative} asks for unpinned flags {sorted(extra)}"
        if names:
            observed[relative] = names
    stale = {f for f in ALLOWED_FLAGS if f not in observed}
    assert not stale, f"stale flag allowance: {sorted(stale)}"
