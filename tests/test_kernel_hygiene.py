"""Hygiene rules for the migrated TileLang kernels, expressed as a ratchet.

Two rules from ``docs/plan-migrate-tileinfer.md`` §7 are only enforceable here, because the failure mode
is "the file imports fine and computes the wrong thing later":

* ``from __future__ import annotations`` breaks TileLang's parser (it evaluates annotations at JIT time),
  so no kernel file may contain it;
* a kernel that reads ``os.environ`` makes its compiled artifact depend on the *process*, which is exactly
  the property that makes a compile key unreliable.  The migrated kernels still carry such reads; the set
  is pinned here so it can only shrink.  Deleting an entry is the fix; adding one fails the test.

Both checks walk the AST rather than the text: five of these files *document* the rules in their
docstrings, and a grep would either flag the documentation or (worse) be "fixed" by deleting it.
"""

from __future__ import annotations

import ast
import pathlib

KERNELS = pathlib.Path(__file__).resolve().parents[1] / "src" / "ascend_attention" / "kernels"

#: file -> the env variable names it is still allowed to read (the ratchet, only shrinks).
ALLOWED_ENV_READS = {
    "attention/paged_decode.py": {"TILEINFER_ASCEND_TARGET"},
    "attention/paged_decode_ascend950.py": {
        "TILEINFER_POOL_CAP",
        "TILEINFER_SIMD_SOFTMAX",
        "TILEINFER_PAGE_PIPELINE",
        "TILEINFER_SW_PIPELINE",
        "TILEINFER_SPLIT_DEBUG",
        "TILEINFER_PREFILL_NOSCAN_OFF",
        "TILEINFER_PREFILL_SPLIT_MS",
    },
}


def _python_files():
    return sorted(p for p in KERNELS.rglob("*.py") if p.name != "__init__.py")


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _env_names(tree: ast.Module) -> set:
    names: set = set()
    for node in ast.walk(tree):
        # os.environ.get("NAME", ...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if (
                func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.add(node.args[0].value)
        # os.environ["NAME"]
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "environ"
                and isinstance(value.value, ast.Name)
                and value.value.id == "os"
                and isinstance(node.slice.value, str)
            ):
                names.add(node.slice.value)
    return names


def test_no_future_annotations_import_in_kernels():
    offenders = []
    for path in _python_files():
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                if any(alias.name == "annotations" for alias in node.names):
                    offenders.append(str(path.relative_to(KERNELS)))
    assert not offenders, f"TileLang cannot parse these: {offenders}"


def test_env_reads_only_in_the_pinned_set():
    observed = {}
    for path in _python_files():
        names = _env_names(_tree(path))
        if names:
            observed[str(path.relative_to(KERNELS))] = names
        relative = str(path.relative_to(KERNELS))
        allowed = ALLOWED_ENV_READS.get(relative, set())
        extra = names - allowed
        assert not extra, (
            f"{relative} reads {sorted(extra)} without an allowance; parameterise the value into the "
            "compile spec instead of reading the environment"
        )
    stale = {f for f in ALLOWED_ENV_READS if f not in observed}
    assert not stale, f"stale allowance (file no longer reads env): {sorted(stale)}"
