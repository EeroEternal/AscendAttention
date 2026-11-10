"""Dependency-direction guards: the library must stay engine-free and importable on a bare box.

Two checks, because they catch different mistakes:

* AST scan — a module-level ``import vllm`` would make the package unimportable in the platform's
  start-up path, and a lazy one inside a function is still a leak (the library must not know the
  engine at all);
* subprocess import — the engine and the TileLang toolchain must not be dragged in by
  ``import ascend_attention``.  (``torch_npu`` is deliberately *not* asserted here: this torch build
  imports it on its own, which is an environment property, not ours — the AST check above already
  guarantees we never import it at module scope.)
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ascend_attention"

FORBIDDEN_PREFIXES = ("vllm", "vllm_ascend", "torch_npu", "tilelang", "ascend_platform")
#: Importing these at module scope would make `import ascend_attention` device/host dependent.
LAZY_ONLY = ("torch_npu", "tilelang")


def _python_files():
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _module_names(tree):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def _toplevel_names(tree):
    """Only imports executed at module import time (not inside functions/methods)."""
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
        elif isinstance(node, ast.If):  # import inside `if TYPE_CHECKING:` / try blocks at top
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import):
                    names.extend(alias.name for alias in inner.names)
                elif isinstance(inner, ast.ImportFrom) and inner.module:
                    names.append(inner.module)
    return names


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_engine_or_platform_imports(path):
    names = _module_names(ast.parse(path.read_text(encoding="utf-8")))
    for name in names:
        assert not name.startswith(("vllm", "vllm_ascend", "ascend_platform")), (
            f"{path.name} imports {name!r}: the dependency direction is platform -> operators"
        )


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_optional_dependencies_are_imported_lazily(path):
    for name in _toplevel_names(ast.parse(path.read_text(encoding="utf-8"))):
        assert not name.startswith(LAZY_ONLY), (
            f"{path.name} imports {name!r} at module scope; it must be lazily imported "
            "so the package stays importable on a CPU-only box"
        )


def test_import_pulls_no_engine_and_no_accelerator():
    code = (
        "import sys; import ascend_attention; "
        "bad=[m for m in ('vllm','vllm_ascend','tilelang') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC.parent)},
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"import ascend_attention pulled in {proc.stdout.strip()}"


def test_public_surface_is_the_three_verbs():
    import ascend_attention as aa

    for name in ("capabilities", "plan", "run"):
        assert callable(getattr(aa, name)), name
    assert set(aa.__all__) >= {"capabilities", "plan", "run", "capability_manifest"}
