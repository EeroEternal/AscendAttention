"""The operator layer must not import the migration source - at any scope, including lazily.

A clean-environment run of Qwen3.6 crashed on the first prefill with
``ModuleNotFoundError: No module named 'tileinfer'`` raised from ``ops/fla/hostcache.py``: the GDN
chunk chain read its switches from ``tileinfer.serving.flags``. Every machine in the migration had
``tileinfer`` installed, so nothing noticed - and the existing hygiene test could not, because it
imported modules and those imports were inside functions.

So this test does not import anything: it parses every file under ``src/`` and looks at the AST, which
sees function-scope imports too.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

FORBIDDEN_ROOTS = ("tileinfer",)


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, node.module or ""


def test_no_module_imports_the_migration_source():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for lineno, module in _imports(path):
            if module.split(".")[0] in FORBIDDEN_ROOTS:
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {module}")
    assert not offenders, (
        "ascend_attention must run without the migration source installed; these imports reach into it:\n  "
        + "\n  ".join(offenders)
    )


def test_the_package_imports_with_the_migration_source_blocked():
    """A subprocess with ``tileinfer`` poisoned: every module must still import."""
    code = (
        "import sys, pkgutil, importlib\n"
        "sys.modules['tileinfer'] = None\n"  # None in sys.modules makes `import tileinfer` raise ImportError
        "import ascend_attention\n"
        "failed = []\n"
        "for mod in pkgutil.walk_packages(ascend_attention.__path__, 'ascend_attention.'):\n"
        "    try:\n"
        "        importlib.import_module(mod.name)\n"
        "    except Exception as exc:\n"
        "        if 'tileinfer' in f'{type(exc).__name__}: {exc}':\n"
        "            failed.append((mod.name, f'{type(exc).__name__}: {exc}'))\n"
        "print(failed)\n"
        "raise SystemExit(1 if failed else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
