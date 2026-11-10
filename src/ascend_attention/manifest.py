"""The published manifest: the library's capabilities as pure JSON, for the platform to read.

Contract
--------
The platform repo (`AscendPlatform`) owns the reader (`contracts/manifest.py`) and the schema; this
module owns the writer.  The emitted document is exactly::

    {"schema": 1,
     "library": {"name": "ascend-attention", "version": "0.1.0",
                 "requires_platform": ">=0.1,<0.2"},
     "ops": [{"name": ..., "shapes": {...}, "dtypes": [...], "kv_layouts": [...],
              "graph_capturable": bool, "measured_ratio": float, "bench": "bench/..."}]}

Two rules are enforced *here*, before the platform's validator ever sees the file:

* an implementation without ``measured_ratio`` **and** ``bench`` never reaches the manifest — the
  platform literally cannot select what is not listed;
* ``graph_capturable`` is stated for every op (the platform's parser rejects a missing answer,
  because a non-capturable op inside a captured region is silently wrong numerics).

Aggregation rule for a multi-implementation op (one entry per op name, as the schema demands):
the entry describes the *selected* implementation, dtypes and layouts are the union over the
eligible implementations (the op as a whole can serve them), ``graph_capturable`` is the selected
implementation's tier (conservative), and ``measured_ratio`` is the best recorded ratio among the
eligible ones — i.e. "what this op buys you".  The per-implementation detail is kept in the extra
``impls`` field, which the platform's parser ignores by design.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from . import dispatch
from ._version import __version__
from .spec import REGISTRY, ImplSpec, OpEntry

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_SCHEMA",
    "LIBRARY_NAME",
    "REQUIRES_PLATFORM",
    "manifest",
    "dumps",
    "write",
    "path",
    "load",
]

MANIFEST_SCHEMA_VERSION = 1

#: The distribution/owner name the platform matches against (``library.name``).
LIBRARY_NAME = "ascend-attention"

#: The platform range this build was written against (`AscendPlatform` is 0.1.x).
REQUIRES_PLATFORM = ">=0.1,<0.2"

#: A compact JSON-Schema copy, so a third party can validate without importing either repo.
MANIFEST_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AscendAttention operator manifest",
    "type": "object",
    "required": ["schema", "library", "ops"],
    "properties": {
        "schema": {"const": MANIFEST_SCHEMA_VERSION},
        "library": {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "requires_platform": {"type": "string"},
            },
        },
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "name",
                    "dtypes",
                    "kv_layouts",
                    "graph_capturable",
                    "measured_ratio",
                    "bench",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "shapes": {"type": "object"},
                    "dtypes": {"type": "array", "items": {"type": "string"}},
                    "kv_layouts": {"type": "array", "items": {"type": "string"}},
                    "graph_capturable": {"type": "boolean"},
                    "measured_ratio": {"type": "number", "exclusiveMinimum": 0},
                    "bench": {"type": "string"},
                    "impl": {"type": "string"},
                    "impls": {"type": "array", "items": {"type": "string"}},
                    "form": {"type": "string"},
                },
            },
        },
    },
}


def _eligible(entry: OpEntry) -> List[ImplSpec]:
    """Advertised, enabled, above the contract line — in priority order."""
    out = []
    for impl in entry.impls:
        if not impl.advertised or not impl.enabled_by_default:
            continue
        if impl.measured_ratio is not None and float(impl.measured_ratio) < dispatch.DEFAULT_THRESHOLD:
            continue
        out.append(impl)
    return out


def _dtypes(impls: Iterable[ImplSpec]) -> List[str]:
    seen = set()
    for impl in impls:
        if impl.shape is not None:
            seen.update(str(d) for d in impl.shape.dtypes)
    return sorted(seen)


def _layouts(impls: Iterable[ImplSpec]) -> List[str]:
    seen = set()
    for impl in impls:
        seen.update(str(l) for l in impl.kv_layouts)
    return sorted(seen)


def _shapes(impl: ImplSpec) -> Dict[str, Any]:
    """Flatten the impl's shape language into the manifest's ``shapes`` object."""
    out: Dict[str, Any] = {}
    if impl.shape is not None:
        from .spec import _dim_repr

        out["ndim"] = impl.shape.ndim
        out["dims"] = [_dim_repr(d) for d in (impl.shape.dims or (None,) * impl.shape.ndim)]
    for key, constraint in impl.shapes.items():
        from .spec import _dim_repr

        out[key] = _dim_repr(constraint)
    return out


def _entry(entry: OpEntry) -> Dict[str, Any]:
    impls = _eligible(entry)
    if not impls:
        raise ValueError(
            f"{entry.name}: no admissible implementation to advertise "
            "(needs measured_ratio + bench; see docs/plan-migrate-tileinfer.md §3.2)"
        )
    decision = dispatch.select(entry)
    primary = decision.impl if decision.impl in impls else impls[0]
    ratio = max(float(i.measured_ratio) for i in impls)
    return {
        "name": entry.name,
        "shapes": _shapes(primary),
        "dtypes": _dtypes(impls),
        "kv_layouts": _layouts(impls),
        "graph_capturable": primary.graph == "capture_safe",
        "measured_ratio": ratio,
        "bench": primary.bench or "",
        "impl": primary.impl,
        "impls": [i.impl for i in impls],
        "form": primary.form,
    }


def manifest(*, only: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """The manifest as a plain dict (JSON-serialisable; no callables anywhere)."""
    names = tuple(only) if only is not None else REGISTRY.names()
    ops = []
    for name in names:
        entry = REGISTRY.entry(name)
        if not _eligible(entry):
            continue
        ops.append(_entry(entry))
    return {
        "schema": MANIFEST_SCHEMA_VERSION,
        "library": {
            "name": LIBRARY_NAME,
            "version": __version__,
            "requires_platform": REQUIRES_PLATFORM,
        },
        "ops": ops,
    }


def dumps(**kwargs: Any) -> str:
    return json.dumps(manifest(**kwargs), indent=2, sort_keys=False)


def path() -> Path:
    """Where the shipped manifest lives (next to the package, inside the wheel)."""
    return Path(__file__).with_name("manifest.json")


def write(target: Optional[os.PathLike] = None) -> Path:
    """Regenerate the committed manifest (CI does this and diffs the result)."""
    target = Path(target) if target is not None else path()
    target.write_text(dumps() + "\n", encoding="utf-8")
    return target


def load(target: Optional[os.PathLike] = None) -> Dict[str, Any]:
    target = Path(target) if target is not None else path()
    return json.loads(target.read_text(encoding="utf-8"))


def __main__() -> None:  # pragma: no cover - CLI convenience
    target = write()
    print(f"wrote {target}")


if __name__ == "__main__":  # pragma: no cover
    __main__()
