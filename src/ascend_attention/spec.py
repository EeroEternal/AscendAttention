"""Declarative capability specs — the data that both the dispatch gate and the manifest use.

Why data and not predicates
---------------------------
The platform repo (`AscendPlatform`) reads this library's capabilities as **plain JSON** at
start-up, on a CPU box, before any tensor exists, and rejects an unsupported batch shape before it
is ever formed.  A callable predicate cannot do that: it needs the arguments.  So the fourth gate of
the dispatch chain is a *shape-language evaluation*, not a lambda.

Everything an implementation may depend on is therefore a field here: dtype set, KV layout set,
whether it can enter a graph, which plan flavour it needs, what it measured, and what it requires at
runtime.  ``supports()`` evaluates this file's data; ``manifest()`` serialises the same data.  One
source, two consumers — that is the whole point.

Vocabulary notes (kept in sync with `AscendPlatform`'s `DeviceProfile`):

* ``dtypes`` are **long torch names** (``"bfloat16"``, not ``"bf16"``) because the device profile
  compares them verbatim.
* ``kv_layouts`` is the **memory format** vocabulary (``"ND"`` / ``"NZ"``), not the page-major /
  head-major distinction.  ``NHD`` vs ``HND`` is an *op parameter* and is declared in ``shapes``
  under the key ``"kv_layout"``; conflating the two makes a device profile reject a perfectly good
  kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "FORMS",
    "GRAPHS",
    "PLAN_KINDS",
    "KV_LAYOUTS",
    "DEVICES",
    "MUL",
    "IN",
    "MIN",
    "MAX",
    "ShapeSpec",
    "ImplSpec",
    "OpEntry",
    "Registry",
    "REGISTRY",
    "dim_in",
    "dim_mul",
    "dim_min",
    "dim_max",
]

#: How an implementation is produced.  ``torch`` = python-level composition (no build artifact);
#: ``tilelang`` = compiled kernel; ``ascendc`` = reserved (no source in either repo yet);
#: ``vendor_code`` = vendored third-party source; ``vendor_call`` = direct call into a vendor op.
FORMS: Tuple[str, ...] = ("torch", "tilelang", "ascendc", "vendor_code", "vendor_call")

#: The forms whose availability depends on the machine: a manifest built on a box without the vendor
#: package cannot reproduce a snapshot built on one that has it.
VENDOR_FORMS: Tuple[str, ...] = ("vendor_code", "vendor_call")

#: Three tiers, not two: "allocates but never syncs" is *not* the same as "capture safe".
GRAPHS: Tuple[str, ...] = ("capture_safe", "eager_only", "unknown")

#: ``attention`` = schedule + workspace + split/merge; ``kernel`` = a compiled handle + staging;
#: ``stateless`` = pure torch composition, allowed to allocate.
PLAN_KINDS: Tuple[str, ...] = ("attention", "kernel", "stateless")

#: Memory-format vocabulary, matching ``DeviceProfile.layouts``.
KV_LAYOUTS: Tuple[str, ...] = ("ND", "NZ")

#: Device keys an implementation may declare.  ``any`` means "runs wherever torch runs".
DEVICES: Tuple[str, ...] = ("any", "cpu", "ascend950", "ascend910b")


# --------------------------------------------------------------------------- dimension language


def dim_mul(value: int) -> Tuple[str, int]:
    """Dimension must be a multiple of ``value``."""
    return ("mul", int(value))


def dim_in(*values: int) -> Tuple[str, Tuple[int, ...]]:
    """Dimension must be one of ``values``."""
    return ("in", tuple(int(v) for v in values))


def dim_min(value: int) -> Tuple[str, int]:
    """Dimension must be at least ``value``."""
    return ("min", int(value))


def dim_max(value: int) -> Tuple[str, int]:
    """Dimension must be at most ``value``."""
    return ("max", int(value))


#: Back-compat aliases used in the docs.
MUL, IN, MIN, MAX = dim_mul, dim_in, dim_min, dim_max

Dim = Any  # int | ("mul", n) | ("in", (…,)) | ("min", n) | ("max", n) | None


def _dim_matches(spec: Dim, value: int) -> bool:
    if spec is None:
        return True
    if isinstance(spec, int):
        return int(value) == spec
    kind, arg = spec
    if kind == "mul":
        return int(value) % int(arg) == 0
    if kind == "in":
        return int(value) in set(int(v) for v in arg)
    if kind == "min":
        return int(value) >= int(arg)
    if kind == "max":
        return int(value) <= int(arg)
    raise ValueError(f"unknown dimension constraint {spec!r}")


def _dim_repr(spec: Dim) -> Any:
    """JSON-able form used by the manifest (``{"mul": 16}`` / ``{"in": [128, 256]}``)."""
    if spec is None:
        return None
    if isinstance(spec, int):
        return int(spec)
    kind, arg = spec
    if kind == "in":
        return {kind: list(arg)}
    return {kind: int(arg)}


@dataclass(frozen=True)
class ShapeSpec:
    """The only shape knowledge a dispatch decision may use — and it must be free to evaluate."""

    ndim: int
    dims: Tuple[Dim, ...] = ()
    dtypes: frozenset = frozenset({"bfloat16"})
    contiguous: bool = True

    def __post_init__(self) -> None:
        if self.dims and len(self.dims) != self.ndim:
            raise ValueError(
                f"ShapeSpec has ndim={self.ndim} but {len(self.dims)} dimension constraints"
            )
        if not self.dtypes:
            raise ValueError("ShapeSpec needs at least one dtype")

    def matches(self, shape: Sequence[int], dtype: Any, contiguous: bool = True) -> bool:
        """Free check: reads metadata only, never a tensor value."""
        shape = tuple(int(d) for d in shape)
        if len(shape) != self.ndim:
            return False
        if self.contiguous and not contiguous:
            return False
        name = _dtype_name(dtype)
        if name not in self.dtypes:
            return False
        for spec, value in zip(self.dims, shape):
            if not _dim_matches(spec, value):
                return False
        return True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ndim": self.ndim,
            "dims": [_dim_repr(d) for d in self.dims] if self.dims else [None] * self.ndim,
            "dtypes": sorted(self.dtypes),
            "contiguous": self.contiguous,
        }


def _dtype_name(dtype: Any) -> str:
    """``torch.bfloat16`` / ``"bf16"`` / ``"bfloat16"`` all normalise to ``"bfloat16"``."""
    text = str(dtype).strip()
    if "." in text:  # torch.bfloat16 -> bfloat16
        text = text.rsplit(".", 1)[-1]
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "half": "float16"}
    return aliases.get(text, text)


# --------------------------------------------------------------------------- implementation spec


@dataclass(frozen=True)
class ImplSpec:
    """One implementation of one operator, with everything needed to gate and to advertise it."""

    op: str
    impl: str
    form: str
    device: str = "any"
    shape: Optional[ShapeSpec] = None
    kv_layouts: frozenset = frozenset({"ND"})
    graph: str = "unknown"
    plan_kind: str = "stateless"
    enabled_by_default: bool = True
    measured_ratio: Optional[float] = None
    ratio_note: str = ""
    bench: Optional[str] = None
    requires: frozenset = frozenset()
    provenance: str = ""
    note: str = ""
    #: Extra data-only predicates: ``{"head_dim": dim_in(128, 256)}`` keyed by pure kwargs.
    shapes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.op or not self.impl:
            raise ValueError("ImplSpec needs both an op and an impl name")
        if self.form not in FORMS:
            raise ValueError(f"{self.impl}: form {self.form!r} not in {FORMS}")
        if self.graph not in GRAPHS:
            raise ValueError(f"{self.impl}: graph {self.graph!r} not in {GRAPHS}")
        if self.plan_kind not in PLAN_KINDS:
            raise ValueError(f"{self.impl}: plan_kind {self.plan_kind!r} not in {PLAN_KINDS}")
        if self.device not in DEVICES:
            raise ValueError(f"{self.impl}: device {self.device!r} not in {DEVICES}")
        if not self.kv_layouts:
            raise ValueError(f"{self.impl}: kv_layouts must not be empty (state ['ND'] if N/A)")
        unknown = set(self.kv_layouts) - set(KV_LAYOUTS)
        if unknown:
            raise ValueError(f"{self.impl}: unknown kv_layouts {sorted(unknown)}")
        # The README's rule, as a registration-time assertion rather than a note in a doc.
        if self.measured_ratio is None and self.enabled_by_default:
            raise ValueError(
                f"{self.impl}: enabled_by_default requires a measured_ratio "
                "(only the baseline may claim 1.0 by definition)"
            )
        if self.enabled_by_default and not self.bench:
            raise ValueError(f"{self.impl}: an enabled implementation must name its bench script")
        if self.measured_ratio is not None and float(self.measured_ratio) <= 0:
            raise ValueError(f"{self.impl}: measured_ratio must be > 0")

    # ----------------------------------------------------------------- queries

    @property
    def advertised(self) -> bool:
        """May this implementation appear in the published manifest?

        The platform's schema requires *both* ``measured_ratio`` and ``bench``; an unmeasured
        implementation is therefore invisible to the platform (which is stronger than "default
        off": the platform cannot even select it by accident).
        """
        return self.measured_ratio is not None and bool(self.bench)

    def accepts_shape(
        self,
        *,
        shape: Optional[Sequence[int]] = None,
        dtype: Any = None,
        kv_layout: Optional[str] = None,
        graph: Optional[bool] = None,
        contiguous: bool = True,
        **pure: Any,
    ) -> bool:
        """Data-only acceptance test.  No tensors, no device, no timing."""
        if shape is not None and self.shape is not None:
            if not self.shape.matches(shape, dtype if dtype is not None else self._only_dtype(),
                                     contiguous):
                return False
        elif shape is not None and self.shape is None:
            return False
        # The dtype gate must not depend on ``shape`` being passed: callers that describe a call with
        # named dimensions (``plan(op, batch=..., vocab=..., dtype=...)``) never supply it, and skipping
        # the check there silently routed float32 logits into a bf16-only vendor path.
        allowed = self.shape.dtypes if self.shape is not None else frozenset()
        if dtype is not None and allowed:
            wanted = str(dtype).replace("torch.", "")
            if wanted not in {str(d) for d in allowed}:
                return False
        if kv_layout is not None and str(kv_layout) not in self.kv_layouts:
            return False
        if graph is True and self.graph != "capture_safe":
            return False
        for key, constraint in self.shapes.items():
            if key in pure and pure[key] is not None and not _dim_matches(constraint, pure[key]):
                return False
        return True

    def _only_dtype(self) -> Any:
        if self.shape is not None and len(self.shape.dtypes) == 1:
            return next(iter(self.shape.dtypes))
        return dtype_any()

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "op": self.op,
            "impl": self.impl,
            "form": self.form,
            "device": self.device,
            "dtypes": sorted(self.shape.dtypes) if self.shape else [],
            "kv_layouts": sorted(self.kv_layouts),
            "graph": self.graph,
            "plan_kind": self.plan_kind,
            "enabled_by_default": self.enabled_by_default,
            "measured_ratio": self.measured_ratio,
            "ratio_note": self.ratio_note,
            "bench": self.bench,
            "requires": sorted(self.requires),
            "provenance": self.provenance,
            "note": self.note,
        }
        if self.shape is not None:
            data["shape_spec"] = self.shape.as_dict()
        if self.shapes:
            data["shapes"] = {k: _dim_repr(v) for k, v in self.shapes.items()}
        return data


def dtype_any() -> str:
    return "*"


@dataclass
class OpEntry:
    """One operator name and its implementations, in priority order."""

    name: str
    doc: str = ""
    impls: List[ImplSpec] = field(default_factory=list)

    def add(self, impl: ImplSpec) -> "OpEntry":
        if impl.op != self.name:
            raise ValueError(f"{impl.impl}: registered under {impl.op!r} != {self.name!r}")
        names = {i.impl for i in self.impls}
        if impl.impl in names:
            raise ValueError(f"duplicate impl {impl.impl!r} for op {self.name!r}")
        self.impls.append(impl)
        return self

    @property
    def baseline(self) -> Optional[ImplSpec]:
        for impl in self.impls:
            if impl.form == "torch":
                return impl
        return None

    def eligible(self, **query: Any) -> List[ImplSpec]:
        """Implementations that pass the static gates *and* accept the query, in priority order."""
        out: List[ImplSpec] = []
        for impl in self.impls:
            if not impl.enabled_by_default:
                continue
            if self.baseline is not None and impl.measured_ratio is not None:
                if impl is not self.baseline and float(impl.measured_ratio) < 1.0:
                    continue
            if impl.accepts_shape(**query):
                out.append(impl)
        return out


class Registry:
    """Process-wide operator registry.  Registration order is priority order."""

    def __init__(self) -> None:
        self._ops: Dict[str, OpEntry] = {}

    def op(self, name: str, doc: str = "") -> OpEntry:
        entry = self._ops.get(name)
        if entry is None:
            entry = self._ops[name] = OpEntry(name=name, doc=doc)
        elif doc and not entry.doc:
            entry.doc = doc
        return entry

    def register(self, impl: ImplSpec) -> ImplSpec:
        self.op(impl.op).add(impl)
        return impl

    def names(self) -> Tuple[str, ...]:
        return tuple(self._ops)

    def entry(self, name: str) -> OpEntry:
        try:
            return self._ops[name]
        except KeyError:
            raise KeyError(
                f"unknown operator {name!r}; known: {sorted(self._ops)}"
            ) from None

    def impls(self, name: str) -> Tuple[ImplSpec, ...]:
        return tuple(self.entry(name).impls)

    def all_impls(self) -> Iterable[ImplSpec]:
        for entry in self._ops.values():
            for impl in entry.impls:
                yield impl

    #: op -> planner callable (set by the op modules, keeps `plan()` generic).
    planners: Dict[str, Callable[..., Any]] = {}

    def planner(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        self.entry(name)
        self.planners[name] = fn
        return fn


#: The one registry the package exposes.
REGISTRY = Registry()
