"""AscendAttention — an engine-agnostic library of Ascend fused operators.

Three verbs, and nothing else public:

.. code-block:: python

    from ascend_attention import capabilities, plan, run

    caps = capabilities()                    # what exists, where, and what it measured
    p = plan("gdn_gating", shape=(1, 5, 16), dtype="bfloat16")
    g, beta = run(p, a_log, a, b, dt_bias)   # tensors and pure data only

Design rules this module exists to enforce (see ``README.md``):

* **no engine in here** — no ``vllm`` / ``vllm_ascend`` import anywhere in the package (there is a
  test that walks the AST of every file);
* **dispatch is cheap** — :func:`run` never measures, never compiles, never syncs; the four gates
  are a dict lookup, two booleans and a shape-language evaluation;
* **one manifestation of capability** — the same :class:`~ascend_attention.spec.ImplSpec` data
  answers :func:`supports`, drives :func:`plan`, and is serialised by
  :func:`~ascend_attention.manifest.manifest` for the platform to read at start-up;
* **optional dependencies are optional** — ``torch_npu`` and ``tilelang`` are imported inside
  functions only, so ``import ascend_attention`` works on a CPU-only box.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ._version import __version__
from .dispatch import Decision, describe, select
from .device import DeviceProfile, current as current_device, profile as device_profile
from .manifest import manifest as _manifest
from .plan import OpPlan
from .spec import REGISTRY, ImplSpec, ShapeSpec

# Importing the operator packages is what registers their implementations.  This is the only
# module-level side effect in the package, and it is confined to a single line.
from . import ops as _ops  # noqa: F401,E402  (registration)

__all__ = [
    "__version__",
    "capabilities",
    "supports",
    "plan",
    "run",
    "capability_manifest",
    "describe",
    "implementations",
    "device",
    "OpPlan",
    "ImplSpec",
    "ShapeSpec",
    "DeviceProfile",
]


def capabilities(op: Optional[str] = None) -> Dict[str, Any]:
    """What this build can do, as plain data.  No device is touched and no tensor is built."""
    if op is not None:
        return describe(op)
    return {
        "library": "ascend-attention",
        "version": __version__,
        "device": device_profile(current_device().name).name,
        "ops": describe_all(),
    }


def describe_all() -> List[Dict[str, Any]]:
    return [describe(name) for name in REGISTRY.names()]


def implementations(op: str) -> Tuple[ImplSpec, ...]:
    """Every registered implementation of ``op``, selected one first."""
    decision = select(REGISTRY.entry(op))
    impls = list(REGISTRY.impls(op))
    if decision.impl is not None:
        impls.sort(key=lambda i: 0 if i.impl == decision.impl.impl else 1)
    return tuple(impls)


def supports(
    op: str,
    *,
    shape: Optional[Sequence[int]] = None,
    dtype: Any = None,
    kv_layout: Optional[str] = None,
    graph: Optional[bool] = None,
    contiguous: bool = True,
    **pure: Any,
) -> bool:
    """Could ``op`` run this call **without** building a plan?  Pure metadata, zero allocation."""
    entry = REGISTRY.entry(op)
    return bool(entry.eligible(shape=shape, dtype=dtype, kv_layout=kv_layout, graph=graph,
                               contiguous=contiguous, **pure))


def plan(op: str, *, impl: Optional[str] = None, **pure: Any) -> OpPlan:
    """Build a plan: pure data plus a handle, no device work beyond what the planner needs.

    ``impl`` pins an implementation explicitly (oracle/A-B runs); otherwise the four gates decide.
    """
    entry = REGISTRY.entry(op)
    if impl is not None:
        chosen = next((i for i in entry.impls if i.impl == impl), None)
        if chosen is None:
            raise KeyError(
                f"{op} has no impl {impl!r}; known: {[i.impl for i in entry.impls]}"
            )
        decision = Decision(op, chosen, "pinned by argument")
    else:
        decision = select(entry, **pure)
        if decision.impl is None:
            reasons = "; ".join(f"{name}: {why}" for name, why in decision.rejected)
            raise RuntimeError(f"no implementation for {op}: {decision.reason} ({reasons})")
    planner = REGISTRY.planners.get(op)
    if planner is None:
        raise RuntimeError(f"{op} has no planner registered")
    return planner(decision.impl, **pure)


def run(p: OpPlan, *tensors: Any) -> Any:
    """Execute a plan.  Tensors in, tensors out — no engine metadata, no kwargs soup."""
    if not isinstance(p, OpPlan):
        raise TypeError(f"run() takes an OpPlan from plan(), got {type(p).__name__}")
    return p(*tensors)


def capability_manifest(**kwargs: Any) -> Dict[str, Any]:
    """The published manifest (see :mod:`ascend_attention.manifest`).

    Named ``capability_manifest`` on purpose: a module attribute called ``manifest`` would shadow
    the submodule of the same name, and ``ascend_attention.manifest.load()`` would stop working.
    """
    return _manifest(**kwargs)


def device() -> DeviceProfile:
    return current_device()
