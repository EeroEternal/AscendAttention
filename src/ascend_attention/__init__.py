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
from .capture import capture_guard, check_capture
from .dispatch import Decision, describe as describe_op, select
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
    "describe_op",
    "implementations",
    "device",
    "OpPlan",
    "ImplSpec",
    "ShapeSpec",
    "DeviceProfile",
    "capture_guard",
    "check_capture",
    "prewarm",
]


def capabilities(op: Optional[str] = None) -> Dict[str, Any]:
    """What this build can do, as plain data.  No device is touched and no tensor is built.

    With ``op`` this is the per-op description (``describe_op``); without it, the whole inventory.
    """
    if op is not None:
        return describe_op(op)
    return {
        "library": "ascend-attention",
        "version": __version__,
        "device": device_profile(current_device().name).name,
        "ops": describe_all(),
    }


def describe_all() -> List[Dict[str, Any]]:
    return [describe_op(name) for name in REGISTRY.names()]


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


def describe() -> Dict[str, Any]:
    """The provider descriptor: what this package serves and where the code is.

    Same document as ``ascend_attention.provider.descriptor()``, exposed at the package root because the
    platform's pinned-root discovery (``ASCEND_PLATFORM_OPS=ascend_attention``) calls ``describe()`` on
    the package itself - a user pointing the platform at a checkout should not have to install metadata
    for the binding to work.
    """
    from .provider import descriptor as _descriptor

    return _descriptor()


def device() -> DeviceProfile:
    return current_device()


def prewarm(keys: Iterable[str]) -> Tuple[str, ...]:
    """Deduplicate and order the compile keys a platform should pay for, **without compiling**.

    A compile takes 70-100 s of GIL (``architecture.md:185-197``), which inside a serving process looks
    like a hang, so this library never compiles in the background and never on a call path: ``plan``
    records which artifact it needs in :meth:`OpPlan.spec_key`, and this function is the pure-data step
    that tells the platform what that set is.  Calling it from a warm-up task outside the request path is
    the intended use; calling it is cheap and side-effect free, so calling it twice is harmless.

    Args:
        keys: whatever the caller collected from ``plan(...).spec_key()``; duplicates are fine.

    Returns:
        The distinct keys, sorted, as a tuple.  Sorting matters: it makes the prewarm log diffable
        between two boots.
    """
    if keys is None or isinstance(keys, (str, bytes)):
        raise TypeError("prewarm takes an iterable of compile keys, not a single string")
    return tuple(sorted({str(key) for key in keys}))
