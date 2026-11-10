"""The dispatch gate: four steps, in a fixed order, none of which may allocate or sync.

Inherited verbatim from ``tileinfer.vllm.ops.candidates`` (the one piece of that repo whose
*contract* is worth copying unchanged), with one substitution: step ④ evaluates a
:class:`~ascend_attention.spec.ShapeSpec` instead of calling a lambda, so the platform can run the
same test with plain JSON at start-up.

The gate, in order
------------------
1. ``ASCEND_ATTN_CANDIDATES=off`` — process-wide kill switch, read per call (a dict hit).
2. ``enabled_by_default`` — the explicit "this is a candidate, not the default" flag.
3. ``measured_ratio >= threshold`` — a recorded number below 1.0 is ineligible **even if enabled**.
4. device acceptance — an impl tied to one device is rejected elsewhere (pure data: ``impl.device``
   against the current profile's name), so a vendor or kernel impl can be *enabled* without making a
   CPU box route to it;
5. shape / dtype / layout / graph acceptance — pure data (:meth:`ImplSpec.accepts_shape`).

Falling through every step returns the baseline (the ``torch`` implementation), which is the
pre-candidate behaviour byte for byte.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .spec import REGISTRY, ImplSpec, OpEntry

__all__ = [
    "DEFAULT_THRESHOLD",
    "ENV_CANDIDATES",
    "ENV_IMPL",
    "ENV_PLAN",
    "candidates_enabled",
    "pin_impl",
    "pinned_plan",
    "select",
    "describe",
    "kill_switch_state",
]

logger = logging.getLogger("ascend_attention.dispatch")

#: Contract line for a recorded measurement.  ``1.0`` = parity with the baseline.
DEFAULT_THRESHOLD = 1.0

#: Kill switch.  Direction is one-way on purpose: it can only *disarm*.
ENV_CANDIDATES = "ASCEND_ATTN_CANDIDATES"

#: Optional implementation pin (``op=impl``, comma separated).  Diagnostics / A-B runs only.
ENV_IMPL = "ASCEND_ATTN_IMPL"

#: Optional plan-variant pin (``op=variant``); consumed by implementations that have variants.
ENV_PLAN = "ASCEND_ATTN_PLAN"

_FALSE = {"", "0", "false", "off", "no"}

#: Cached device-profile name; probing ``/dev`` costs a ``stat`` and dispatch may not do that per call.
_DEVICE_NAME: Optional[str] = None

#: ``module -> bool`` result of the one ``find_spec`` probe per module; never imports anything.
_HAVE_MODULE: Dict[str, bool] = {}


def module_available(name: str) -> bool:
    """Is ``name`` importable?  ``find_spec`` only - a spec probe does not execute the module.

    The result is cached because ``find_spec`` walks ``sys.path`` (a few ``stat`` calls) and dispatch is
    allowed to be cheap; a missing module cannot appear later in the same process without a restart.
    """
    if name not in _HAVE_MODULE:
        import importlib.util

        try:
            _HAVE_MODULE[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            _HAVE_MODULE[name] = False
    return _HAVE_MODULE[name]


def current_device_name() -> str:
    """The profile name dispatch compares ``ImplSpec.device`` against (cached after the first call)."""
    global _DEVICE_NAME
    if _DEVICE_NAME is None:
        from .device import current

        _DEVICE_NAME = current().name
    return _DEVICE_NAME


def candidates_enabled() -> bool:
    """Is the kill switch on?  Unset / garbage means *on* (shipping behaviour must be the default)."""
    raw = os.environ.get(ENV_CANDIDATES)
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSE


def kill_switch_state() -> Dict[str, Any]:
    return {"env": ENV_CANDIDATES, "raw": os.environ.get(ENV_CANDIDATES), "enabled": candidates_enabled()}


def _parse_pins(env: str) -> Dict[str, str]:
    raw = os.environ.get(env)
    if not raw:
        return {}
    pins: Dict[str, str] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            logger.warning("%s: ignoring %r (expected op=impl)", env, chunk)
            continue
        op, _, impl = chunk.partition("=")
        pins[op.strip()] = impl.strip()
    return pins


def pin_impl(op: str) -> Optional[str]:
    return _parse_pins(ENV_IMPL).get(op)


def pinned_plan(op: str) -> Optional[str]:
    return _parse_pins(ENV_PLAN).get(op)


@dataclass(frozen=True)
class Decision:
    """What ``select`` decided, plus why — cheap enough to build only when asked."""

    op: str
    impl: Optional[ImplSpec]
    reason: str
    rejected: Tuple[Tuple[str, str], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "impl": None if self.impl is None else self.impl.impl,
            "reason": self.reason,
            "rejected": [{"impl": name, "why": why} for name, why in self.rejected],
        }


def select(entry: OpEntry, **query: Any) -> Decision:
    """Pick an implementation for one call.  Zero allocation, zero sync, no timing."""
    pinned = pin_impl(entry.name)
    rejected = []

    # ① kill switch: disarms everything except the baseline (and an explicit pin).
    global_on = candidates_enabled()

    # Candidates first (highest recorded ratio first), the torch baseline last: the baseline is the
    # fallback, never the first thing tried, and registration order must not decide a performance
    # question.  Sorting is over a handful of specs and stable, so it cannot allocate per tensor.
    candidates = [i for i in entry.impls if i.form != "torch"]
    ordered = sorted(candidates, key=lambda i: -(i.measured_ratio or 0.0)) + [
        i for i in entry.impls if i.form == "torch"
    ]

    for impl in ordered:
        if pinned is not None and impl.impl == pinned:
            return Decision(entry.name, impl, f"pinned by {ENV_IMPL}", tuple(rejected))
        if pinned is not None and impl.impl != pinned and impl.form != "torch":
            rejected.append((impl.impl, f"pinned to {pinned}"))
            continue
        # ② static flag
        if not global_on and impl.form != "torch":
            rejected.append((impl.impl, f"{ENV_CANDIDATES}=off"))
            continue
        if impl.measured_ratio is not None and float(impl.measured_ratio) < DEFAULT_THRESHOLD:
            rejected.append(
                (impl.impl, f"measured {float(impl.measured_ratio):.2f}x < {DEFAULT_THRESHOLD:.2f}x")
            )
            continue
        if not impl.enabled_by_default:
            rejected.append((impl.impl, "off by default"))
            continue
        # ③ availability: an impl whose optional dependency is absent must not be chosen at all
        missing = [name for name in sorted(impl.requires) if not module_available(name)]
        if missing:
            rejected.append((impl.impl, f"missing {', '.join(missing)}"))
            continue
        # ③ device acceptance: a kernel written for the 950 must not be chosen on a 910B or a CPU
        if impl.device not in ("any", current_device_name()):
            rejected.append((impl.impl, f"needs device {impl.device}"))
            continue
        # ④ data-only acceptance
        if not impl.accepts_shape(**query):
            rejected.append((impl.impl, "shape/dtype/layout/graph does not match"))
            continue
        return Decision(entry.name, impl, "eligible", tuple(rejected))

    baseline = entry.baseline
    if baseline is None:
        return Decision(entry.name, None, "no eligible implementation and no torch baseline", tuple(rejected))
    return Decision(entry.name, baseline, "baseline", tuple(rejected))


def describe(op: str) -> Dict[str, Any]:
    """The gate state of every implementation of ``op``, as plain data (for doctor/reports)."""
    entry = REGISTRY.entry(op)
    rows = []
    for impl in entry.impls:
        decision = select(entry, shape=None)
        data = impl.as_dict()
        data["status"] = (
            "selected"
            if decision.impl is not None and decision.impl.impl == impl.impl
            else "eligible manually"
        )
        data["advertised"] = impl.advertised
        rows.append(data)
    return {
        "op": op,
        "doc": entry.doc,
        "switch": kill_switch_state(),
        "threshold": DEFAULT_THRESHOLD,
        "baseline": None if entry.baseline is None else entry.baseline.impl,
        "impls": rows,
    }
