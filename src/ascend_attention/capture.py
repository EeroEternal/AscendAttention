"""``capture_guard``: does a declared ``graph`` tier survive contact with the actual call?

The three tiers in :data:`ascend_attention.spec.GRAPHS` are only useful if they are *checkable*:

* ``capture_safe`` — may allocate (the caching allocator handles static shapes), but must never read
  the device back to the host and must never synchronise;
* ``eager_only`` — correct, but needs a host read, an allocation whose size depends on data, or a
  compile;
* ``unknown`` — not verified.

"May allocate but may not sync" is the distinction a two-tier scheme loses, so the guard reports the
two counts **separately** instead of a single verdict.  It watches real dispatch: a
``TorchDispatchMode`` sees every aten op, so it catches ``.item()`` (``aten::_local_scalar_dense``),
``.cpu()``/``.numpy()`` (device->host copies), explicit synchronisation and allocations without reading
any source code.

Limits, stated so nobody over-trusts a green run: it only sees ops that go through the torch dispatcher.
A host read inside a compiled TileLang kernel, a ``ctypes`` call, or a Python-level loop over a list
that was already synced earlier is invisible here.  That is why a ``capture_safe`` declaration still
needs a device A/B under real graph capture — the guard rules out the cheap mistakes, not all of them.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

import torch

__all__ = [
    "ALLOC_PREFIXES",
    "SYNC_OPS",
    "CaptureGuard",
    "capture_guard",
    "check_capture",
]

#: Allocations are allowed under capture when their sizes are static; counted, not judged.
ALLOC_PREFIXES = (
    "aten::empty",
    "aten::new_empty",
    "aten::zeros",
    "aten::ones",
    "aten::full",
    "aten::clone",
    "aten::contiguous",
    "aten::to",
    "aten::rand",
    "aten::normal",
    "aten::uniform",
)

#: Device -> host reads and explicit synchronisation.  Any of these inside a captured region either
#: fails the capture or (worse) silently breaks the CUDA/NPU graph's dependency tracking.
SYNC_OPS = frozenset(
    {
        "aten::_local_scalar_dense",  # .item(), float(t), int(t), bool(t)
        "aten::item",
        "aten::_local_scalar",
        "aten::cpu",
        "aten::numpy",
        "aten::tolist",
        "aten::synchronize",
        "aten::cuda_synchronize",
        "aten::npu_synchronize",
    }
)

#: ``aten::to``/``aten::copy_`` are only a problem when they move data to the host; the guard looks at
#: the target device rather than the op name for those.
_CPU_TRANSFER_OPS = frozenset({"aten::to", "aten::copy_", "aten::_to_copy"})


class CaptureGuard:
    """Records syncs and allocations performed inside ``with guard:``.

    Attributes:
        syncs: the offending ops, in call order, as ``"aten::name(args...)"`` strings.
        allocs: every allocating op, in call order (expected under ``capture_safe``).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.syncs: List[str] = []
        self.allocs: List[str] = []
        self.ops: List[str] = []
        self._enabled = enabled
        self._mode: Optional[Any] = None

    # -- reporting ---------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "syncs": len(self.syncs),
            "allocs": len(self.allocs),
            "distinct_ops": len(set(self.ops)),
            "sync_ops": sorted({s.split("(")[0] for s in self.syncs}),
            "capture_safe": not self.syncs,
        }

    def explain(self) -> str:
        if not self.syncs:
            return f"capture-safe: {len(self.allocs)} allocations, no host reads"
        head = ", ".join(sorted({s.split("(")[0] for s in self.syncs}))
        return f"NOT capture-safe: {len(self.syncs)} host read(s)/sync(s) via {head}"

    # -- dispatch -----------------------------------------------------------------
    def _classify(self, func: Any, args: Sequence[Any], kwargs: Dict[str, Any]) -> None:
        raw = str(func)
        self.ops.append(raw)
        name = _canonical(raw)
        if name in SYNC_OPS:
            self.syncs.append(f"{name}({_short(args)})")
            return
        if name in _CPU_TRANSFER_OPS and _moves_to_cpu(args, kwargs):
            self.syncs.append(f"{name}({_short(args)})")
            return
        if any(name.startswith(prefix) for prefix in ALLOC_PREFIXES):
            self.allocs.append(name)

    def __enter__(self) -> "CaptureGuard":
        if not self._enabled:
            return self
        mode = _Mode(self)
        mode.__enter__()
        self._mode = mode
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._mode is not None:
            self._mode.__exit__(*exc_info)
            self._mode = None


class _Mode:
    """Thin adapter so the guard does not have to subclass ``TorchDispatchMode`` at import time."""

    def __init__(self, guard: CaptureGuard) -> None:
        from torch.utils._python_dispatch import TorchDispatchMode

        guard_ref = guard

        class _Inner(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
                guard_ref._classify(func, args, kwargs or {})
                return func(*args, **(kwargs or {}))

        self._inner = _Inner()

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._inner.__exit__(*exc_info)


def _canonical(raw: str) -> str:
    """``aten.empty.memory_format`` / ``aten::empty`` -> ``aten::empty``.

    The dispatch mode reports ``aten.<op>.<overload>`` while the docs and error messages speak
    ``aten::<op>``; normalising here keeps the classification tables readable.
    """
    text = raw.replace("::", ".")
    parts = text.split(".")
    if len(parts) >= 2 and parts[0] == "aten":
        return f"aten::{parts[1]}"
    return raw


def _moves_to_cpu(args: Sequence[Any], kwargs: Dict[str, Any]) -> bool:
    """A device->host copy, as opposed to a dtype/shape cast that happens to name ``cpu``.

    On a CPU-only box every tensor is already host-resident, so nothing here is observable; that is a
    property of the box, not of the operator, and the tier claim for device operators has to be checked
    on the device.
    """
    source = next((a for a in args if isinstance(a, torch.Tensor)), None)
    if source is None or source.device.type == "cpu":
        return False
    return _lands_on_cpu(args, kwargs)


def _lands_on_cpu(args: Sequence[Any], kwargs: Dict[str, Any]) -> bool:
    for value in list(args) + list(kwargs.values()):
        device = getattr(value, "type", None)
        if isinstance(device, str) and device == "cpu":
            return True
        if isinstance(value, torch.device) and value.type == "cpu":
            return True
    return False


def _short(args: Sequence[Any], limit: int = 2) -> str:
    parts = []
    for arg in args[:limit]:
        if isinstance(arg, torch.Tensor):
            parts.append(f"<{arg.dtype} {tuple(arg.shape)} on {arg.device.type}>")
        else:
            parts.append(repr(arg)[:40])
    return ", ".join(parts)


@contextmanager
def capture_guard(enabled: bool = True):
    """``with capture_guard() as guard: run(p, *tensors)`` — then read ``guard.summary()``."""
    guard = CaptureGuard(enabled=enabled)
    with guard:
        yield guard


def check_capture(fn, *args, **kwargs) -> Dict[str, Any]:
    """Call ``fn`` under the guard and return its summary plus the result under ``"result"``."""
    guard = CaptureGuard()
    with guard:
        result = fn(*args, **kwargs)
    summary = guard.summary()
    summary["result"] = result
    summary["detail"] = guard.explain()
    return summary
