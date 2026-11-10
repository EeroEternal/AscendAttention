"""Device facts as data — the only place a device constant may live.

The platform owns a ``DeviceProfile`` of its own; this is the **operator side's** mirror: the
minimum a kernel needs to know in order to answer "can I run here?" without importing ``torch_npu``
(which is optional, and must never be imported at module scope — ``import ascend_attention`` has to
work on a CPU-only box).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

__all__ = ["DeviceProfile", "PROFILES", "profile", "profiles", "current", "profile_for_dtype"]

logger = logging.getLogger("ascend_attention.device")


@dataclass(frozen=True)
class DeviceProfile:
    """What the kernels need to know about a device; field names mirror the platform's profile."""

    name: str
    vendor: str = "ascend"
    hbm_bytes: Optional[int] = None
    compute_units: int = 0
    cube_units: int = 0
    vector_units: int = 0
    #: Kernel-side tile size (the page size a compiled kernel is built for).
    kernel_page_size: int = 128
    #: Engine-side page count multiple used by hybrid KV (platform fact, mirrored for sanity).
    engine_page_multiple: int = 2048
    dtypes: Tuple[str, ...] = ("bfloat16", "float16")
    kv_cache_dtypes: Tuple[str, ...] = ("auto", "bfloat16", "float16")
    layouts: Tuple[str, ...] = ("ND", "NZ")
    capabilities: frozenset = frozenset({"hybrid_kv", "mamba_state"})

    def supports_dtype(self, dtype: str) -> bool:
        return str(dtype) in self.dtypes

    def supports_layout(self, layout: str) -> bool:
        return str(layout) in self.layouts


#: ``Ascend950PR_9579`` = 1 die, 131072 MiB HBM, 28 cube cores / 56 vector cores.
#: Numbers are the ones TileInfer measured on the 102 box (docs/status.md §5) — do not retype them
#: anywhere else; add a profile instead.
PROFILES: Dict[str, DeviceProfile] = {
    "ascend950": DeviceProfile(
        name="ascend950",
        hbm_bytes=131072 << 20,
        compute_units=56,
        cube_units=28,
        vector_units=56,
        kernel_page_size=128,
        engine_page_multiple=2048,
        dtypes=("bfloat16", "float16"),
        layouts=("ND", "NZ"),
        capabilities=frozenset({"hybrid_kv", "mamba_state"}),
    ),
    "ascend910b": DeviceProfile(
        name="ascend910b",
        hbm_bytes=65536 << 20,
        compute_units=48,
        cube_units=24,
        vector_units=48,
        kernel_page_size=128,
        dtypes=("bfloat16", "float16"),
        layouts=("ND",),
        capabilities=frozenset({"hybrid_kv"}),
    ),
    "cpu": DeviceProfile(
        name="cpu",
        vendor="cpu",
        compute_units=0,
        cube_units=0,
        vector_units=0,
        dtypes=("float32", "float16", "bfloat16"),
        kv_cache_dtypes=("auto", "float32", "float16", "bfloat16"),
        layouts=("ND",),
        capabilities=frozenset(),
    ),
}


def profiles() -> Tuple[str, ...]:
    return tuple(PROFILES)


def profile(name: str) -> DeviceProfile:
    try:
        return PROFILES[str(name)]
    except KeyError:
        raise KeyError(f"unknown device {name!r}; known: {sorted(PROFILES)}") from None


def _npu_device_nodes_present() -> bool:
    """Cheap pre-check: without the driver nodes there is no point importing ``torch_npu``.

    Importing it costs a driver probe (and, in a sandbox that masks ``/dev``, prints a scary
    ``dcmi`` error on stderr).  The nodes are the ground truth: ``/dev/davinci0`` is the card and
    ``/dev/davinci_manager`` the management node, both created by the Ascend driver.
    """
    import os

    return os.path.exists("/dev/davinci0") or os.path.exists("/dev/davinci_manager")


def current() -> DeviceProfile:
    """The device this process would run on.  Never imports ``torch_npu`` at module scope."""
    import sys

    if "torch_npu" in sys.modules or _npu_device_nodes_present():
        try:  # pragma: no cover - device dependent
            import torch
            import torch_npu  # noqa: F401  (registers the "npu" backend)

            if torch.npu.is_available():
                return PROFILES["ascend950"]
        except Exception as exc:  # noqa: BLE001 - options are optional
            logger.debug("no NPU device: %r", exc)
    return PROFILES["cpu"]


def profile_for_dtype(dtype: str) -> Optional[DeviceProfile]:
    """First profile that supports ``dtype`` — used by CPU-side plan validation."""
    for candidate in PROFILES.values():
        if candidate.supports_dtype(dtype):
            return candidate
    return None
