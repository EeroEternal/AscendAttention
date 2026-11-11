"""Operator implementations, one subpackage per family.

Importing this package registers every implementation with :data:`ascend_attention.spec.REGISTRY`.
Subpackages are imported eagerly (registration is cheap and must be deterministic); the *kernels*
they reference are imported lazily inside their functions, so this stays true on a CPU-only box.
"""

from __future__ import annotations

from . import attention, fla, gdn, mla, moe, quant, sampling  # noqa: F401  (registration side effect)

__all__ = ["attention", "fla", "gdn", "mla", "moe", "quant", "sampling"]
