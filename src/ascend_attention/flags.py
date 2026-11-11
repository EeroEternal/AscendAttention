"""This package's runtime switches, read from **this package's** namespace only.

The switches below started life as ``tileinfer.serving.flags`` lookups during the migration.  That
import is what broke a clean environment: the flag reader was reached from the GDN prefill chain
(``chunk_local_cumsum`` -> ``_chunk_cumsum`` -> ``chunk_gated_delta_rule_fwd``, 30 times per step on
Qwen3.6), so ``import ascend_attention`` succeeded but the first prefill raised
``ModuleNotFoundError: No module named 'tileinfer'`` - a hard dependency on the migration *source*,
invisible on any machine that happened to have it installed.

Two rules keep that from coming back:

* switches are ``ASCEND_ATTENTION_*`` environment variables, read with :func:`flag`; there is no flag
  file and no second namespace to fall back to (a fallback would just hide the next leak);
* no module under ``src/`` may import ``tileinfer`` at all, at any scope.  ``tests/test_zero_tileinfer.py``
  walks the AST of every file to enforce it, because a lazy import inside a function is exactly the
  shape the old ``import``-time hygiene test could not see.

Defaults are the measured choice, which is why most of them are ``False``: an experiment that lost its
A/B does not become a default, it becomes a switch nobody sets.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = ["ENV_PREFIX", "flag", "env_name"]

#: Every switch this package reads is ``ASCEND_ATTENTION_<suffix>``.
ENV_PREFIX = "ASCEND_ATTENTION_"

_TRUTHY = frozenset({"1", "true", "True", "TRUE", "yes", "on"})
_FALSY = frozenset({"0", "false", "False", "FALSE", "no", "off", ""})


def env_name(suffix: str) -> str:
    """``"GDN_CONV_FAST"`` -> ``"ASCEND_ATTENTION_GDN_CONV_FAST"`` (the only accepted spelling)."""
    suffix = suffix.strip().upper()
    if not suffix:
        raise ValueError("env_name: suffix must not be empty")
    return ENV_PREFIX + suffix


def flag(suffix: str, default: bool, *, environ: Optional[dict] = None) -> bool:
    """Read one switch, or ``default`` when it is unset.

    Unrecognised values raise instead of being read as ``False``: a typo (``ASCEND_ATTENTION_GDN_CONV_FAST=ture``)
    silently selecting the *other* branch is how a measurement attributes a difference to the wrong thing.
    """
    source = os.environ if environ is None else environ
    name = env_name(suffix)
    raw = source.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean: use one of {sorted(_TRUTHY)} or {sorted(_FALSY - {''})}, "
        "or unset it for the default"
    )
