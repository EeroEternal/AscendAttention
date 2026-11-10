"""Attention capabilities.

In P0 this package only *registers what exists*: the portable ``reference`` backend, which is an
oracle and therefore explicitly **not** advertised (no ``measured_ratio``, so it can never become
the default and never appears in the manifest).  The Ascend 950 decode/prefill kernels arrive in P1
together with their first real measurement; until then, ``BatchAttention`` (``ascend_attention.api``)
is the module the platform talks to.
"""

from __future__ import annotations

from ...spec import REGISTRY, ImplSpec

__all__ = ["REFERENCE"]

#: The portable oracle: correct on any backend, ~100x slower than production.  Registered so that
#: ``supports()`` and ``describe()`` can answer "we *can* compute this, portably", and so P1's
#: numerical tests have a name to compare against.
REFERENCE = ImplSpec(
    op="paged_decode_attention",
    impl="reference",
    form="torch",
    device="any",
    shape=None,
    kv_layouts=frozenset({"ND"}),
    graph="eager_only",
    plan_kind="attention",
    enabled_by_default=False,
    measured_ratio=None,
    ratio_note="oracle, not a candidate: no measurement is possible or desired",
    bench=None,
    provenance="tileinfer/attention/backends/reference.py + tileinfer/testing/reference.py",
    note="portable fp64-capable oracle; the production kernel lands in P1",
)

REGISTRY.register(REFERENCE)
