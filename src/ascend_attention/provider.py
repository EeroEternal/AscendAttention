"""The provider entry point: how AscendPlatform discovers this library and binds semantic ops.

The contract (their ``contracts/provider.py``, ``docs/notes/repo-boundary.md`` §5 there) is three pieces
and this module is the operator half of all three:

1. **Discovery** - a setuptools entry point in the group ``ascend_platform.providers`` named
   ``ascend_attention``, whose value is a factory (:func:`provider`). Their reader also accepts a pinned
   package via ``ASCEND_PLATFORM_OPS=ascend_attention``, which is why the *package* also exposes
   ``describe()`` (see ``__init__.py``) - a user pointing the platform at this checkout gets the same
   descriptor without installing metadata.
2. **Descriptor** - :func:`descriptor` returns::

       {"name": "ascend_attention", "version": "0.1.0", "requires_platform": ">=0.1,<0.2",
        "ops": {"moe.experts": {"target": "ascend_attention.ops.moe.compute:fused_experts",
                                "dtypes": [...], "graph": "capture_safe", "note": "...",
                                "platform_name": "moe.experts", "impl": "torch.fused_experts",
                                "impls": [...], "measured_ratio": 1.0, "op": "moe_fused_experts"}}}

   ``target`` is ``module:attribute`` because that is what their ``ProviderOp.resolve()`` imports; the
   extra keys are ignored by their parser and kept because they are what a human debugging a binding
   wants (which implementation, what ratio, which of our op names).
3. **Binding** - :meth:`Provider.load` takes a semantic name and returns an :class:`OperatorHandle`.
   Every target resolves to a *callable*: either the algorithm function itself (MoE compute, sampling,
   layout) or a one-shot wrapper in ``entry.py`` that plans and runs in one call.

What this module does **not** do: import or name an engine, decide policy (enable/disable, ordering,
fallback), or claim an op it cannot resolve - a target that does not import is a bug here, and
``tests/test_provider.py`` resolves all of them.

The *manifest* (``manifest.json``) is the other artefact: the same op surface as pure JSON with the
platform's schema (``measured_ratio``, ``graph_capturable``, ``platform_name``). Descriptor = "what I
serve and where the code is"; manifest = "what I can serve today, with numbers". Both are published.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from . import vocabulary
from ._version import __version__
from .manifest import MANIFEST_SCHEMA_VERSION, REQUIRES_PLATFORM, manifest

__all__ = [
    "ENTRY_POINT",
    "NOTES",
    "ENTRY_POINT_GROUP",
    "ALIASES",
    "PROVIDER_GROUP",
    "PROVIDER_NAME",
    "TARGETS",
    "DescriptorError",
    "OperatorHandle",
    "Provider",
    "descriptor",
    "load",
    "provider",
]

#: The group the platform scans. Mirrors its constant (``contracts/provider.py::PROVIDER_GROUP``);
#: ``tests/test_provider.py`` fails if the two ever differ.
PROVIDER_GROUP = "ascend_platform.providers"

#: Backwards-compatible alias: the docs and the first version of this module used this name.
ENTRY_POINT_GROUP = PROVIDER_GROUP

#: Our name inside that group (one provider per operator library).
PROVIDER_NAME = "ascend_attention"

#: ``module:attribute`` as written in ``pyproject.toml``'s ``[project.entry-points]``.
ENTRY_POINT = "ascend_attention.provider:provider"

#: Semantic name -> ``module:attribute`` of the callable the platform binds.
#:
#: Hand-maintained on purpose. A generated table would hide the only decision that matters here (which
#: symbol is the entry for this op), and the failure it would hide - pointing at something that happens
#: to have the right name - is exactly the incident's shape.
#:
#: ``attention.paged_decode`` is deliberately **absent**: the kernel is device-validated and graph-safe,
#: but it has no op-level A/B (0.362x against the vendor fused op, i.e. slower), so it is not advertised
#: in the manifest and this descriptor lists exactly what the manifest lists. Publishing it would let the
#: platform bind a slower implementation than the engine's own and call it by accident - the opposite of
#: what a provider contract is for. It stays reachable through ``plan(impl="tilelang.decode950")``.
TARGETS: Mapping[str, str] = {
    "attention.paged_prefill": "ascend_attention.entry:paged_prefill",
    "gdn.gating": "ascend_attention.entry:gdn_gating",
    "gdn.conv_update": "ascend_attention.entry:gdn_conv_update",
    "gdn.chunk_wu": "ascend_attention.entry:gdn_chunk_wu",
    "moe.slot_layout": "ascend_attention.ops.moe.slot_layout:slot_layout_from_group_list",
    "moe.prenz_pack": "ascend_attention.ops.moe.prenz:pack_nd_to_compact_nz",
    "moe.routing_topk_softmax": "ascend_attention.ops.moe.compute:topk_softmax",
    "moe.align_block_size": "ascend_attention.ops.moe.compute:moe_align_block_size",
    "moe.swiglu_gate": "ascend_attention.ops.moe.compute:silu_and_mul",
    "moe.experts": "ascend_attention.ops.moe.compute:fused_experts",
    "moe.experts_unquantized": "ascend_attention.ops.moe.compute:unquantized_fused_moe",
    "sampling.topk_threshold": "ascend_attention.ops.sampling:topk_threshold",
    "sampling.greedy": "ascend_attention.ops.sampling:greedy",
    "sampling.top_p": "ascend_attention.ops.sampling:top_p_filter",
}


#: Compatibility keys the platform's current table asks for, mapped to the semantic name that owns them.
#:
#: Values are *engine-symbol shaped* because they are that platform table's existing keys, not our
#: vocabulary. They are published as aliases (``alias_of`` in the descriptor) so that table keeps working
#: while its owner moves to semantic names; when ``contracts/vocabulary.py`` lands there, delete this dict
#: and the platform loses nothing. Keeping the alias here rather than renaming our op is deliberate: a
#: semantic name is a promise about meaning, and we should not rename it to match someone's legacy key.
ALIASES: Mapping[str, str] = {
    "unquantized_fused_moe": "moe.experts_unquantized",
}


#: Extra sentences for the descriptor's ``note``, where the target's *shape* is easy to misread.
#:
#: The measured case: the platform's table assigned our ``unquantized_fused_moe`` function to a slot that
#: expects a *method object factory*, so its patch did ``subclass = impl(base)`` and reported "the target
#: vanished between planning and apply" - the target did not vanish, it was called with a class. A note is
#: cheap; a week of reading that message is not.
NOTES: Mapping[str, str] = {
    "moe.experts_unquantized": (
        "plain math: (hidden_states, w1, w2, topk_weights, topk_ids, ...) -> hidden_states. It is NOT a "
        "class factory for an engine's MoE method object; that wrapper belongs to the engine's ABI layer."
    ),
    "moe.experts": (
        "plain math, unquantised only: a non-None quant_config raises instead of silently computing in "
        "bf16."
    ),
}


class DescriptorError(RuntimeError):
    """The descriptor this library is about to publish is inconsistent."""


def _resolve(target: str) -> Callable[..., Any]:
    module_name, _, attribute = target.partition(":")
    if not module_name or not attribute:
        raise DescriptorError(f"target {target!r} is not 'module:attribute'")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - a broken target is a fact about this provider
        raise DescriptorError(f"cannot import {module_name}: {type(exc).__name__}: {exc}") from None
    try:
        value = getattr(module, attribute)
    except AttributeError:
        raise DescriptorError(f"{module_name} has no attribute {attribute!r}") from None
    if not callable(value):
        raise DescriptorError(f"{target} resolves to a non-callable")
    return value


@dataclass(frozen=True)
class OperatorHandle:
    """One semantic op: the callable the platform binds, plus the facts it needs to gate on it."""

    platform_name: str
    op: str
    impl: str
    graph: str
    measured_ratio: float
    impls: Tuple[str, ...]
    target: str
    fn: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def plan(self, **pure: Any) -> Any:
        """Build the strict plan for this op (pure data in, no device work beyond the planner)."""
        from . import plan as _plan

        return _plan(self.op, impl=self.impl, **pure)

    def run(self, plan: Any, *tensors: Any) -> Any:
        from . import run as _run

        return _run(plan, *tensors)

    def describe(self) -> Dict[str, Any]:
        return {
            "platform_name": self.platform_name,
            "op": self.op,
            "impl": self.impl,
            "impls": list(self.impls),
            "graph": self.graph,
            "measured_ratio": self.measured_ratio,
            "target": self.target,
        }


class Provider:
    """The object the entry point returns: publish the descriptor, load one op by semantic name."""

    name = PROVIDER_NAME
    group = PROVIDER_GROUP

    def describe(self) -> Dict[str, Any]:
        """The descriptor their ``parse_descriptor`` reads. Pure data; no engine names, no callables."""
        ops: Dict[str, Dict[str, Any]] = {}
        for entry in manifest()["ops"]:
            semantic = entry["platform_name"]
            try:
                target = TARGETS[semantic]
            except KeyError:
                raise DescriptorError(
                    f"{semantic} ({entry['name']}) is advertised but has no target; add it to "
                    "provider.TARGETS (an op the platform cannot resolve is worse than one it "
                    "does not see)"
                ) from None
            ops[semantic] = {
                "target": target,
                "dtypes": list(entry["dtypes"]),
                "graph": "capture_safe" if entry["graph_capturable"] else "eager_only",
                "note": " ".join(
                    part
                    for part in (f"{entry['name']} via {entry['impl']}", NOTES.get(semantic, ""))
                    if part
                ),
                # The keys below are ours: their parser ignores unknown fields, and a human (or a
                # doctor command) needs them to tell "bound" from "bound to the oracle".
                "platform_name": semantic,
                "op": entry["name"],
                "impl": entry["impl"],
                "impls": list(entry["impls"]),
                "measured_ratio": entry["measured_ratio"],
                "op_graph": entry["graph_capturable"],
            }
        for alias, canonical in ALIASES.items():
            if canonical not in ops:
                raise DescriptorError(f"alias {alias!r} points at {canonical!r}, which is not served")
            ops[alias] = {**ops[canonical], "platform_name": canonical, "alias_of": canonical}
        return {
            "name": PROVIDER_NAME,
            "version": __version__,
            "requires_platform": REQUIRES_PLATFORM,
            "schema": MANIFEST_SCHEMA_VERSION,
            "ops": ops,
        }

    def ops(self) -> Tuple[str, ...]:
        """Semantic names this provider serves (advertised ops only, in manifest order)."""
        return tuple(self.describe()["ops"])

    def load(self, platform_name: str) -> OperatorHandle:
        entries = self.describe()["ops"]
        try:
            entry = entries[platform_name]
        except KeyError:
            raise KeyError(
                f"{platform_name!r} is not provided (or not advertised) by {PROVIDER_NAME}; "
                f"available: {sorted(entries)}"
            ) from None
        target = entry["target"]
        return OperatorHandle(
            platform_name=platform_name,
            op=entry["op"],
            impl=entry["impl"],
            graph=entry["graph"],
            measured_ratio=float(entry["measured_ratio"]),
            impls=tuple(entry["impls"]),
            target=target,
            fn=_resolve(target),
        )

    # A provider is also its own factory, so ``entry.load()()`` and ``entry.load()`` both work: the
    # platform should not have to remember which of the two its loader does.
    def __call__(self) -> "Provider":
        return self

    def __repr__(self) -> str:
        return f"<Provider {PROVIDER_NAME!r} group={PROVIDER_GROUP!r} ops={len(self.ops())}>"


_PROVIDER = Provider()


def provider() -> Provider:
    """The factory named by the entry point."""
    return _PROVIDER


def descriptor() -> Dict[str, Any]:
    return _PROVIDER.describe()


def load(platform_name: str) -> OperatorHandle:
    return _PROVIDER.load(platform_name)


def _self_check() -> None:  # pragma: no cover - called from tests
    """Fail early if the published surface is inconsistent."""
    described = _PROVIDER.describe()
    for semantic, entry in described["ops"].items():
        if semantic in ALIASES:
            # A compatibility key is deliberately *not* a semantic name: it is the platform's legacy
            # symbol, published so its current table keeps working. It must still point at a real op.
            if entry.get("alias_of") not in described["ops"]:
                raise AssertionError(f"alias {semantic!r} does not point at a served op")
            continue
        if not vocabulary.NAME_PATTERN.match(semantic):
            raise AssertionError(f"{semantic!r} is not family.member")
        if vocabulary.op_for(semantic) != entry["op"]:
            raise AssertionError(f"{semantic}: vocabulary says {vocabulary.op_for(semantic)}")
        _resolve(entry["target"])
    extra = (set(TARGETS) | set(ALIASES)) - set(described["ops"])
    if extra:
        raise AssertionError(f"TARGETS names ops that are not advertised: {sorted(extra)}")
