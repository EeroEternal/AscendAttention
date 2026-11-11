"""The provider contract: discovery, descriptor, and binding by semantic name.

The incident this closes: the platform's family table imported thirteen engine names unconditionally and
bound none of them, while nothing in either repo ever asked "who provides this name?". The provider is the
answer - one entry point, a descriptor that names a callable per semantic op, and a loader. So these tests
are about the *contract*, not about the kernels behind it:

* the entry point in the shipped metadata points at something importable;
* the descriptor parses in the platform's own reader, and every target resolves to a callable;
* the advertised ops, the vocabulary and the target table are the same set;
* discovery works both ways the platform implements it (pinned package, entry point);
* nothing engine-shaped leaks into the published document.
"""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path

import pytest
import torch

from ascend_attention import manifest, provider, vocabulary

ROOT = Path(__file__).resolve().parents[1]

#: Engine packages the operator layer must not import or name. Kept here, not imported from the
#: platform: this repo should not need the platform to know what an engine is.
ENGINE_ROOTS = ("vllm", "vllm_ascend", "mindie", "mindspore", "triton", "flashinfer")


# ------------------------------------------------------------------ discovery


def test_pyproject_declares_the_entry_point_and_the_target_imports():
    """Checking the TOML (not installed metadata) keeps this honest on a PYTHONPATH checkout."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    group = data["project"]["entry-points"][provider.PROVIDER_GROUP]
    assert group == {"ascend_attention": provider.ENTRY_POINT}
    module_name, _, attribute = provider.ENTRY_POINT.partition(":")
    factory = getattr(importlib.import_module(module_name), attribute)
    assert callable(factory)
    assert factory() is provider.provider(), "the factory must be stable, not build a new provider"


def test_the_package_itself_exposes_describe_for_the_pinned_root_path():
    """The platform's first discovery route calls ``describe()`` on the pinned package, not on a module."""
    import ascend_attention

    assert callable(ascend_attention.describe)
    assert ascend_attention.describe()["name"] == provider.PROVIDER_NAME


def test_entry_point_group_matches_the_platform_constant():
    try:
        contracts = importlib.import_module("ascend_platform.contracts.provider")
    except Exception:  # pragma: no cover - platform not installed
        pytest.skip("AscendPlatform is not importable; the cross-repo half cannot be checked here")
    assert contracts.PROVIDER_GROUP == provider.PROVIDER_GROUP


def test_platform_discovery_finds_this_provider_through_the_pinned_root(monkeypatch):
    try:
        module = importlib.import_module("ascend_platform.contracts.provider")
    except Exception:  # pragma: no cover - platform not installed
        pytest.skip("AscendPlatform is not importable")
    monkeypatch.setenv("ASCEND_PLATFORM_OPS", "ascend_attention")
    module.reset_provider()
    try:
        found = module.discover()
        assert found is not None, "the platform did not see this package as a provider"
        assert found.name == provider.PROVIDER_NAME
        assert isinstance(found.version, str) and found.version
        assert set(found.op_names()) == set(provider.TARGETS) | set(provider.ALIASES)
        assert found.requires_platform
    finally:
        module.reset_provider()


def test_an_installed_distribution_exposes_exactly_this_provider():
    """Only meaningful for an installed wheel; the PYTHONPATH checkout cannot see entry points."""
    from importlib import metadata

    try:
        entries = list(metadata.entry_points(group=provider.PROVIDER_GROUP))
    except TypeError:  # pragma: no cover - very old importlib
        pytest.skip("importlib.metadata.entry_points(group=...) is unavailable")
    ours = [entry for entry in entries if entry.name == provider.PROVIDER_NAME]
    if not ours:  # pragma: no cover - not installed in this environment
        pytest.skip("ascend-attention is not installed as a distribution")
    assert ours[0].load()().describe()["name"] == provider.PROVIDER_NAME


# ------------------------------------------------------------------ descriptor


def _described() -> dict:
    return provider.descriptor()


def test_descriptor_has_the_shape_the_platform_parses():
    described = _described()
    assert isinstance(described["name"], str) and described["name"]
    assert isinstance(described["version"], str) and described["version"]
    assert isinstance(described["requires_platform"], str) and described["requires_platform"]
    assert isinstance(described["ops"], dict) and described["ops"]
    for semantic, entry in described["ops"].items():
        assert ":" in entry["target"], semantic
        assert entry["target"].startswith("ascend_attention."), entry["target"]
        assert entry["graph"] in ("capture_safe", "eager_only")


def test_descriptor_parses_in_the_platform_reader():
    try:
        from ascend_platform.contracts.provider import parse_descriptor
    except Exception:  # pragma: no cover - platform not installed
        pytest.skip("AscendPlatform is not importable")
    parsed = parse_descriptor(_described())
    assert parsed.name == provider.PROVIDER_NAME
    assert set(parsed.op_names()) == set(_described()["ops"])
    for name in parsed.op_names():
        assert callable(parsed.ops[name].resolve()), name


def test_every_target_resolves_to_a_callable():
    for semantic, target in provider.TARGETS.items():
        assert callable(provider._resolve(target)), (semantic, target)


def test_the_advertised_ops_vocabulary_and_targets_are_the_same_set():
    described = set(_described()["ops"])
    advertised = {op["platform_name"] for op in manifest.manifest()["ops"]}
    canonical = described - set(provider.ALIASES)
    assert canonical == advertised == set(provider.TARGETS)
    for semantic in canonical:
        assert vocabulary.NAME_PATTERN.match(semantic)
        assert vocabulary.op_for(semantic) in {op["name"] for op in manifest.manifest()["ops"]}


def test_compatibility_aliases_point_at_the_semantic_op_that_owns_them():
    """The platform's table still asks ``unquantized_fused_moe``; it must land on our semantic op."""
    described = _described()["ops"]
    for alias, canonical in provider.ALIASES.items():
        assert described[alias]["alias_of"] == canonical
        assert described[alias]["target"] == described[canonical]["target"]
        handle = provider.load(alias)
        assert handle.platform_name == canonical or handle.platform_name == alias
        assert callable(handle.fn)


def test_descriptor_carries_no_engine_vocabulary():
    text = json.dumps(_described()).lower()
    for engine in ENGINE_ROOTS:
        assert engine not in text, f"the descriptor names {engine}"


def test_paged_decode_is_not_published_and_that_is_a_decision():
    """It is the one op with a device-validated kernel that we still do not hand to the platform.

    Reason: no op-level A/B, and the number that does exist (0.362x vs the vendor fused op) says the
    kernel is *slower*, so binding it would make the platform pick the worse implementation by name.
    """
    assert "attention.paged_decode" not in provider.TARGETS
    assert "attention.paged_decode" not in _described()["ops"]
    with pytest.raises(KeyError, match="available"):
        provider.load("attention.paged_decode")
    # still reachable by pinning the implementation, which is how a negative result stays usable
    from ascend_attention import implementations

    assert "tilelang.decode950" in {impl.impl for impl in implementations("paged_decode_attention")}


def test_the_manifest_still_carries_platform_name_for_every_op():
    """Two artefacts, one vocabulary: the platform's manifest reader keys off the same names."""
    for op in manifest.manifest()["ops"]:
        assert op["platform_name"] in provider.TARGETS


# ------------------------------------------------------------------ load / bind


def test_load_returns_a_handle_for_every_served_name():
    described = _described()["ops"]
    for semantic in provider.provider().ops():
        handle = provider.load(semantic)
        assert handle.platform_name == semantic
        assert handle.target == described[semantic]["target"]
        assert handle.measured_ratio > 0
        assert handle.graph in ("capture_safe", "eager_only")
        assert handle.impl in handle.impls


def test_load_rejects_an_unknown_name_with_the_available_list():
    with pytest.raises(KeyError, match="available"):
        provider.load("moe.experts_v3")
    # our internal op name is not a semantic name: the platform must ask by vocabulary
    with pytest.raises(KeyError, match="available"):
        provider.load("moe_fused_experts")


def test_a_moe_handle_computes():
    handle = provider.load("moe.experts")
    torch.manual_seed(0)
    hidden, inter, experts, tokens, top_k = 4, 3, 3, 4, 1
    x = torch.randn(tokens, hidden, dtype=torch.float64)
    w1 = torch.randn(experts, 2 * inter, hidden, dtype=torch.float64)
    w2 = torch.randn(experts, hidden, inter, dtype=torch.float64)
    ids = torch.topk(torch.rand(tokens, experts), top_k, dim=-1).indices.to(torch.int32)
    weights = torch.ones(tokens, top_k, dtype=torch.float64)
    compute = importlib.import_module("ascend_attention.ops.moe.compute")
    assert torch.allclose(handle(x, w1, w2, weights, ids), compute.fused_experts(x, w1, w2, weights, ids))


def test_an_attention_handle_refuses_to_guess_head_counts():
    """A wrapper that inferred num_qo_heads from a shape would be a silently wrong number machine."""
    handle = provider.load("attention.paged_prefill")
    with pytest.raises(TypeError, match="num_qo_heads"):
        handle(torch.zeros(1), torch.zeros(1), torch.zeros(1))


def test_a_handle_also_offers_the_strict_plan_and_run_path():
    handle = provider.load("moe.align_block_size")
    plan = handle.plan(num_tokens=3, top_k=2, num_experts=4, block_size=2)
    ids = torch.tensor([[0, 1], [1, 2], [3, 0]], dtype=torch.int32)
    sorted_ids, expert_ids, total = handle.run(plan, ids)
    assert int(total) > 0 and sorted_ids.numel() == 3 * 2 + 4 * 2
    assert expert_ids.numel() >= 1


def test_engine_names_are_refused_with_the_place_to_get_them():
    """``engine.moe_method`` is the platform's ABI slot, and its own install.py passes it from there.

    Their log line for a missing slot is "provider does not serve 'engine.moe_method' (it serves 19
    op(s))", which sends the reader looking in this repo. The refusal must say where it actually lives.
    """
    with pytest.raises(KeyError, match="engine ABI namespace"):
        provider.load("engine.moe_method")
    with pytest.raises(KeyError, match="engine/abi"):
        provider.load("engine.anything")


def test_no_advertised_name_is_in_the_engine_namespace():
    """An operator library that publishes an engine.* name has started importing an engine."""
    from ascend_attention import provider as provider_module

    served = set(provider_module.descriptor()["ops"])
    assert not {name for name in served if name.startswith(provider_module.ENGINE_NAMESPACE)}
    families = {name.split(".", 1)[0] for name in served if "." in name}
    assert families <= provider_module.OPERATOR_NAMESPACES, sorted(
        families - provider_module.OPERATOR_NAMESPACES
    )


def test_the_functional_signatures_the_engine_adapter_depends_on():
    """The platform's ABI adapter calls these positionally; a signature change must fail here.

    Pinned because the adapter lives in the other repo and cannot see this one: if ``topk_softmax``
    grows a positional argument, their call site silently passes the wrong tensor. The arities below are
    the contract the e2e run was verified against (docs/e2e-qwen36.md).
    """
    import inspect

    from ascend_attention.ops.moe import compute

    def positional(fn):
        return [
            name
            for name, param in inspect.signature(fn).parameters.items()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]

    assert positional(compute.topk_softmax) == ["scores", "top_k"]
    assert positional(compute.moe_align_block_size) == ["topk_ids", "num_experts", "block_size"]
    # the engine's MoE method object calls this with exactly these five positional arguments plus
    # keyword options (``activation``/``apply_router_weight_on_input``/``global_num_experts``/
    # ``expert_map``/``quant_config``) - see their ``_MoEKernel.apply``
    assert positional(compute.fused_experts)[:5] == [
        "hidden_states",
        "w1",
        "w2",
        "topk_weights",
        "topk_ids",
    ]
    assert positional(compute.unquantized_fused_moe) == [
        "hidden_states",
        "w1",
        "w2",
        "topk_weights",
        "topk_ids",
    ], "num_experts must stay keyword-only: the engine passes five positionals"


def test_the_provider_self_check_passes():
    """One call that would have caught both incidents: unnamed op, unresolvable target, dangling alias."""
    provider._self_check()


def test_every_handle_is_callable_or_documents_what_it_needs():
    """No handle may be a dead end: each one either computes or has a docstring naming its arguments."""
    for semantic in provider.provider().ops():
        handle = provider.load(semantic)
        assert callable(handle.fn)
        assert handle.fn.__doc__, f"{semantic} target has no docstring"
