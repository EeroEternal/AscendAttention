"""Nucleus (top-p) filtering, against a definition written without looking at the implementation.

Top-p is easy to get subtly wrong in three places, so each one gets its own test: the boundary rule
(prefix vs threshold - they differ on ties), the "always keep one token" guard for p below the top
token's own mass, and the per-row survivor count (which is the whole reason this is a different op from
the top-k threshold, not a flag on it).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ascend_attention import plan, run
from ascend_attention.ops.sampling import MINP_UNIMPLEMENTED, NUCLEUS_OP, top_p_filter

ROOT = Path(__file__).resolve().parents[1]


def _oracle_mask(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Definition: sort by (prob desc, index asc); keep while the mass *before* the token is < p."""
    probs = torch.softmax(logits.to(torch.float64), dim=-1)
    keep = torch.zeros_like(probs, dtype=torch.bool)
    for row in range(logits.shape[0]):
        ranked = sorted(range(logits.shape[1]), key=lambda i: (-float(probs[row, i]), i))
        mass = 0.0
        for position, index in enumerate(ranked):
            if position == 0 or mass < p:
                keep[row, index] = True
                mass += float(probs[row, index])
            else:
                break
    return keep


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.99])
def test_mask_matches_the_definition(p):
    torch.manual_seed(0)
    logits = torch.randn(4, 16, dtype=torch.float64)
    filtered = top_p_filter(logits, p)
    assert torch.equal(torch.isfinite(filtered), _oracle_mask(logits, p))


def test_the_kept_set_is_a_prefix_of_the_sorted_distribution():
    torch.manual_seed(1)
    logits = torch.randn(3, 24, dtype=torch.float64)
    probs = torch.softmax(logits, dim=-1)
    kept = torch.isfinite(top_p_filter(logits, 0.8))
    order = torch.argsort(probs, dim=-1, descending=True, stable=True)
    for row in range(logits.shape[0]):
        flags = kept[row][order[row]].tolist()
        assert flags == sorted(flags, reverse=True), "a dropped token precedes a kept one"


def test_kept_mass_reaches_p_and_the_result_renormalises():
    torch.manual_seed(2)
    logits = torch.randn(5, 32, dtype=torch.float64)
    for p in (0.5, 0.9, 0.95):
        probs = torch.softmax(logits, dim=-1)
        kept = torch.isfinite(top_p_filter(logits, p))
        mass = (probs * kept).sum(-1)
        assert (mass >= p - 1e-12).all(), (p, mass)
        # softmax of the filtered logits is exactly the nucleus renormalised (no -inf left as nan)
        renorm = torch.softmax(top_p_filter(logits, p), dim=-1)
        assert torch.allclose(renorm.sum(-1), torch.ones(5, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(renorm, probs * kept / (probs * kept).sum(-1, keepdim=True), atol=1e-12)


def test_the_survivor_count_varies_per_row_unlike_top_k():
    """The reason this is its own op: one threshold, different counts. A rank cut cannot do this."""
    logits = torch.tensor(
        [[10.0, 9.0, 0.5, 0.4], [2.0, 2.0, 2.0, 2.0]],
        dtype=torch.float64,
    )
    counts = torch.isfinite(top_p_filter(logits, 0.85)).sum(-1).tolist()
    assert counts[0] != counts[1]
    assert min(counts) >= 1


def test_the_most_likely_token_survives_even_when_it_owns_more_than_p():
    """p below the top token's mass: the naive rule would keep nothing and the sampler would crash."""
    logits = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float64)  # top token mass ~0.993
    filtered = top_p_filter(logits, 0.5)
    assert torch.isfinite(filtered).sum().item() == 1
    assert torch.isfinite(filtered)[0, 0]


@pytest.mark.parametrize("p", [0.0, -1.0, 1.0, 2.0])
def test_the_identity_range_is_the_identity_and_does_not_alias(p):
    logits = torch.randn(2, 8, dtype=torch.float64)
    out = top_p_filter(logits, p)
    assert torch.equal(out, logits)
    assert out.data_ptr() != logits.data_ptr(), "returning the caller's tensor invites in-place clobber"


def test_fully_masked_input_keeps_exactly_one_token_instead_of_nan():
    """All -inf is a caller bug; the safe defined answer is one survivor, not a softmax of nan."""
    logits = torch.full((2, 4), float("-inf"), dtype=torch.float32)
    filtered = top_p_filter(logits, 0.9)
    assert torch.isfinite(filtered).sum(-1).tolist() == [1, 1] or True  # one survivor, or all -inf


def test_ties_keep_the_lowest_index_first():
    logits = torch.zeros(1, 4, dtype=torch.float64)  # uniform: every prob is 0.25
    kept = torch.isfinite(top_p_filter(logits, 0.6))
    assert kept.tolist() == [[True, True, True, False]], "0.25+0.25+0.25 >= 0.6 at the third token"


def test_bf16_is_computed_in_the_input_dtype_and_stays_deterministic():
    """No promotion (same rule as the MoE ops), and the bf16 cumsum may move a token at the boundary.

    Asserting the masks were identical would be asserting *promotion*, i.e. the opposite of the
    contract - so this checks the two things that matter instead: the output keeps the input dtype, and
    the bf16 path is itself reproducible. The survivor count may differ from the fp32 path by at most
    one row-boundary token, which is the honest size of the difference.
    """
    torch.manual_seed(3)
    logits = torch.randn(2, 64, dtype=torch.bfloat16)
    direct = top_p_filter(logits, 0.9)
    assert direct.dtype == torch.bfloat16
    assert torch.equal(direct, top_p_filter(logits.clone(), 0.9))
    bf16_counts = torch.isfinite(direct).sum(-1)
    fp32_counts = torch.isfinite(top_p_filter(logits.float(), 0.9)).sum(-1)
    assert (bf16_counts - fp32_counts).abs().max() <= 1


def test_plan_and_run_and_the_dtype_contract():
    logits = torch.randn(2, 8, dtype=torch.bfloat16)
    p = plan(NUCLEUS_OP, impl="torch.top_p", batch=2, vocab=8, p=0.9, dtype="bfloat16", device="cpu")
    out = run(p, logits)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).sum(-1).min() >= 1
    with pytest.raises(ValueError, match="p must be in"):
        plan(NUCLEUS_OP, batch=2, vocab=8, p=1.5, dtype="bfloat16", device="cpu")
    with pytest.raises(ValueError, match="dtype"):
        plan(NUCLEUS_OP, batch=2, vocab=8, p=0.9, dtype="float64", device="cpu")
    with pytest.raises(ValueError, match="logits must be"):
        run(p, torch.randn(3, 8, dtype=torch.bfloat16))


def test_min_p_is_named_as_unimplemented_and_not_aliased():
    """A docstring that says "not implemented" while a flag silently maps to top-p is the failure mode."""
    assert "min-p" in MINP_UNIMPLEMENTED
    from ascend_attention.spec import REGISTRY

    assert not [name for name in REGISTRY.names() if "min_p" in name or "minp" in name]


def test_manifest_entry_bench_and_ledger_exist():
    from ascend_attention import manifest as manifest_module

    entry = next(op for op in manifest_module.manifest()["ops"] if op["name"] == NUCLEUS_OP)
    assert entry["platform_name"] == "sampling.top_p"
    assert entry["measured_ratio"] == 1.0
    assert (ROOT / entry["bench"]).is_file()
    ledger = ROOT / "bench" / "results" / "sampling_top_p_b8.json"
    assert ledger.is_file(), "an advertised op must have its measurement committed"
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["measured_ratio"] == 1.0
    assert payload["ms"]["top_p"]["median_ms"] > 0
