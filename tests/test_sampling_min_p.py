"""Min-p filtering: a relative probability floor, checked against its definition.

The rule is one sentence - keep tokens with ``prob >= min_p * max_prob`` - so the tests are about the
places that sentence is ambiguous or easy to get wrong: the comparison is inclusive (a token exactly at
the floor survives), the floor is per row (each row has its own max), p=1 degenerates to "argmax plus its
ties" rather than "keep everything" or "keep one arbitrary token", and the count is monotone in p.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ascend_attention import plan, run
from ascend_attention.ops.sampling import MINP_OP, min_p_filter, top_p_filter

ROOT = Path(__file__).resolve().parents[1]


def _oracle_mask(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return torch.ones_like(logits, dtype=torch.bool)
    probs = torch.softmax(logits.to(torch.float64), dim=-1)
    floor = probs.max(dim=-1, keepdim=True).values * p
    return probs >= floor


@pytest.mark.parametrize("p", [0.05, 0.2, 0.5, 0.9, 1.0])
def test_mask_matches_the_definition(p):
    torch.manual_seed(0)
    logits = torch.randn(4, 24, dtype=torch.float64)
    assert torch.equal(torch.isfinite(min_p_filter(logits, p)), _oracle_mask(logits, p))


def test_the_floor_is_per_row_not_global():
    """A shared max would keep almost everything in a peaked row - the classic implementation slip."""
    logits = torch.tensor(
        [[20.0, 0.0, 0.0, 0.0], [1.0, 0.9, 0.8, 0.7]],
        dtype=torch.float64,
    )
    kept = torch.isfinite(min_p_filter(logits, 0.5))
    assert kept[0].sum() == 1, "peaked row: only the max survives"
    assert kept[1].sum() > 1, "flat row: several tokens are within half the max"


def test_the_comparison_is_inclusive():
    """A token exactly at the floor must survive; '>' instead of '>=' is invisible in random tests."""
    # probs = [0.5, 0.25, 0.25] -> min_p=0.5 puts two tokens exactly at the floor
    logits = torch.log(torch.tensor([[2.0, 1.0, 1.0]], dtype=torch.float64))
    kept = torch.isfinite(min_p_filter(logits, 0.5))
    assert kept.tolist() == [[True, True, True]]


def test_p_one_keeps_the_argmax_and_its_ties():
    logits = torch.tensor([[3.0, 3.0, 0.0]], dtype=torch.float64)
    kept = torch.isfinite(min_p_filter(logits, 1.0))
    assert kept.tolist() == [[True, True, False]]


def test_the_survivor_count_is_monotone_in_p():
    torch.manual_seed(1)
    logits = torch.randn(3, 32, dtype=torch.float64)
    counts = [int(torch.isfinite(min_p_filter(logits, p)).sum(-1).sum()) for p in (0.0, 0.1, 0.3, 0.6, 1.0)]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[-1] == 3, "p=1: one survivor per row"


def test_p_zero_is_the_identity_and_does_not_alias():
    logits = torch.randn(2, 8, dtype=torch.float64)
    out = min_p_filter(logits, 0.0)
    assert torch.equal(out, logits)
    assert out.data_ptr() != logits.data_ptr()


def test_min_p_then_top_p_is_a_sensible_composition():
    """The reference stack orders min-p first; the composition must be well defined and idempotent-ish."""
    torch.manual_seed(2)
    logits = torch.randn(3, 64, dtype=torch.float64)
    after_min = min_p_filter(logits, 0.1)
    both = top_p_filter(after_min, 0.9)
    # already -inf stays -inf, and the result is at most as large as either filter alone
    assert torch.isfinite(both).sum() <= torch.isfinite(after_min).sum()
    assert torch.isfinite(both).sum() <= torch.isfinite(top_p_filter(logits, 0.9)).sum()
    assert torch.isfinite(both).any(dim=-1).all()


def test_bf16_is_computed_in_the_input_dtype():
    torch.manual_seed(3)
    logits = torch.randn(2, 128, dtype=torch.bfloat16)
    out = min_p_filter(logits, 0.3)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, min_p_filter(logits.clone(), 0.3))


def test_plan_and_run_validation():
    logits = torch.randn(2, 8, dtype=torch.bfloat16)
    p = plan(MINP_OP, impl="torch.min_p", batch=2, vocab=8, p=0.5, dtype="bfloat16", device="cpu")
    assert torch.equal(run(p, logits), min_p_filter(logits, 0.5))
    with pytest.raises(ValueError, match="p must be in"):
        plan(MINP_OP, batch=2, vocab=8, p=1.5, dtype="bfloat16", device="cpu")
    with pytest.raises(ValueError, match="dtype"):
        plan(MINP_OP, batch=2, vocab=8, p=0.5, dtype="float64", device="cpu")
    with pytest.raises(ValueError, match="logits must be"):
        run(p, torch.randn(3, 8, dtype=torch.bfloat16))


def test_manifest_entry_bench_and_ledger_exist():
    from ascend_attention import manifest as manifest_module

    entry = next(op for op in manifest_module.manifest()["ops"] if op["name"] == MINP_OP)
    assert entry["platform_name"] == "sampling.min_p"
    assert entry["measured_ratio"] == 1.0
    assert (ROOT / entry["bench"]).is_file()
    ledger = ROOT / "bench" / "results" / "sampling_min_p_b8.json"
    assert ledger.is_file()
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["ms"]["min_p"]["median_ms"] > 0
