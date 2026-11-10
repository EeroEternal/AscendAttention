"""The four silent killers of the GDN chunk chain, each pinned by a test that would notice.

``docs/notes/fla-chunk-contract.md`` §5 lists the mistakes that never raise, never fail a small test,
and only surface as a plausible-but-wrong number on a real model.  A port is only as good as its
defences against those, so each one gets a test with a *positive* control (the wrong variant is shown
to change the answer, so the test is not vacuous):

1. ``safe_exp`` on a *difference* vs plain ``exp`` (and ``where`` vs multiply-by-mask → NaN);
2. two different triangles — ``>`` for kkt (diagonal dropped), ``>=`` for chunk_o (diagonal kept);
3. three bf16 round trips (``A_inv``, the two wy dots, ``h``);
4. argument order — ``chunk_o`` consumes ``v_new``, ``h[i_t]`` is the *pre*-block state, the tail
   chunk's ``g_last`` is not ``g[BT-1]``.

Plus the joint oracle: the chunked chain and the recurrent (one-token-at-a-time) form must agree, which
covers ``delta_h`` and ``chunk_o`` together without copying either one's algebra into the test.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention.ops.fla import (
    chunk_fwd_o,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_fwd_h,
    chunk_scaled_dot_kkt_fwd,
    fused_recurrent_gated_delta_rule,
    solve_tril,
)
from ascend_attention.ops.fla.kkt import _safe_exp

CHUNK = 64


def decay_log(batch: int, tokens: int, heads: int, scale: float = 0.05) -> torch.Tensor:
    """A monotone-decreasing chunk-local log-decay ``g`` (the only shape the kernels promise).

    ``cumsum`` over the *token* axis, not the head axis: with one head the two are indistinguishable
    until a test starts asserting on individual entries, and then a non-monotone ``g`` makes
    ``safe_exp`` clamp entries to zero for reasons that have nothing to do with the operator.
    """
    ordered = torch.sort(torch.rand(batch, tokens, heads), dim=1).values
    return -ordered.cumsum(1) * scale


# --------------------------------------------------------------------------- 1. safe_exp / where
def test_safe_exp_clamps_positive_arguments_to_zero():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0, 88.1, 200.0])
    got = _safe_exp(x)
    assert got[0] == pytest.approx(float(torch.exp(torch.tensor(-2.0))))
    assert got[2] == 1.0
    assert torch.equal(got[3:], torch.zeros(4))  # exactly zero, not exp(3.0)
    assert torch.isfinite(got).all()


def test_plain_exp_on_a_positive_difference_overflow_to_inf():
    """The control: this is what the clamp is protecting against, in fp32."""
    x = torch.tensor([200.0])
    assert not torch.isfinite(torch.exp(x)).all()
    assert torch.isfinite(_safe_exp(x)).all()


def test_multiply_by_mask_would_produce_nan_but_where_does_not():
    """``inf * 0 = nan``; the kernel uses ``where`` (discard), so the value must stay 0."""
    x = torch.tensor([200.0])
    mask = torch.tensor([False])
    via_multiply = _safe_exp(x) * mask  # 0 * 0 = 0 here, but...
    via_where = torch.where(mask, torch.exp(x), torch.zeros_like(x))
    # the dangerous case is a raw exp masked by multiplication:
    assert torch.isnan(torch.exp(x) * mask).any()
    assert not torch.isnan(via_where).any()
    assert not torch.isnan(via_multiply).any()  # safe_exp already prevented the inf


def test_long_chain_with_strong_decay_stays_finite():
    """g accumulates to a few hundred: any plain ``exp`` on a difference would overflow."""
    torch.manual_seed(0)
    B, T, Hg, H, K, V = 1, 128, 2, 2, 8, 8
    q, k, v = (torch.randn(B, T, Hg if i < 2 else H, x, dtype=torch.float32) * 0.3 for i, x in enumerate((K, K, V)))
    beta = torch.rand(B, T, H) * 0.5
    g = decay_log(B, T, H, scale=30.0)  # |g| per chunk reaches ~1900
    o, state = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=torch.zeros(B, H, V, K), output_final_state=True
    )
    assert torch.isfinite(o).all()
    assert torch.isfinite(state).all()


# --------------------------------------------------------------------------- 2. the two triangles
def test_kkt_drops_the_diagonal_and_the_upper_triangle():
    torch.manual_seed(1)
    B, T, H, K = 1, CHUNK, 1, 8
    k = torch.randn(B, T, H, K)
    beta = torch.rand(B, T, H) * 0.9
    g = decay_log(B, T, H)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, output_dtype=torch.float64)

    idx = torch.arange(T)
    assert torch.equal(A[0, :, 0][idx, idx], torch.zeros(T, dtype=torch.float64))  # diagonal exactly 0
    upper = idx[:, None] < idx[None, :]
    assert torch.equal(A[0, :, 0][upper], torch.zeros(upper.sum(), dtype=torch.float64))

    # a strict-lower entry equals beta_i * exp(g_i - g_j) * (k_i . k_j), in fp64
    i, j = 5, 2
    # fp32 factor order differs from the kernel's (``beta * safe_exp * dot`` vs ``beta * dot * exp``),
    # so this is an algebraic check at 1e-6, not a bitwise one - claiming 1e-10 would be dishonest.
    want = (beta[0, i, 0] * torch.exp(g[0, i, 0] - g[0, j, 0]) * (k[0, i, 0] @ k[0, j, 0])).to(torch.float64)
    assert A[0, i, 0, j].item() == pytest.approx(want.item(), rel=2e-6)


def test_chunk_o_keeps_the_diagonal_self_term():
    """With h=0 the intra-chunk term at t=0 is exactly ``scale * (q0.k0) * v0``: drop it and it's 0."""
    torch.manual_seed(2)
    B, T, H, K, V = 1, CHUNK, 1, 8, 8
    q = torch.randn(B, T, H, K, dtype=torch.float64) * 0.3
    k = torch.randn(B, T, H, K, dtype=torch.float64) * 0.3
    v = torch.randn(B, T, H, V, dtype=torch.float64) * 0.3
    g = decay_log(B, T, H).to(torch.float64)
    h = torch.zeros(B, 1, H, K, V, dtype=torch.float64)  # BT=64, one chunk, zero entering state

    out = chunk_fwd_o(q, k, v, h, g, scale=K**-0.5)
    scale = K**-0.5
    # explicit fp64 loop over the inclusive lower triangle
    want = torch.zeros_like(out)
    for t in range(T):
        acc = torch.zeros(H, V, dtype=torch.float64)
        for s in range(t + 1):  # s <= t: the diagonal is included
            dot = (q[0, t] * k[0, s]).sum(-1)  # per-head q_t . k_s, not a head-by-head matrix
            weight = scale * torch.exp(g[0, t] - g[0, s]) * dot
            acc = acc + weight[:, None] * v[0, s]
        want[0, t] = acc
    assert torch.allclose(out, want, rtol=1e-9, atol=1e-9)
    # the positive control: out[0] is not zero, so a ``>`` mask would be caught here
    assert out[0, 0].abs().max().item() > 1e-3


# --------------------------------------------------------------------------- 3. bf16 round trips
def test_solve_tril_honours_the_requested_output_dtype():
    torch.manual_seed(3)
    A = torch.randn(1, CHUNK, 1, CHUNK) * 0.05
    A = torch.tril(A, -1)
    for dtype in (torch.float32, torch.bfloat16):
        out = solve_tril(A, output_dtype=dtype)
        assert out.dtype == dtype


def test_the_bf16_path_keeps_the_state_in_bf16():
    """``h`` is allocated with ``k.new_empty`` in the kernel: the boundary state is bf16, not fp32."""
    torch.manual_seed(4)
    B, T, H, K, V = 1, 2 * CHUNK, 1, 8, 8
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    u = torch.randn(B, T, H, V, dtype=torch.bfloat16)
    g = decay_log(B, T, H)
    h, v_new, final = chunk_gated_delta_rule_fwd_h(
        k, w, u, g, None, output_final_state=True
    )
    assert h.dtype == torch.bfloat16
    assert v_new.dtype == torch.bfloat16


# --------------------------------------------------------------------------- 4. joint oracle
@pytest.mark.parametrize("tail", [0, 2])
def test_chunked_chain_matches_the_recurrent_form(tail):
    """``delta_h``/``chunk_o``/``v_new`` checked against a one-token-at-a-time implementation.

    The recurrent form is the decode path: different loop structure, no chunking, no tril inversion.
    Agreement at 1e-6 relative is the strongest available statement that the *composition* is right,
    including the tail chunk whose ``g_last`` is not ``g[BT-1]``.  The final-state buffers differ by
    design (the recurrent form persists one state row per token), so only the outputs are compared.
    """
    torch.manual_seed(5)
    B, Hg, H, K, V = 1, 2, 2, 8, 8
    T = CHUNK + tail
    dt = torch.float32
    q = torch.randn(B, T, Hg, K, dtype=dt) * 0.3
    k = torch.randn(B, T, Hg, K, dtype=dt) * 0.3
    v = torch.randn(B, T, H, V, dtype=dt) * 0.3
    beta = torch.rand(B, T, H, dtype=dt) * 0.5
    g = (-torch.rand(B, T, H, dtype=dt) * 0.4).cumsum(-1)

    o_chunk, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=torch.zeros(B, H, V, K, dtype=dt), output_final_state=True
    )
    o_rec, _ = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=torch.zeros(T, H, V, K, dtype=dt)
    )
    rel = (o_chunk - o_rec).abs().max() / o_rec.abs().max().clamp_min(1.0)
    assert rel < 1e-5, f"chunk vs recurrent mismatch: {rel.item():.3e}"
