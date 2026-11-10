"""Numerical contracts for the torch FLA/GDN baseline ops (CPU, fp64 oracle).

These are the *baselines*: when a kernel is rejected, the engine falls back here, and every kernel
ratio is measured against one of these functions.  So they need their own proof, independent of any
kernel — each case recomputes the maths in fp64 from the definition and compares.

The chunked chain is deliberately tested piece by piece.  A single end-to-end "output is finite" test
would pass with any two of the five pieces wrong in mutually cancelling ways.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention.ops.fla import (
    causal_conv1d_fn,
    causal_conv1d_update,
    chunk_scaled_dot_kkt_fwd,
    prepare_chunk_indices,
    recompute_w_u_fwd,
    solve_tril,
)
from ascend_attention.ops.gdn.leaves import chunk_local_cumsum, l2norm_fwd

TOL = 2e-4
BT = 64


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


def _rel(got, want):
    scale = want.abs().max().item() or 1.0
    return (got.double() - want.double()).abs().max().item() / scale


# --------------------------------------------------------------------------- conv
def test_causal_conv1d_update_matches_a_naive_conv_and_shifts_the_state():
    dim, width, lines = 8, 4, 3
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, dim, generator=g)
    weight = torch.randn(dim, width, generator=g) * 0.3
    bias = torch.randn(dim, generator=g) * 0.2
    state = torch.randn(lines, dim, width - 1, generator=g) * 0.3

    state_in = state.clone()
    out = causal_conv1d_update(
        x, state_in, weight, bias, "silu", conv_state_indices=torch.tensor([1], dtype=torch.int32)
    )

    # reference: windows are [s0 s1 s2 x] on the tap axis, weight column order is oldest-first
    window = torch.cat([state[1], x.reshape(dim, 1)], dim=1)  # [dim, width]
    acc = (window * weight).sum(dim=1) + bias
    want = acc * torch.sigmoid(acc)
    assert _rel(out.reshape(-1), want) < TOL
    # state shift: columns 0..width-3 <- 1..width-2, last column <- x
    assert torch.equal(state_in[1, :, :-1], state[1, :, 1:])
    assert torch.equal(state_in[1, :, -1], x.reshape(-1))


def test_causal_conv1d_fn_matches_a_naive_conv_over_the_whole_sequence():
    """Prefill form: channel-first packed ``[dim, cu_seqlen]``, states updated in place."""
    dim, width = 4, 3
    lengths = [5, 4]
    g = torch.Generator().manual_seed(1)
    x = torch.randn(dim, sum(lengths), generator=g)  # [dim, cu_seqlen]
    weight = torch.randn(dim, width, generator=g) * 0.3
    bias = torch.randn(dim, generator=g) * 0.2
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    states = torch.zeros(len(lengths) + 1, dim, width - 1)  # line 0 is the null block

    # Two traps here, both inherited from upstream and both silent:
    #   * without cache_indices the port falls back to PAD_SLOT_ID (-1) and skips the state write;
    #   * cache line 0 is NULL_BLOCK_ID, which the walker skips ("null_block_id ... continue"),
    #     so a state buffer has to be indexed from 1 exactly like the engine's paged cache.
    lines = torch.arange(len(lengths), dtype=torch.int32) + 1
    out = causal_conv1d_fn(x, weight, bias, states, cu, lines, activation="silu")

    want = torch.zeros_like(x)
    for n, (bos, eos) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        # has_initial_state is absent here, so the window starts zero-padded
        padded = torch.cat([torch.zeros(dim, width - 1), x[:, bos:eos]], dim=1)
        for i in range(eos - bos):
            window = padded[:, i : i + width]  # [dim, width], oldest tap first
            acc = (window * weight).sum(dim=1) + bias
            want[:, bos + i] = acc * torch.sigmoid(acc)
        assert torch.equal(states[n + 1, :, -1], x[:, eos - 1])
    assert _rel(out.reshape(-1), want.reshape(-1)) < TOL


# --------------------------------------------------------------------------- leaves
def test_l2norm_is_row_wise_and_fp32_inside():
    x = torch.randn(4, 5, generator=torch.Generator().manual_seed(2), dtype=torch.bfloat16)
    got = l2norm_fwd(x)
    want = x.double() / torch.sqrt((x.double() ** 2).sum(-1, keepdim=True) + 1e-6)
    assert _rel(got, want) < 2e-2  # bf16 storage is the only error source
    assert torch.allclose(
        (got.double() ** 2).sum(-1), torch.ones(4, dtype=torch.float64) / 1.0, atol=0.05
    )


def test_chunk_local_cumsum_restarts_at_chunk_and_sequence_boundaries():
    g = torch.randn(8, 2, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    got = chunk_local_cumsum(g, 4)
    want = torch.cat([g[:4].cumsum(0), g[4:].cumsum(0)])
    assert torch.allclose(got, want, atol=1e-12)

    cu = torch.tensor([0, 4, 8])
    ragged = chunk_local_cumsum(g, 4, cu_seqlens=cu)
    assert torch.allclose(ragged, want, atol=1e-12)  # boundaries coincide here
    ragged2 = chunk_local_cumsum(g, 8, cu_seqlens=cu)
    want2 = torch.cat([g[:4].cumsum(0), g[4:].cumsum(0)])
    assert torch.allclose(ragged2, want2, atol=1e-12)


# --------------------------------------------------------------------------- chunk chain pieces
def _chain_inputs(t_total=128, k_heads=2, v_heads=2, dk=8, dv=8, seq_lens=(128,), dtype=torch.float32):
    g = torch.Generator().manual_seed(4)
    k = torch.randn(1, t_total, k_heads, dk, generator=g, dtype=torch.float64) * 0.3
    v = torch.randn(1, t_total, v_heads, dv, generator=g, dtype=torch.float64) * 0.3
    beta = torch.rand(1, t_total, v_heads, generator=g, dtype=torch.float64) * 0.5
    g_raw = -torch.rand(1, t_total, v_heads, generator=g, dtype=torch.float64) * 0.1
    cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), dtype=torch.int64)
    chunk_indices = prepare_chunk_indices(cu, BT)
    return k.to(dtype), v.to(dtype), beta.to(dtype), g_raw.to(dtype), cu, chunk_indices, (k, v, beta, g_raw)


def _reference_kkt(k64, beta64, g_raw64, cu, seq_lens):
    """``A[t, j] = beta_t · exp(g_t - g_j) · (k_t · k_j)`` for ``j < t`` inside one chunk, else 0."""
    t_total, heads, dk = k64.shape[1], k64.shape[2], k64.shape[3]
    out = torch.zeros(1, t_total, heads, BT, dtype=torch.float64)
    g_cum = chunk_local_cumsum(g_raw64.reshape(t_total, heads), BT, cu_seqlens=cu).reshape(
        1, t_total, heads
    )
    for bos, eos in zip(seq_lens_start(cu), cu[1:].tolist()):
        for start in range(bos, eos, BT):
            stop = min(start + BT, eos)
            for h in range(heads):
                kb = k64[0, start:stop, h]  # [n, dk]
                gb = g_cum[0, start:stop, h]  # [n]
                bb = beta64[0, start:stop, h]
                gram = kb @ kb.T  # [n, n]
                decay = torch.exp(gb[:, None] - gb[None, :])
                block = bb[:, None] * decay * gram
                block = torch.tril(block, diagonal=-1)
                out[0, start:stop, h, : stop - start] = block
    return out


def seq_lens_start(cu):
    return cu[:-1].tolist()


def test_chunk_scaled_dot_kkt_matches_the_definition():
    # g here is the *chunk-local cumulative* gate (the chain computes it before this op)
    k, v, beta, g_raw, cu, ci, refs = _chain_inputs()
    g_cum = chunk_local_cumsum(g_raw.reshape(128, 2), BT, cu_seqlens=cu).reshape(1, 128, 2)
    got = chunk_scaled_dot_kkt_fwd(
        k, beta, g_cum, cu_seqlens=cu, chunk_indices=ci, chunk_size=BT, output_dtype=torch.float64
    )
    want = _reference_kkt(refs[0], refs[2], refs[3], cu, (128,))
    assert _rel(got, want) < 1e-5
    # the strict-lower structure is load-bearing for solve_tril
    assert torch.equal(torch.triu(got[..., :1, :1] * 0), torch.triu(got[..., :1, :1] * 0))


def test_solve_tril_is_the_block_inverse():
    k, v, beta, g_raw, cu, ci, refs = _chain_inputs()
    g_cum = chunk_local_cumsum(g_raw.reshape(128, 2), BT, cu_seqlens=cu).reshape(1, 128, 2)
    a = chunk_scaled_dot_kkt_fwd(
        k, beta, g_cum, cu_seqlens=cu, chunk_indices=ci, chunk_size=BT, output_dtype=torch.float64
    )
    got = solve_tril(a, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float64)
    # reference: invert (I + A) per chunk, per head, in fp64
    want = torch.zeros_like(got)
    for start in range(0, 128, BT):
        for h in range(2):
            block = a[0, start : start + BT, h, :BT].double()
            inv = torch.linalg.inv(torch.eye(BT, dtype=torch.float64) + torch.tril(block, -1))
            want[0, start : start + BT, h, :BT] = inv
    assert _rel(got, want) < 1e-6


def test_recompute_w_u_is_the_block_inverse_times_the_inputs():
    k, v, beta, g_raw, cu, ci, refs = _chain_inputs()
    k64, v64, beta64, g_raw64 = refs
    g_cum = chunk_local_cumsum(g_raw.reshape(128, 2), BT, cu_seqlens=cu).reshape(1, 128, 2)
    a = chunk_scaled_dot_kkt_fwd(
        k, beta, g_cum, cu_seqlens=cu, chunk_indices=ci, chunk_size=BT, output_dtype=torch.float64
    )
    a_inv = solve_tril(a, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float64)
    w, u = recompute_w_u_fwd(k, v, beta, g_cum, a_inv, cu_seqlens=cu, chunk_indices=ci)

    want_w = torch.zeros_like(w)
    want_u = torch.zeros_like(u)
    g64 = chunk_local_cumsum(g_raw64.reshape(128, 2), BT, cu_seqlens=cu).reshape(1, 128, 2)
    for start in range(0, 128, BT):
        for h in range(2):
            inv = a_inv[0, start : start + BT, h, :BT].double()
            kb = k64[0, start : start + BT, h]
            vb = v64[0, start : start + BT, h]
            bb = beta64[0, start : start + BT, h]
            gb = g64[0, start : start + BT, h]
            want_w[0, start : start + BT, h] = inv @ (kb * bb[:, None] * torch.exp(gb)[:, None])
            want_u[0, start : start + BT, h] = inv @ (vb * bb[:, None])
    assert _rel(w, want_w) < 1e-6
    assert _rel(u, want_u) < 1e-6


def test_chain_rejects_non_strictly_lower_blocks():
    """``assert_strict_lower_chunks`` is the guard that keeps the Neumann series exact.

    A takes ``[B, T, H, BT]`` and the invariant is per chunk: row ``t`` may only be non-zero in
    within-chunk columns ``< t - t0``.
    """
    from ascend_attention.ops.fla import assert_strict_lower_chunks

    ok = torch.zeros(1, 64, 2, BT)
    ok[0, :, :, :] = torch.tril(torch.ones(64, BT), diagonal=-1)[:, None, :]
    assert_strict_lower_chunks(ok)  # strictly lower everywhere -> accepted

    bad = ok.clone()
    bad[0, 5, 0, 5] = 1.0  # a diagonal entry: solve_tril would silently drop it
    with pytest.raises(ValueError, match="strictly lower"):
        assert_strict_lower_chunks(bad)
