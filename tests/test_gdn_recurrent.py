"""The two GDN recurrent entry points must agree, step for step.

``ops/fla/recurrent.py`` says this file exists ("...which is exactly what the test suite does to prove
the two agree (``tests/test_gdn_recurrent.py``)"), and it did not — until now. That matters beyond
tidiness: the *decode* form is what a serving path calls once per generated token, and it is the only
place where the gate is derived inside the operator from ``a``/``b``/``A_log``/``dt_bias`` instead of
being handed a pre-computed ``g``. A sign or a softplus on the wrong side of that derivation is a
silent wrong output, not an exception, so the equivalence has to be pinned.

The claim being tested, stated the way the docstring states it: the packed decode form with ``T`` calls
and a carried state is the sequence form over the same ``T`` tokens, with
``g = -exp(A_log) * softplus(a + dt_bias)`` and ``beta = sigmoid(b)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ascend_attention.ops.fla.recurrent import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)

torch.manual_seed(0)

B, T, H, HV, K, V = 2, 5, 2, 2, 8, 8
ROWS = 1 + B * T  # row 0 stays unused: <= 0 means "no state"


def _gate(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """The derivation the decode kernel does internally (recurrent.py:59-66)."""
    return -torch.exp(A_log)[None, :] * F.softplus(a + dt_bias[None, :])


def _inputs(dtype=torch.float32):
    q = torch.randn(B, T, H, K, dtype=dtype) * 0.5
    k = torch.randn(B, T, H, K, dtype=dtype) * 0.5
    v = torch.randn(B, T, HV, V, dtype=dtype) * 0.5
    a = torch.randn(B, T, HV, dtype=dtype)
    b = torch.randn(B, T, HV, dtype=dtype)
    A_log = torch.randn(HV, dtype=torch.float32) * 0.3
    dt_bias = torch.randn(HV, dtype=torch.float32) * 0.2
    state = torch.randn(ROWS, HV, V, K, dtype=torch.float32) * 0.1
    scale = K**-0.5
    return q, k, v, a, b, A_log, dt_bias, state, scale


def test_the_decode_form_equals_the_sequence_form_over_the_same_tokens():
    q, k, v, a, b, A_log, dt_bias, state, scale = _inputs()
    g = _gate(a, A_log, dt_bias)
    beta = torch.sigmoid(b)

    # One state row per sequence per token: sequence ``n`` owns rows 1+n*T .. n*T+T (row 0 is null).
    # The sequence form reads its initial state from the *first* row of each row-block and writes every
    # token's state; the decode form reads and writes the single row it is given, so both start from the
    # same content and are compared after the last token.
    seq_rows = torch.tensor(
        [[1 + n * T + t for t in range(T)] for n in range(B)], dtype=torch.int32
    )
    first = seq_rows[:, 0]
    last = seq_rows[:, -1]
    seq_out, seq_state = fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=state.clone(),
        inplace_final_state=True,
        ssm_state_indices=seq_rows,
    )

    # the caller's packing, quoted from the decode docstring: [q: H*K][k: H*K][v: HV*V]
    row = H * K * 2 + HV * V
    dec_state = state.clone()
    dec_out = torch.empty(B, 1, HV, V, dtype=q.dtype)
    stepper = first.clone()  # one state row per sequence, row > 0

    for t in range(T):
        mixed = torch.cat(
            [
                q[:, t].reshape(B, H * K),
                k[:, t].reshape(B, H * K),
                v[:, t].reshape(B, HV * V),
            ],
            dim=-1,
        )
        assert mixed.shape == (B, row)
        _, dec_state = fused_recurrent_gated_delta_rule_packed_decode(
            mixed,
            a[:, t],
            b[:, t],
            A_log,
            dt_bias,
            scale,
            dec_state,
            dec_out,
            stepper,
            use_qk_l2norm_in_kernel=False,
        )

    assert torch.allclose(dec_out[:, 0], seq_out[:, T - 1], atol=1e-5, rtol=1e-4), (
        "the last decode step and the sequence form's last output disagree:\n"
        f"max abs diff {(dec_out[:, 0] - seq_out[:, T - 1]).abs().max().item():.3e}"
    )
    for n in range(B):
        assert torch.allclose(dec_state[first[n]], seq_state[last[n]], atol=1e-5, rtol=1e-4), (
            f"final state of sequence {n} disagrees: max abs diff "
            f"{(dec_state[first[n]] - seq_state[last[n]]).abs().max().item():.3e}"
        )


def test_a_state_row_index_of_zero_is_a_no_op_and_writes_zeros():
    """Upstream's masked rows (recurrent.py:296-299): no state read, zeros out, state untouched."""
    q, k, v, a, b, A_log, dt_bias, state, scale = _inputs()
    before = state.clone()
    dec_out = torch.full((B, 1, HV, V), 7.0)
    mixed = torch.cat(
        [q[:, 0].reshape(B, H * K), k[:, 0].reshape(B, H * K), v[:, 0].reshape(B, HV * V)], dim=-1
    )
    _, after = fused_recurrent_gated_delta_rule_packed_decode(
        mixed,
        a[:, 0],
        b[:, 0],
        A_log,
        dt_bias,
        scale,
        state,
        dec_out,
        torch.zeros(B, dtype=torch.int32),
        use_qk_l2norm_in_kernel=False,
    )
    assert torch.count_nonzero(dec_out) == 0, "a masked row must write zeros, not keep the buffer"
    assert torch.equal(after, before), "a masked row must not touch the state"
