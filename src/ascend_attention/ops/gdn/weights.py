"""GDN input-projection weight shaping: fuse ``in_proj_ba`` into ``in_proj_qkvz``, then fix the row order.

Both steps are **load-time weight transforms, not runtime operators**, so they live here as pure
functions rather than in the registry: there is no shape to dispatch on, no device, and nothing to
measure.  What they do need is a proof, and the proof is exact: a GEMM accumulates over ``K`` in the
same order for every column, so permuting output columns (rows of the weight) is a bitwise identity,
not an approximation.  ``tests/test_gdn_weights.py`` checks it with ``torch.equal``.

Why fuse at all (TileInfer E100/E99): decode runs two skinny GEMMs per GDN layer, ``in_proj_qkvz``
(M=16, K=2048, N=12288) and ``in_proj_ba`` (M=16, K=2048, N=64).  The second is 0.26 MB — pure
kernel-launch tax, ~3.8 µs per call in-engine ≈ 0.67 % of TPOT.  Catting it onto the tail of the
first makes it one GEMM (N=12352).  Why then reorder rows (H34): the fused output still has to go
through ``fix_query_key_value_ordering`` (view + split + reshape) and three ``rearrange`` calls plus a
``cat``, every step a real copy.  Reordering the *rows* at load time makes a single GEMM's output
columns already be the consumption order, so the forward path only needs zero-copy slices.

Guards that stay at the call site (they are engine facts, not weight facts): TP=1, no quantisation,
interleaved layout, both projections plain linear weights, and ``v_heads % k_heads == 0``.
"""

from __future__ import annotations

import torch

__all__ = [
    "apply_row_order",
    "fuse_inproj_weight",
    "reorder_rows_plan",
    "split_interleaved",
]


def fuse_inproj_weight(qkvz: torch.Tensor, ba: torch.Tensor) -> torch.Tensor:
    """``cat([qkvz, ba], dim=0)`` with the two shapes actually checked.

    The check is not ceremony: a silent mismatch here produces a weight whose first ``n_qkvz`` rows are
    right and whose tail is garbage, i.e. every ``b``/``a`` gate wrong while the layer still runs.
    """
    if qkvz.ndim != 2 or ba.ndim != 2:
        raise ValueError(f"expected 2-D weights, got {tuple(qkvz.shape)} and {tuple(ba.shape)}")
    if qkvz.shape[1] != ba.shape[1]:
        raise ValueError(
            f"hidden size mismatch: qkvz K={qkvz.shape[1]} vs ba K={ba.shape[1]}"
        )
    if qkvz.dtype != ba.dtype:
        raise ValueError(f"dtype mismatch: {qkvz.dtype} vs {ba.dtype}")
    return torch.cat([qkvz, ba], dim=0)


def split_interleaved(
    num_k_heads: int, num_v_heads: int, head_k_dim: int, head_v_dim: int
) -> torch.Tensor:
    """Rows of the fused weight in **consumption order**, expressed with reshapes only.

    This is deliberately an independent formulation of the same mapping ``reorder_rows_plan`` builds
    with Python loops: one of the two is wrong if they disagree, and it is much more likely that a loop
    index is wrong than that a ``view``/``reshape`` pair is.

    Interleaved source layout (checkpoint order), per key-head group ``g``::

        [g0: q(dk) k(dk) v(vpg*dv) z(vpg*dv)] [g1: ...] ... ||  [g0: b(vpg) a(vpg)] [g1: ...] ...

    Consumption order::

        [q_all | k_all | v_all | z_all || b_all | a_all]   (group-major flat inside each part)
    """
    if num_k_heads <= 0 or num_v_heads <= 0:
        raise ValueError("invalid head config")
    if num_v_heads % num_k_heads != 0:
        raise ValueError("v_heads must be a multiple of k_heads (interleaved layout)")
    ng = num_k_heads
    vpg = num_v_heads // num_k_heads
    group = 2 * head_k_dim + 2 * vpg * head_v_dim
    n_qkvz = ng * group
    n_ba = ng * 2 * vpg

    qkvz = torch.arange(n_qkvz).view(ng, group)
    ba = torch.arange(n_qkvz, n_qkvz + n_ba).view(ng, 2 * vpg)
    parts = [
        qkvz[:, 0:head_k_dim],
        qkvz[:, head_k_dim : 2 * head_k_dim],
        qkvz[:, 2 * head_k_dim : 2 * head_k_dim + vpg * head_v_dim],
        qkvz[:, 2 * head_k_dim + vpg * head_v_dim :],
        ba[:, :vpg],
        ba[:, vpg:],
    ]
    return torch.cat([part.reshape(-1) for part in parts])


def reorder_rows_plan(
    num_k_heads: int, num_v_heads: int, head_k_dim: int, head_v_dim: int, tp_size: int = 1
) -> torch.Tensor:
    """``plan[i]`` = the source row that becomes row ``i`` (TileInfer H34, TP-aware).

    Kept as the loop form on purpose: this is the code that ships, and
    ``tests/test_gdn_weights.py`` cross-checks it against :func:`split_interleaved` for every small
    head configuration instead of trusting one derivation.
    """
    if num_k_heads <= 0 or num_v_heads <= 0 or tp_size <= 0:
        raise ValueError("invalid head config")
    if num_v_heads % num_k_heads != 0 or num_k_heads % tp_size != 0:
        raise ValueError("head counts not divisible (interleaved layout)")
    ng = num_k_heads // tp_size
    vpg = num_v_heads // num_k_heads
    dk, dv = head_k_dim, head_v_dim
    group = 2 * dk + 2 * vpg * dv
    n_qkvz = ng * group

    qkvz: list[int] = []
    for lo in range(0, 2 * dk, dk):  # q_all, then k_all
        for g in range(ng):
            base = g * group + lo
            qkvz.extend(range(base, base + dk))
    for lo in (2 * dk, 2 * dk + vpg * dv):  # v_all, then z_all
        for g in range(ng):
            base = g * group + lo
            qkvz.extend(range(base, base + vpg * dv))

    ba: list[int] = []
    for g in range(ng):  # b_all
        for j in range(vpg):
            ba.append(n_qkvz + g * 2 * vpg + j)
    for g in range(ng):  # a_all
        for j in range(vpg):
            ba.append(n_qkvz + g * 2 * vpg + vpg + j)
    return torch.tensor(qkvz + ba, dtype=torch.long)


def apply_row_order(weight: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
    """``weight[plan]`` — an exact row permutation, so the result is bitwise, not approximately, equal."""
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(weight.shape)}")
    if plan.numel() != weight.shape[0]:
        raise ValueError(
            f"plan has {plan.numel()} entries for a weight with {weight.shape[0]} rows"
        )
    if not torch.equal(torch.sort(plan).values, torch.arange(weight.shape[0])):
        raise ValueError("plan is not a permutation of the weight rows")
    return weight[plan.to(weight.device)].contiguous()
