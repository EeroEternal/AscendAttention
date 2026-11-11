"""One-shot entries: plan and run in a single call, for callers that hold tensors and nothing else.

Why this module exists: the provider contract (``provider.py``) publishes a *callable* per semantic op,
and the platform resolves it as ``module:attribute`` - it never sees our ``plan``/``run`` split. For ops
whose implementation already is a tensor function (the MoE compute baselines, sampling) the target is
that function. For the plan-driven ops (attention, GDN) the target is one of the wrappers below: they
build the plan from the caller's pure-data arguments and execute immediately.

Two rules kept from the rest of the library:

* **the plan stage still does the work** - a wrapper calls ``plan()`` then the plan, so a compile or a
  workspace allocation is not smuggled into a step, it happens where the caller asked for it;
* **nothing is guessed** - the wrappers do not infer ``num_qo_heads`` or ``head_dim`` from tensor
  shapes. A wrong guess here is a silently wrong number, and the shape that looks like 32 heads x 64
  is also 16 heads x 128.

Callers that want to amortise planning (the normal serving case) should use ``plan()``/``run()``
directly and keep the plan; these wrappers exist for the platform's binding and for tests.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "mla_decode",
    "gdn_chunk_wu",
    "gdn_conv_update",
    "gdn_gating",
    "paged_decode",
    "paged_prefill",
]


def _plan_then_run(op: str, pure: dict, impl: Optional[str], *tensors: Any) -> Any:
    from . import plan

    return plan(op, impl=impl, **pure)(*tensors)


def paged_decode(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_indptr: Any = None,
    kv_indices: Any = None,
    kv_last_page_len: Any = None,
    kv_layout: str = "NHD",
    dtype: Any = None,
    impl: Optional[str] = None,
) -> Any:
    """Plan and run one decode step.

    ``q`` is ``[total_q, num_qo_heads * head_dim]`` (or the 4-D form the backend accepts); the KV cache
    is state and is captured by the plan, so it is not part of the executed call.
    """
    pure = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        kv_layout=kv_layout,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    if dtype is not None:
        pure["dtype"] = dtype
    return _plan_then_run("paged_decode_attention", pure, impl, q)


def paged_prefill(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qo_indptr: Any = None,
    qo_len: Optional[int] = None,
    batch: Optional[int] = None,
    kv_indptr: Any = None,
    kv_indices: Any = None,
    kv_last_page_len: Any = None,
    kv_layout: str = "NHD",
    dtype: Any = None,
    impl: Optional[str] = None,
) -> Any:
    """Plan and run one prefill/append step; ``q`` is ``[batch, num_qo_heads, qo_len, head_dim]``."""
    pure = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        kv_layout=kv_layout,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    if qo_len is not None:
        pure["qo_len"] = int(qo_len)
    if batch is not None:
        pure["batch"] = int(batch)
    if dtype is not None:
        pure["dtype"] = dtype
    return _plan_then_run("paged_prefill_attention", pure, impl, q)


def gdn_gating(
    A_log: Any,
    a: Any,
    b: Any,
    dt_bias: Any,
    *,
    beta: float = 1.0,
    threshold: float = 20.0,
    impl: Optional[str] = None,
) -> Any:
    """Plan and run the gating op; the plan needs ``(T, H)`` and the dtype, both taken from ``a``."""
    pure = dict(shape=tuple(int(d) for d in a.shape), dtype=str(a.dtype).replace("torch.", ""))
    plan = _plan(gdn_gating_op(), pure, impl)
    return plan(A_log, a, b, dt_bias, beta=beta, threshold=threshold)


def gdn_conv_update(
    x: Any,
    state: Any,
    weight: Any,
    bias: Any,
    line: int,
    *,
    width: int = 4,
    impl: Optional[str] = None,
) -> Any:
    """Plan and run one conv-state update; ``state`` is ``[lines, channels, width]``."""
    channels = int(state.shape[1])
    pure = dict(
        channels=channels,
        width=int(width),
        lines=int(state.shape[0]),
        dtype=str(state.dtype).replace("torch.", ""),
    )
    plan = _plan(gdn_conv_op(), pure, impl)
    return plan(x, state, weight, bias, int(line))


def gdn_chunk_wu(
    a_kkt: Any,
    k: Any,
    v: Any,
    beta: Any,
    g_cumsum: Any,
    cu_seqlens: Any,
    chunk_indices: Any,
    *,
    impl: Optional[str] = None,
) -> Any:
    """Plan and run the W/U recomputation; the compile-time spec is read from the tensors' geometry."""
    t_total, hv, dim = (int(d) for d in k.shape)
    pure = dict(
        t_total=t_total,
        hv_heads=hv,
        hk_heads=int(a_kkt.shape[1]) if a_kkt.dim() > 1 else hv,
        dim=dim,
        num_seqs=int(cu_seqlens.numel()) - 1,
        nt_chunks=int(chunk_indices.shape[0]),
        dtype=str(k.dtype).replace("torch.", ""),
    )
    plan = _plan(gdn_wu_op(), pure, impl)
    return plan(a_kkt, k, v, beta, g_cumsum, cu_seqlens, chunk_indices)


# --------------------------------------------------------------------------- small indirections
#
# The op names live in the family modules (importing them at module scope would drag the whole package
# into ``ascend_attention.entry``, and the platform imports one target at a time).


def gdn_gating_op() -> str:
    from .ops.gdn import OP

    return OP


def gdn_conv_op() -> str:
    from .ops.gdn import CONV_OP

    return CONV_OP


def gdn_wu_op() -> str:
    from .ops.gdn import WU_OP

    return WU_OP


def _plan(op: str, pure: dict, impl: Optional[str]) -> Any:
    from . import plan

    return plan(op, impl=impl, **pure)



def mla_decode(
    q_nope: Any,
    q_pe: Any,
    c_kv: Any,
    k_pe: Any,
    *,
    num_heads: int,
    kv_indptr: Any,
    kv_indices: Any,
    kv_last_page_len: Any,
    sm_scale: Optional[float] = None,
    impl: Optional[str] = None,
) -> Any:
    """Plan and run one MLA decode step over a paged compressed cache (returns the latent output)."""
    pure = dict(
        num_heads=num_heads,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        c_kv=c_kv,
        k_pe=k_pe,
    )
    if sm_scale is not None:
        pure["sm_scale"] = sm_scale
    return _plan_then_run("mla_decode_attention", pure, impl, q_nope, q_pe)
