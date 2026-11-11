"""MLA (multi-head latent attention), decode step, over a **paged compressed** KV cache.

What MLA is, in the form that decides the arithmetic: the cache does not hold per-head K and V. It holds
one shared latent ``c_kv`` (``D_c`` wide) plus a decoupled RoPE key ``k_pe`` (``D_r`` wide), per token per
layer. Scores come from two small projections, and the value side is the latent itself::

    s[b,h] = (q_nope[b,h] @ c_kv[0:l]^T + q_pe[b,h] @ k_pe[0:l]^T) / sqrt(D_c + D_r)
    p       = softmax(s)
    out[b,h] = p @ c_kv[0:l]                      # [D_c], latent output

Two consequences worth stating because they are where implementations go wrong:

* the softmax scale is ``1/sqrt(D_c + D_r)``, **not** ``1/sqrt(D_c)``: the score is a sum of two dot
  products of different widths, and using the wrong one is a silent ~1.4x temperature change;
* the output is the **latent**, not a per-head value vector. The ``W_uv`` up-projection that turns it into
  ``D_v`` per head is a separate GEMM owned by the model; folding it in here would make this op
  model-specific, which is exactly what an operator layer must not be.

This file is a **reference implementation only**: correct, readable, and slow (a Python loop per request).
No MLA kernel is migrated, so the op is advertised as a baseline with a definitional 1.0 and it is the
only implementation - which is why it is enabled: a missing op is worse than a slow one for a model that
cannot start without it, and the ratio field says plainly that no speed claim is being made.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

from ...plan import OpPlan
from ...spec import REGISTRY, ImplSpec, ShapeSpec, dim_in, dim_min

__all__ = [
    "OP",
    "REFERENCE",
    "kv_len_from_pages",
    "mla_decode_reference",
    "plan_mla",
    "run_mla",
]

OP = "mla_decode_attention"

#: The latent width is what the model ships (DeepSeek-V2/V3 use 512 + 64); the reference accepts the
#: declared set and refuses anything else instead of silently using the wrong scale.
LATENT_DIMS = (128, 256, 512, 576)
ROPE_DIMS = (32, 64, 128)

REFERENCE = ImplSpec(
    op=OP,
    impl="torch.reference",
    form="torch",
    device="any",
    shape=ShapeSpec(
        ndim=4,
        dims=(dim_min(1), dim_min(1), 1, dim_in(*LATENT_DIMS)),
        dtypes=frozenset({"bfloat16", "float16", "float32"}),
        contiguous=False,
    ),
    kv_layouts=frozenset({"ND"}),
    graph="eager_only",
    plan_kind="attention",
    enabled_by_default=True,
    measured_ratio=1.0,
    bench="bench/bench_mla_decode.py",
    ratio_note=(
        "definitional 1.0: the only implementation, and a Python loop per request. No MLA kernel is "
        "migrated, so there is nothing to compare against - the ledger records absolute latency at the "
        "shape the models use (D_c=512, D_r=64, 128 heads, 2048-token KV)."
    ),
    provenance="no TileInfer origin: written here from the MLA definition (DeepSeek-V2/V3 latent attention)",
    note=(
        "paged, compressed KV cache: c_kv [pages, page_size, D_c] + k_pe [pages, page_size, D_r]; "
        "returns the LATENT output [B, H, 1, D_c]. Scale is 1/sqrt(D_c + D_r) - using 1/sqrt(D_c) is a "
        "silent temperature error. W_uv up-projection is the model's, not this op's."
    ),
)

REGISTRY.register(REFERENCE)


def kv_len_from_pages(kv_indptr: torch.Tensor, kv_last_page_len: torch.Tensor, page_size: int) -> torch.Tensor:
    """Per-request KV length from the page table, without a host round trip.

    ``kv_indptr`` is the cumulative page count (``[B+1]``), so the number of pages of request ``b`` is
    ``indptr[b+1] - indptr[b]``; the last page may be partially filled.
    """
    pages = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int64)
    live = (pages - 1) * int(page_size) + kv_last_page_len.to(torch.int64)
    return live.clamp(min=0)


def mla_decode_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    *,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_heads: int,
    sm_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The definition, executed per request; the oracle for any future kernel.

    ``q_nope``/``q_pe``: ``[B, H, 1, D_c]`` / ``[B, H, 1, D_r]``; ``c_kv``/``k_pe``: paged as above.
    """
    if q_nope.dim() != 4 or q_nope.shape[2] != 1:
        raise ValueError(f"mla_decode expects q [B, H, 1, D_c], got {tuple(q_nope.shape)}")
    batch = int(q_nope.shape[0])
    if int(q_nope.shape[1]) != num_heads:
        raise ValueError(f"num_heads={num_heads} but q has {int(q_nope.shape[1])} heads")
    latent = int(q_nope.shape[-1])
    rope = int(q_pe.shape[-1])
    page_size = int(c_kv.shape[1])
    if c_kv.shape[2] != latent:
        raise ValueError(f"c_kv latent width {int(c_kv.shape[2])} != q_nope width {latent}")
    if k_pe.shape[2] != rope:
        raise ValueError(f"k_pe rope width {int(k_pe.shape[2])} != q_pe width {rope}")
    if int(kv_indptr.numel()) != batch + 1:
        raise ValueError(f"kv_indptr must be [{batch + 1}], got {int(kv_indptr.numel())}")

    scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(latent + rope)
    lengths = kv_len_from_pages(kv_indptr, kv_last_page_len, page_size).to(q_nope.device)
    max_len = int(lengths.max()) if batch and int(lengths.max()) > 0 else 0
    compute = q_nope.dtype
    result = torch.zeros(
        (batch, num_heads, 1, latent), dtype=compute, device=q_nope.device
    )
    if max_len == 0:
        return out if out is None else out.copy_(result) or out

    # One gathered window per request: pages -> positions. ``positions >= length`` is padding.
    for b in range(batch):
        pages = kv_indices[int(kv_indptr[b]) : int(kv_indptr[b + 1])].to(torch.long)
        length = int(lengths[b])
        if length == 0:
            continue
        c = c_kv.index_select(0, pages).reshape(-1, latent)[:length]
        pe = k_pe.index_select(0, pages).reshape(-1, rope)[:length]
        scores = (q_nope[b, :, 0] @ c.T) * scale + (q_pe[b, :, 0] @ pe.T) * scale
        probs = torch.softmax(scores, dim=-1)
        result[b, :, 0] = probs @ c

    if out is not None:
        out.copy_(result)
        return out
    return result


def plan_mla(impl: ImplSpec, **pure: Any) -> OpPlan:
    """Plan one MLA decode step. The cache is state (captured); ``q_*`` is the executed input."""
    c_kv = pure.get("c_kv")
    k_pe = pure.get("k_pe")
    if c_kv is None or k_pe is None:
        raise ValueError("mla_decode_attention.plan needs c_kv and k_pe (the compressed cache)")
    num_heads = int(pure["num_heads"])
    latent = int(pure.get("latent_dim") or c_kv.shape[-1])
    rope = int(pure.get("rope_dim") or k_pe.shape[-1])
    page_size = int(c_kv.shape[1])
    kv_indptr = pure.get("kv_indptr")
    if kv_indptr is None:
        raise ValueError("mla_decode_attention.plan needs kv_indptr")
    batch = int(kv_indptr.numel()) - 1
    dtype_name = str(pure.get("dtype") or c_kv.dtype).rsplit(".", 1)[-1]
    # The compressed cache is 3-D by construction ([pages, page_size, D]), so MLA has no "NHD" choice to
    # make; the argument exists so a caller that always passes its layout does not have to special-case us.
    kv_layout = str(pure.get("kv_layout", "ND"))
    if kv_layout not in impl.kv_layouts:
        raise ValueError(f"{OP}/{impl.impl}: kv_layout {kv_layout!r} not in {sorted(impl.kv_layouts)}")
    if latent not in LATENT_DIMS:
        raise ValueError(f"{OP}: latent width {latent} is outside the declared set {LATENT_DIMS}")
    if rope not in ROPE_DIMS:
        raise ValueError(f"{OP}: rope width {rope} is outside the declared set {ROPE_DIMS}")
    if impl.shape is not None and not impl.shape.matches((batch, num_heads, 1, latent), dtype_name):
        raise ValueError(
            f"{OP}/{impl.impl}: q=({batch}, {num_heads}, 1, {latent}) dtype={dtype_name} is outside the "
            "declared contract"
        )

    sm_scale = pure.get("sm_scale")
    kv_indices = pure.get("kv_indices")
    kv_last_page_len = pure.get("kv_last_page_len")

    def execute(
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tuple(q_nope.shape) != (batch, num_heads, 1, latent):
            raise ValueError(
                f"{OP}/{impl.impl}: q_nope must be {(batch, num_heads, 1, latent)}, "
                f"got {tuple(q_nope.shape)}"
            )
        if tuple(q_pe.shape) != (batch, num_heads, 1, rope):
            raise ValueError(
                f"{OP}/{impl.impl}: q_pe must be {(batch, num_heads, 1, rope)}, got {tuple(q_pe.shape)}"
            )
        return mla_decode_reference(
            q_nope,
            q_pe,
            c_kv,
            k_pe,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
            num_heads=num_heads,
            sm_scale=sm_scale,
            out=out,
        )

    return OpPlan(
        op=OP,
        impl=impl.impl,
        plan_kind=impl.plan_kind,
        shape=(batch, num_heads, 1, latent),
        dtypes=(dtype_name,),
        kv_layouts=(kv_layout,),
        graph=impl.graph,
        meta={
            "num_heads": num_heads,
            "latent_dim": latent,
            "rope_dim": rope,
            "page_size": page_size,
            "mode": "decode",
            "sm_scale": float(sm_scale) if sm_scale is not None else 0.0,
        },
        state={"c_kv": c_kv, "k_pe": k_pe, "kv_indptr": kv_indptr, "kv_indices": kv_indices},
        execute=execute,
    )


def run_mla(p: OpPlan, q_nope: torch.Tensor, q_pe: torch.Tensor, out: Optional[torch.Tensor] = None):
    return p(q_nope, q_pe, out)


REGISTRY.op(
    OP,
    doc="MLA decode over a paged compressed KV cache (latent + RoPE key), reference implementation. "
    "Returns the latent output; the W_uv up-projection belongs to the model.",
)
REGISTRY.planner(OP, plan_mla)
