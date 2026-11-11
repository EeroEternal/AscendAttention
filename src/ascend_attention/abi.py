"""Kernel ABI arithmetic: pool-capacity quantisation and the zero-copy capacity view.

These functions used to live in three different layers of TileInfer — ``declared_pool_cap`` and
``pool_view`` in the attention kernel module, ``capacity_view`` in ``serving/buckets.py`` (a
layering inversion: the kernel imported the serving layer), ``padded_rows`` in the kernel too.
They belong together and they belong **here**, because "which capacity was this kernel compiled
for" is part of the kernel's ABI, not of a serving policy.

Why quantise at all: a TileLang compile is 70-100 s and holds the GIL.  A key built from the exact
page count drifts on every batch (9425 -> 9899 pages) and recompiles inside the engine, so the ABI
compiles for a *declared* capacity (next power of two by default) and the caller views the real pool
through it.  Behaviour is byte-for-byte the TileInfer implementation, including the two traps it
documents: the storage offset must be carried over, and ``cap < pool`` is a caller bug.
"""

from __future__ import annotations

import os
from typing import Any, Optional

__all__ = [
    "quantise_tiles",
    "ENV_POOL_CAP",
    "ENV_POOL_CAP_LEGACY",
    "padded_rows",
    "declared_pool_cap",
    "capacity_view",
    "pool_view",
]

#: Pin the declared capacity instead of rounding up (explicit plan data wins over the environment).
ENV_POOL_CAP = "ASCEND_ATTN_POOL_CAP"

#: The TileInfer-era spelling, honoured during the migration so an existing deployment keeps its
#: cache keys.  Libraries should pass ``pin=`` explicitly; this is a compatibility fallback.
ENV_POOL_CAP_LEGACY = "TILEINFER_POOL_CAP"


def padded_rows(group: int) -> int:
    """M-tile size for a given GQA group (ND->NZ copy template needs rows % 16 == 0).

    ``group = 8`` (a very common GQA ratio) must become 32: with ``ROWS = BR // 2`` a bare group of
    8 does not compile.  The caller passes zero-padded rows; padded rows cost cube time, not
    correctness.
    """
    group = int(group)
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")
    return max(32, ((group + 31) // 32) * 32)


def _pinned() -> Optional[int]:
    for name in (ENV_POOL_CAP, ENV_POOL_CAP_LEGACY):
        raw = os.environ.get(name, "").strip()
        if raw:
            return int(raw)
    return None


def declared_pool_cap(pool_pages: int, *, pin: Optional[int] = None) -> int:
    """The capacity a kernel is compiled for: ``pin`` / env / next power of two.

    Pages outside the ``kv_indptr``/``kv_indices`` spans are never dereferenced, so the declared
    capacity only has to *cover* the real pool — its exact value is free, which is what keeps the
    pool size out of every kernel, plan and pre-warm key.
    """
    pages = int(pool_pages)
    if pages <= 0:
        raise ValueError(f"pool_pages must be positive, got {pages}")
    pinned = pin if pin is not None else _pinned()
    if pinned is not None:
        if int(pinned) < pages:
            raise ValueError(
                f"pinned capacity {pinned} is below the real pool ({pages} pages); "
                "the declared capacity must cover every live page"
            )
        return int(pinned)
    return 1 if pages <= 1 else 1 << (pages - 1).bit_length()


def capacity_view(pool: Any, cap: int) -> Any:
    """Present ``pool`` at the declared page capacity, without copying.

    ``torch.as_strided`` refuses to build a view larger than the storage; ``Tensor.set_`` does not
    validate, so the capacity view re-binds the pool's own storage at the declared extent.
    """
    if pool is None:
        return None
    if pool.dim() != 4:
        raise ValueError(
            f"capacity_view needs a [pages, page_size, heads, dim] pool, got shape "
            f"{tuple(pool.shape)}"
        )
    cap = int(cap)
    pages = int(pool.shape[0])
    if cap < pages:
        raise ValueError(
            f"pool grew past the kernel's compiled capacity ({pages} > {cap}); "
            "the declared capacity must cover every live page"
        )
    flat = pool if pool.is_contiguous() else pool.contiguous()
    if cap == pages:
        return flat
    view = pool.new_empty((0,))
    # ``set_`` carries the storage *offset*: the V half of a fused pool lives at a non-zero offset
    # and re-binding at 0 silently makes the kernel attend K twice.
    view.set_(flat.untyped_storage(), flat.storage_offset(), (cap, *flat.shape[1:]), flat.stride())
    return view


def pool_view(k_cache: Any, cap: int) -> Any:
    """Kernel call-site spelling of :func:`capacity_view`."""
    return capacity_view(k_cache, cap)


#: Prefill tile counts are bucketed to powers of two before they reach the compiler.  ``num_tiles``
#: is a *linear function of the prompt length*, so keying the compiled variant on it (which
#: :func:`_cache_key` does, deliberately: anything that changes the emitted program must be in the
#: key) recompiled the kernel for every new sequence length - measured at 129.7 s / 112.9 s of
#: prefill in a 192-token run, against 0.8 s for the same step with the kernel already built.  The
#: plan is still per step (that part is a contract); the *compiled artifact* is now shared by every
#: step whose tile count fits the bucket.
_TILE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def quantise_tiles(num_tiles: int) -> int:
    """Round a live tile count up to the next bucket (at least 1).

    Padding is already a supported state (P3: padded tiles carry ``q_len == 0`` and are skipped by
    the pack/unpack loops and by the kernel's own row mask), so the only cost is a slightly larger
    buffer, not a different computation.
    """
    if num_tiles < 0:
        raise ValueError(f"quantise_tiles expects a non-negative count, got {num_tiles}")
    for bucket in _TILE_BUCKETS:
        if num_tiles <= bucket:
            return bucket
    raise ValueError(
        f"num_tiles={num_tiles} exceeds the largest bucket ({_TILE_BUCKETS[-1]}); raise the bucket "
        "table deliberately, with a measurement, rather than compiling one variant per step"
    )
