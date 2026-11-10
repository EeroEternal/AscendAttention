#!/usr/bin/env python
"""``paged_prefill_attention`` (torch.reference): absolute latency, no ratio to claim.

There is no second prefill implementation in this phase, so ``measured_ratio`` is definitional 1.0.
The number is still worth recording: it is the *floor* a future prefill kernel has to beat, and the
shape below (4 requests x 512 new tokens over a 4k cache) is a plausible serving step.

Usage::

    python bench/bench_prefill_reference.py --batch 4 --qo-len 512 --kv-len 4096
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--qo-len", type=int, default=512)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--h-q", type=int, default=32)
    parser.add_argument("--h-kv", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    import torch

    from ascend_attention.testing.cache import build_paged_cache

    if args.qo_len > args.kv_len:
        parser.error("--qo-len must be <= --kv-len")
    torch.manual_seed(0)
    k = torch.randn(args.batch, args.h_kv, args.kv_len, args.head_dim, dtype=torch.bfloat16)
    v = torch.randn(args.batch, args.h_kv, args.kv_len, args.head_dim, dtype=torch.bfloat16)
    k_cache, v_cache, meta = build_paged_cache(
        k, v, page_size=args.page_size, shuffle_pages=True, seed=3
    )
    q = torch.randn(args.batch, args.h_q, args.qo_len, args.head_dim, dtype=torch.bfloat16) * 0.3

    from ascend_attention import plan

    p = plan(
        "paged_prefill_attention",
        k_cache=k_cache,
        v_cache=v_cache,
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=meta.page_size,
        num_qo_heads=args.h_q,
        num_kv_heads=args.h_kv,
        head_dim=args.head_dim,
        qo_len=args.qo_len,
        batch=args.batch,
        dtype="bfloat16",
        device=args.device,
    )

    samples: List[float] = []
    for _ in range(max(1, args.iters)):
        start = time.perf_counter()
        out = p(q)
        samples.append(time.perf_counter() - start)
    result = {
        "op": "paged_prefill_attention",
        "impl": p.impl,
        "shape": {
            "batch": args.batch,
            "qo_len": args.qo_len,
            "kv_len": args.kv_len,
            "h_q": args.h_q,
            "h_kv": args.h_kv,
            "head_dim": args.head_dim,
            "page_size": args.page_size,
        },
        "dtype": "bfloat16",
        "device": args.device,
        "iters": args.iters,
        "ms_median": round(statistics.median(samples) * 1e3, 3),
        "ms_min": round(min(samples) * 1e3, 3),
        "tokens_per_s": round(args.batch * args.qo_len / statistics.median(samples), 1),
        "measured_ratio": 1.0,
        "ratio_kind": "definitional parity (single prefill implementation; the benchmark floor)",
        "out_shape": list(out.shape),
        "command": " ".join(sys.argv),
        "host": platform.node(),
        "date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        import pathlib

        target = pathlib.Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
