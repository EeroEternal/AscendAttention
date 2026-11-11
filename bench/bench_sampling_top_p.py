#!/usr/bin/env python
"""``bench_sampling_top_p``: absolute latency of the nucleus filter at a serving-shaped vocabulary.

There is no second implementation to compare against, so the ratio is 1.0 by definition; what this
records is the cost a kernel would have to beat, and it is measured at the shape that matters
(B=8 rows, V=152064 = the vocabulary of the model this repo's other sampling numbers use).

What dominates is the sort of the full vocabulary: ``V log V`` per row, followed by a cumulative sum
and a scatter. That is exactly why a real serving stack does not do this on the host, and why the
ledger number is worth having - it bounds how much a nucleus kernel can save before it can claim
anything.

Usage::

    python bench/bench_sampling_top_p.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch

from ascend_attention.ops.sampling import top_p_filter


def _timed(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return {"median_ms": round(statistics.median(times), 4), "min_ms": round(min(times), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "npu"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=152064)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "npu" else torch.float32
    torch.manual_seed(0)
    logits = torch.randn(args.batch, args.vocab, dtype=dtype, device=device)

    timings = {"top_p": _timed(lambda: top_p_filter(logits, args.p), args.iters)}
    survivors = torch.isfinite(top_p_filter(logits, args.p)).sum(-1)
    payload = {
        "op": "sampling_top_p",
        "impl": "torch.top_p",
        "shape": {"batch": args.batch, "vocab": args.vocab, "p": args.p},
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "measured_ratio": 1.0,
        "ratio_note": (
            "definitional 1.0: the only implementation; the number is the absolute cost of a "
            "full-vocabulary sort + cumsum that a nucleus kernel would have to beat"
        ),
        "survivors_median": int(survivors.median()),
        "ms": timings,
        "command": f"bench/bench_sampling_top_p.py --device {args.device} --batch {args.batch}",
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
