#!/usr/bin/env python
"""``bench_sampling_min_p``: absolute latency of the min-p filter at a serving-shaped vocabulary.

Two passes over the vocabulary and no sort, which is why a serving stack applies min-p before top-k and
top-p: this number is the floor of what the sampling chain costs, and the comparison with
``bench_sampling_top_p`` (a full sort) is the interesting one.

Usage::

    python bench/bench_sampling_min_p.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch

from ascend_attention.ops.sampling import min_p_filter


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
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "npu" else torch.float32
    torch.manual_seed(0)
    logits = torch.randn(args.batch, args.vocab, dtype=dtype, device=device)

    timings = {"min_p": _timed(lambda: min_p_filter(logits, args.p), args.iters)}
    survivors = torch.isfinite(min_p_filter(logits, args.p)).sum(-1)
    payload = {
        "op": "sampling_min_p",
        "impl": "torch.min_p",
        "shape": {"batch": args.batch, "vocab": args.vocab, "p": args.p},
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "measured_ratio": 1.0,
        "ratio_note": (
            "definitional 1.0: the only implementation; no sort, so this is the cheap end of the "
            "sampling chain and the number a kernel has to beat"
        ),
        "survivors_median": int(survivors.median()),
        "ms": timings,
        "command": f"bench/bench_sampling_min_p.py --device {args.device} --batch {args.batch}",
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
