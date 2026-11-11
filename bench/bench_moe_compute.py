#!/usr/bin/env python
"""``bench_moe_compute``: absolute latency of the five MoE compute baselines (no ratio to claim).

There is no alternative implementation of these ops in the repo, so ``measured_ratio`` is 1.0 by
definition and this script records what the baseline costs - that is the number a future kernel has to
beat, and the number the platform can multiply by a step count when it estimates a model's MoE time.

Two rules from ``docs/perf-rules.md`` are visible here: the shapes are the ones the ledger claims (so a
reader can recompute them), and the activation/routing inputs are seeded, so two runs on two machines
are comparable.  ``--device npu`` runs the same code on the card, which is what a captured step does.

Usage::

    python bench/bench_moe_compute.py                 # CPU, the default
    python bench/bench_moe_compute.py --device npu
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch

from ascend_attention.ops.moe import compute


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
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inter", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "npu" else torch.float32
    torch.manual_seed(0)
    tokens, hidden, inter = args.tokens, args.hidden, args.inter
    experts, top_k, block_size = args.experts, args.top_k, args.block_size

    scores = torch.randn(tokens, experts, dtype=dtype, device=device)
    weights, ids = compute.topk_softmax(scores, top_k)
    x = torch.randn(tokens, hidden, dtype=dtype, device=device)
    w1 = torch.randn(experts, 2 * inter, hidden, dtype=dtype, device=device)
    w2 = torch.randn(experts, hidden, inter, dtype=dtype, device=device)

    timings = {
        "silu_and_mul": _timed(
            lambda: compute.silu_and_mul(torch.randn(tokens, 2 * inter, dtype=dtype, device=device)),
            args.iters,
        ),
        "topk_softmax": _timed(lambda: compute.topk_softmax(scores, top_k), args.iters),
        "moe_align_block_size": _timed(
            lambda: compute.moe_align_block_size(ids, experts, block_size), args.iters
        ),
        "fused_experts": _timed(
            lambda: compute.fused_experts(x, w1, w2, weights, ids), max(1, args.iters // 4)
        ),
        "unquantized_fused_moe": _timed(
            lambda: compute.unquantized_fused_moe(
                x, w1, w2, weights, ids, num_experts=experts
            ),
            max(1, args.iters // 4),
        ),
    }

    payload = {
        "op": "moe_compute",
        "shape": {
            "tokens": tokens,
            "hidden": hidden,
            "inter": inter,
            "experts": experts,
            "top_k": top_k,
            "block_size": block_size,
        },
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "measured_ratio": 1.0,
        "ratio_note": (
            "definitional 1.0: each op has exactly one implementation here, so there is nothing to "
            "compare against; these are absolute latencies for a future kernel to beat"
        ),
        "ms": timings,
        "command": f"bench/bench_moe_compute.py --device {args.device} --tokens {tokens}",
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
