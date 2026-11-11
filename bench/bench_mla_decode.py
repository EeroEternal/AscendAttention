#!/usr/bin/env python
"""``bench_mla_decode``: absolute latency of the MLA decode reference (no ratio to claim).

The shape is the one the models use: 128 heads, latent 512 + RoPE 64, 2048 cached tokens, page 128. That
is the size a real MLA kernel has to beat, and it is also why MLA exists at all - a per-head KV cache at
that context would be ~10x the traffic.

This is a Python loop over requests and a dense matmul per head; the number is expected to be poor. It is
recorded so that "MLA is implemented" is not confused with "MLA is fast".

Usage::

    python bench/bench_mla_decode.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch

from ascend_attention.ops.mla import mla_decode_reference


def _timed(fn, iters, warmup=2):
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
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--rope", type=int, default=64)
    parser.add_argument("--klen", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "npu" else torch.float32
    torch.manual_seed(0)
    pages = (args.klen + args.page_size - 1) // args.page_size
    total = pages * args.batch
    c_kv = torch.randn(total, args.page_size, args.latent, dtype=dtype, device=device)
    k_pe = torch.randn(total, args.page_size, args.rope, dtype=dtype, device=device)
    q_nope = torch.randn(args.batch, args.heads, 1, args.latent, dtype=dtype, device=device) / args.latent**0.5
    q_pe = torch.randn(args.batch, args.heads, 1, args.rope, dtype=dtype, device=device) / args.rope**0.5
    indptr = torch.arange(0, (args.batch + 1) * pages, pages, dtype=torch.int32)
    kv_indices = torch.arange(total, dtype=torch.long)
    last = torch.full((args.batch,), args.page_size, dtype=torch.int32)

    timings = {
        "mla_decode": _timed(
            lambda: mla_decode_reference(
                q_nope, q_pe, c_kv, k_pe, kv_indptr=indptr, kv_indices=kv_indices,
                kv_last_page_len=last, num_heads=args.heads,
            ),
            args.iters,
        )
    }
    payload = {
        "op": "mla_decode_attention",
        "impl": "torch.reference",
        "shape": {
            "batch": args.batch,
            "heads": args.heads,
            "latent": args.latent,
            "rope": args.rope,
            "kv_len": args.klen,
            "page_size": args.page_size,
        },
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "measured_ratio": 1.0,
        "ratio_note": (
            "definitional 1.0: the only implementation and a per-request Python loop; recorded so that "
            "'implemented' is not read as 'fast'"
        ),
        "ms": timings,
        "command": f"bench/bench_mla_decode.py --device {args.device} --batch {args.batch}",
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
