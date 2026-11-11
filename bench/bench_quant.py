#!/usr/bin/env python
"""``bench_quant``: absolute latency of the fp8 fake-quant primitives (no ratio to claim).

Two shapes, both from the MoE/MLP activation path: the elementwise quantise-dequantise at
4096x7168, and the dequant-then-matmul GEMM at 1024x4096x1024. A real fp8 kernel keeps the operands
quantised, so the GEMM number here is a *definition*, not a target - it is recorded so the ledger has an
honest baseline rather than an empty field.

Usage::

    python bench/bench_quant.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch

from ascend_attention.ops.quant import fp8_gemm_reference, quant_dequant


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
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=7168)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "npu" else torch.float32
    torch.manual_seed(0)
    x = (torch.randn(args.rows, args.cols, dtype=dtype, device=device) * 3).contiguous()
    scale = x.abs().amax(dim=-1) / 448

    m, k, n = 1024, 4096, 1024
    a = (torch.randn(m, k, dtype=dtype, device=device) / k**0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(k, n, dtype=dtype, device=device) / k**0.5).to(torch.float8_e4m3fn)
    a_scale = torch.ones(m, dtype=dtype, device=device)
    b_scale = torch.ones(n, dtype=dtype, device=device)

    timings = {
        "dequant": _timed(lambda: quant_dequant(x, scale, axis="token"), args.iters),
        "gemm": _timed(lambda: fp8_gemm_reference(a, a_scale, b, b_scale), args.iters),
    }
    payload = {
        "op": ["quant_dequant_fp8", "quant_fp8_gemm"],
        "impl": ["torch.fake_quant", "torch.dequant_gemm"],
        "shape": {"dequant": [args.rows, args.cols], "gemm": [m, k, n]},
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "measured_ratio": 1.0,
        "ratio_note": (
            "definitional 1.0: the only implementation, and the definition a future fp8 kernel is graded "
            "against; the GEMM number is dequant-then-matmul, not a hardware fp8 path"
        ),
        "ms": timings,
        "command": f"bench/bench_quant.py --device {args.device}",
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
