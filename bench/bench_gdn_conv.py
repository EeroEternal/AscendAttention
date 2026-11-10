#!/usr/bin/env python
"""``gdn_conv_update`` A/B: fused TileLang kernel vs the torch port it would replace.

Why this A/B is the meaningful one: the baseline here is the *same op* the engine runs today when no
kernel exists (``tileinfer/vllm/ops/fla/causal_conv1d.py: causal_conv1d_update``), not a vendor
attention path.  So the ratio answers exactly the question the platform asks — "use this op, or fall
back to the baseline?" — and it is recorded as ``measured_ratio``.

What this number must never be used for: claiming a step-level win.  TileInfer measured 42.5x on the
kernel and **0.995x end-to-end**, because a ~0.72 ms per-layer kernel is a small slice of a ~103 ms
decode step.  Both levels are written into the JSON; the end-to-end arm belongs to P5.

Usage::

    source scripts/env-ascend102.sh
    python bench/bench_gdn_conv.py --channels 4096 --lines 8
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch

DT = torch.bfloat16


def _build(channels: int, lines: int, width: int, device):
    torch.manual_seed(0)
    x = torch.randn(channels, dtype=DT, device=device) * 0.2
    state = torch.randn(lines, channels, width - 1, dtype=DT, device=device) * 0.2
    weight = torch.randn(channels, width, dtype=DT, device=device) * 0.1
    bias = torch.randn(channels, dtype=torch.float32, device=device) * 0.1
    return x, state, weight, bias


def _time(fn, iters: int) -> float:
    torch.npu.synchronize()
    fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--lines", type=int, default=8)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--line", type=int, default=3)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from ascend_attention import plan

    device = torch.device(args.device)
    x, state0, weight, bias = _build(args.channels, args.lines, args.width, device)

    p_kernel = plan(
        "gdn_conv_update",
        impl="tilelang.conv950",
        channels=args.channels,
        width=args.width,
        lines=args.lines,
        dtype="bfloat16",
    )
    p_torch = plan(
        "gdn_conv_update",
        impl="torch.causal_conv1d_update",
        channels=args.channels,
        width=args.width,
        lines=args.lines,
        dtype="bfloat16",
    )

    def run_kernel():
        st = state0.clone()
        return p_kernel(x, st, weight, bias, args.line)

    def run_torch():
        st = state0.clone()
        return p_torch(x, st, weight, bias, args.line)

    out_k = run_kernel()
    out_t = run_torch()
    scale = float(out_t.float().abs().max()) or 1.0
    result: Dict[str, Any] = {
        "op": "gdn_conv_update",
        "shape": {"channels": args.channels, "lines": args.lines, "width": args.width},
        "dtype": "bfloat16",
        "device": args.device,
        "pairs": args.pairs,
        "iters": args.iters,
        "max_abs_diff_out": float((out_k.float() - out_t.float()).abs().max()),
        "relative_to_scale": float((out_k.float() - out_t.float()).abs().max()) / scale,
        "command": " ".join(sys.argv),
        "host": platform.node(),
        "date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }

    pairs = []
    for index in range(max(4, args.pairs)):
        if index % 2 == 0:
            a, b = _time(run_kernel, args.iters), _time(run_torch, args.iters)
        else:
            b, a = _time(run_torch, args.iters), _time(run_kernel, args.iters)
        pairs.append(b / a)
    result["measured_ratio"] = round(statistics.median(pairs), 4)
    result["pair_ratios"] = [round(r, 4) for r in pairs]
    result["ms"] = {
        "kernel": round(statistics.median([_time(run_kernel, args.iters) for _ in range(3)]) * 1e3, 4),
        "torch": round(statistics.median([_time(run_torch, args.iters) for _ in range(3)]) * 1e3, 4),
    }
    result["ratio_note"] = (
        "op level: torch baseline / fused kernel, in-process alternating pairs, median of pair "
        "ratios. End-to-end was 0.995x in TileInfer (no step-level win) - do not read this as a "
        "serving speedup"
    )
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
