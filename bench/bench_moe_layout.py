#!/usr/bin/env python
"""Absolute latency for the two MoE layout ops (no ratio: neither has an alternative implementation).

``moe_slot_layout`` and ``moe_prenz_pack`` are **baselines by definition** — the entry says
``measured_ratio: 1.0`` because there is nothing to beat, not because a measurement said so.  What can
be measured is their absolute cost, and that is what this script records: if the slot rebuild costs
more than the GEMM it feeds, the "no host sync" design is pointless and someone should know.

Usage::

    source scripts/env-ascend102.sh
    python bench/bench_moe_layout.py --rows 4096 --experts 64
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

BLOCK_M = 32


def _sync(device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def _time(fn, device, iters: int) -> float:
    _sync(device)
    fn()
    _sync(device)
    best = float("inf")
    for _ in range(iters):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        best = min(best, time.perf_counter() - start)
    return best


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=BLOCK_M)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from ascend_attention import plan
    from ascend_attention.ops.moe import em_upper_bound

    device = torch.device(args.device)
    em_static = em_upper_bound(args.rows, args.experts, args.block_m)
    rows_static = em_static * args.block_m

    g = torch.Generator().manual_seed(0)
    # a plausible expert skew: most rows to a few experts, the rest empty
    weights = torch.rand(args.experts, generator=g) ** 3
    weights = weights / weights.sum()
    counts = torch.multinomial(weights, args.rows, replacement=True).bincount(
        minlength=args.experts
    )
    group_list = torch.cumsum(counts, 0).to(torch.int32).to(device)

    p_layout = plan(
        "moe_slot_layout",
        rows_static=rows_static,
        num_rows=args.rows,
        em_static=em_static,
        num_experts=args.experts,
        block_m=args.block_m,
    )
    tile = torch.randn(args.cols, args.cols, generator=g).to(torch.bfloat16).to(device)
    p_pack = plan("moe_prenz_pack", rows=args.cols, cols=args.cols, dtype="bfloat16")

    result: Dict[str, Any] = {
        "ops": ["moe_slot_layout", "moe_prenz_pack"],
        "shape": {
            "rows": args.rows,
            "experts": args.experts,
            "block_m": args.block_m,
            "em_static": em_static,
            "rows_static": rows_static,
            "pack_matrix": [args.cols, args.cols],
        },
        "device": args.device,
        "iters": args.iters,
        "measured_ratio": None,
        "ratio_note": (
            "layout ops have a single implementation, so no ratio exists; these are absolute "
            "latencies of the baseline, recorded so a regression has a reference point"
        ),
        "command": " ".join(sys.argv),
        "host": platform.node(),
        "date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    layout_ms = statistics.median(
        [_time(lambda: p_layout(group_list), device, args.iters) for _ in range(3)]
    ) * 1e3
    pack_ms = statistics.median(
        [_time(lambda: p_pack(tile), device, args.iters) for _ in range(3)]
    ) * 1e3
    result["ms"] = {"slot_layout": round(layout_ms, 4), "prenz_pack": round(pack_ms, 4)}
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
