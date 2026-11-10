#!/usr/bin/env python
"""``gdn_gating`` A/B ledger entry — the script every ratio in the manifest must point at.

Rules this script implements (they are the repo's measurement contract, not decoration):

* **alternate in-process**: baseline and candidate are measured in the same process, in alternating
  order, so a clock/thermal drift cannot masquerade as a speedup;
* **at least four pairs**, and the median of per-pair ratios is the reported number;
* **op level and end-to-end are separate fields** — a kernel-level win that does not survive the
  step is recorded as a loss at the level that matters;
* **record the command, the card and the date** next to the number.

Right now the only implementation is the torch baseline, so the output is a baseline calibration
(ratio 1.0 by definition).  When the P2 kernel lands, ``--impl kernel...`` measures it against the
same baseline in the same process and its ``measured_ratio`` is what may be copied into the spec.

Usage::

    python bench/bench_gdn_gating.py                    # baseline only, JSON to stdout
    python bench/bench_gdn_gating.py --pairs 6 --t 128  # bigger batch
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

from ascend_attention import plan, run


def _build(t: int, h: int, dtype: torch.dtype, device: str):
    g = torch.Generator(device="cpu").manual_seed(0)
    a_log = torch.randn(h, generator=g) * 0.3
    dt_bias = torch.randn(h, generator=g) * 0.1
    a = torch.randn(t, h, generator=g).to(dtype)
    b = torch.randn(t, h, generator=g).to(dtype)
    if device != "cpu":
        a_log, dt_bias = a_log.to(device), dt_bias.to(device)
        a, b = a.to(device), b.to(device)
    return a_log, a, b, dt_bias


def _time_once(fn, iters: int) -> float:
    """Seconds per call, best of ``iters`` — a warm call, never the first."""
    fn()  # warm up (allocation, kernel cache, autograd-free path)
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", type=int, default=128, help="tokens (prefill-shaped)")
    parser.add_argument("--h", type=int, default=16, help="heads")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cpu", help="cpu or npu:0")
    parser.add_argument("--pairs", type=int, default=4, help="alternating A/B pairs (>=4)")
    parser.add_argument("--iters", type=int, default=20, help="calls per timing")
    parser.add_argument("--impl", default=None, help="candidate impl to compare against the baseline")
    args = parser.parse_args(argv)

    dtype = getattr(torch, args.dtype)
    a_log, a, b, dt_bias = _build(args.t, args.h, dtype, args.device)
    base = plan("gdn_gating", impl="torch.gating", shape=(args.t, args.h), dtype=args.dtype)
    arms: Dict[str, Any] = {"baseline": base}
    if args.impl:
        arms["candidate"] = plan("gdn_gating", impl=args.impl, shape=(args.t, args.h), dtype=args.dtype)

    timings: Dict[str, List[float]] = {name: [] for name in arms}
    for _ in range(max(1, args.pairs)):
        for name, p in arms.items():  # alternating order inside each pair
            timings[name].append(_time_once(lambda p=p: run(p, a_log, a, b, dt_bias), args.iters))

    baseline = statistics.median(timings["baseline"])
    result: Dict[str, Any] = {
        "op": "gdn_gating",
        "shape": {"t": args.t, "h": args.h},
        "dtype": args.dtype,
        "device": args.device,
        "pairs": args.pairs,
        "iters": args.iters,
        "ms": {name: round(statistics.median(ts) * 1e3, 4) for name, ts in timings.items()},
        "measured_ratio": None,
        "ratio_note": "",
        "command": " ".join(sys.argv),
        "host": platform.node(),
        "date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    if "candidate" in arms:
        pair_ratios = [
            timings["baseline"][i] / timings["candidate"][i] for i in range(len(timings["baseline"]))
        ]
        ratio = statistics.median(pair_ratios)
        result["measured_ratio"] = round(ratio, 4)
        result["pair_ratios"] = [round(r, 4) for r in pair_ratios]
        result["ratio_note"] = (
            "op-level, in-process alternating A/B, median of paired ratios; "
            "end-to-end not measured by this script"
        )
    else:
        result["measured_ratio"] = 1.0
        result["ratio_note"] = (
            "baseline calibration: the torch composition is the reference by definition, "
            "so this entry can only ever be 1.0 (kernel-level entries come with --impl)"
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
