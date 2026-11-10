#!/usr/bin/env python
"""``sampling_topk_threshold``: vendor DeepSelect vs the torch sort-based threshold.

This is the first ``vendor_call`` op in the repo, so the bench doubles as the check that the vendor
window (bf16, 2-D, stride(1)==1, aligned row stride, scalar k) is really the window the vendor accepts.

Recorded ratio convention: **time ratio** (ours / baseline), like the source repo's table, so *lower is
faster*.  ``measured_ratio`` in the manifest uses the usual "baseline / ours" convention, so the JSON
carries both and says which is which - a factor-of-N flip is the easiest silent error in a ledger.

Usage::

    source scripts/env-ascend102.sh
    python bench/bench_sampling_topk.py --batch 8 --vocab 152064 --k 20
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


def _sync() -> None:
    torch.npu.synchronize()


def _time(fn, iters: int) -> float:
    _sync()
    fn()
    _sync()
    best = float("inf")
    for _ in range(iters):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        best = min(best, time.perf_counter() - start)
    return best


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=152064)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    try:
        import deep_select  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"result": "NOT APPLICABLE", "reason": f"deep_select missing: {exc}"}, indent=2))
        return 0

    from ascend_attention import plan

    device = torch.device(args.device)
    if device.type != "npu":
        print(json.dumps({"result": "NOT APPLICABLE", "reason": "the vendor path needs an NPU"}, indent=2))
        return 0
    torch.manual_seed(0)
    logits = torch.randn(args.batch, args.vocab, dtype=torch.bfloat16, device=device) * 2.0

    p_torch = plan(
        "sampling_topk_threshold", impl="torch.topk_threshold",
        batch=args.batch, vocab=args.vocab, k=args.k, dtype="bfloat16",
    )
    p_vendor = plan(
        "sampling_topk_threshold", impl="vendor.deep_select",
        batch=args.batch, vocab=args.vocab, k=args.k, dtype="bfloat16",
    )

    def run_torch():
        return p_torch(logits)

    def run_vendor():
        return p_vendor(logits)

    result: Dict[str, Any] = {
        "op": "sampling_topk_threshold",
        "shape": {"batch": args.batch, "vocab": args.vocab, "k": args.k},
        "dtype": "bfloat16",
        "device": args.device,
        "pairs": args.pairs,
        "iters": args.iters,
        "command": " ".join(sys.argv),
        "host": platform.node(),
        "date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "ratio_convention": (
            "time_ratio = vendor_ms / torch_ms (lower is faster); measured_ratio = torch / vendor "
            "(higher is better), the convention the manifest uses"
        ),
    }

    # the vendor returns (values, indices) without a guarantee of order; compare the *mask* view
    masked_t = run_torch()
    vendor_out = run_vendor()
    values = vendor_out[0] if isinstance(vendor_out, (tuple, list)) else vendor_out
    result["vendor_returns"] = type(vendor_out).__name__
    result["vendor_values_shape"] = list(values.shape)

    pairs: List[float] = []
    try:
        run_torch(), run_vendor()
        for index in range(max(4, args.pairs)):
            if index % 2 == 0:
                a, b = _time(run_torch, args.iters), _time(run_vendor, args.iters)
            else:
                b, a = _time(run_vendor, args.iters), _time(run_torch, args.iters)
            pairs.append(b / a)
        result["time_ratio"] = round(statistics.median(pairs), 4)
        result["pair_time_ratios"] = [round(r, 4) for r in pairs]
        result["measured_ratio"] = round(1.0 / result["time_ratio"], 4)
        result["ms"] = {
            "torch": round(statistics.median([_time(run_torch, args.iters) for _ in range(3)]) * 1e3, 4),
            "vendor": round(statistics.median([_time(run_vendor, args.iters) for _ in range(3)]) * 1e3, 4),
        }
    except Exception as exc:  # noqa: BLE001
        result["result"] = "VENDOR CALL FAILED"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, indent=2))
        return 0

    result["ratio_note"] = (
        "vendor_call vs torch threshold, in-process alternating pairs, median of per-pair time "
        "ratios. Numerics are only checked structurally here (mask shape/count); a full sampler "
        "equivalence test needs the token distribution, which belongs to P5"
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
