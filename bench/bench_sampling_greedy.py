#!/usr/bin/env python
"""``sampling_greedy``: absolute latency of the argmax baseline (there is nothing to compare against).

``measured_ratio`` is 1.0 *by definition* here — ``torch.argmax`` **is** the semantics — so this script
records latency and a parity check instead of inventing a ratio.  That is the honest entry for a
baseline whose whole job is to be the reference the threshold/greedy routing falls back to.

Usage::

    python bench/bench_sampling_greedy.py --batch 8 --vocab 152064
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
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=152064)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    import torch

    from ascend_attention import plan

    device = torch.device(args.device)
    if device.type == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            print(json.dumps({"result": "NOT APPLICABLE", "reason": "no NPU"}, indent=2))
            return 0
        sync = torch.npu.synchronize
    else:
        sync = lambda: None  # noqa: E731

    torch.manual_seed(0)
    logits = torch.randn(args.batch, args.vocab, dtype=torch.bfloat16, device=device)
    p = plan(
        "sampling_greedy",
        batch=args.batch,
        vocab=args.vocab,
        dtype="bfloat16",
        device=device,
    )

    def run():
        return p(logits)

    sync()
    run()
    sync()
    samples: List[float] = []
    for _ in range(args.iters):
        sync()
        start = time.perf_counter()
        out = run()
        sync()
        samples.append(time.perf_counter() - start)

    reference = torch.argmax(logits, dim=-1)
    result = {
        "op": "sampling_greedy",
        "impl": p.impl,
        "shape": {"batch": args.batch, "vocab": args.vocab},
        "dtype": "bfloat16",
        "device": str(device),
        "iters": args.iters,
        "ms_median": round(statistics.median(samples) * 1e3, 4),
        "ms_min": round(min(samples) * 1e3, 4),
        "measured_ratio": 1.0,
        "ratio_kind": "definitional parity (single implementation; argmax is the semantics)",
        "parity_vs_torch_argmax": bool(torch.equal(out, reference)),
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
    return 0 if result["parity_vs_torch_argmax"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
