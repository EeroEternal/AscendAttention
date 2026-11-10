#!/usr/bin/env python
"""``paged_decode_attention`` A/B ledger against the vendor path.

The comparable baseline is the op the engine would otherwise call: ``torch_npu.npu_fusion_attention``
(FIA), which is exactly what vLLM's Ascend backend uses, given **dense** KV — the form the engine has
when it is not paging.  Our kernel gets the **paged** KV.  Same math, same dtype, same shapes; the
only difference is which implementation runs, which is precisely the decision being made.

Protocol (repo contract, not decoration):

* both arms in one process, alternating order per pair, at least 4 pairs;
* median of per-pair ratios is the reported number;
* op level only — the end-to-end arm belongs to P5;
* the command, host and date are recorded with the number.

Exit code is 0 either way: a loss is a result, not a failure.  Copy the number into
``ops/attention/__init__.py`` only when the protocol above was actually followed.

Usage::

    source scripts/env-ascend102.sh
    python bench/bench_paged_decode.py --batch 8 --length 2048
    python bench/bench_paged_decode.py --batch 64 --length 2048 --json bench/results/decode_b64.json
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

DTYPE = torch.bfloat16


def _synchronize() -> None:
    torch.npu.synchronize()


def _time(fn, iters: int) -> float:
    """Seconds per call with the device synchronised on both sides (else the numbers are fiction)."""
    _synchronize()
    fn()
    _synchronize()
    best = float("inf")
    for _ in range(iters):
        _synchronize()
        start = time.perf_counter()
        fn()
        _synchronize()
        best = min(best, time.perf_counter() - start)
    return best


def _build(batch: int, kv_heads: int, group: int, page_size: int, dim: int, length: int, device):
    from ascend_attention.metadata import RaggedMetadata

    torch.manual_seed(0)
    pages_per_req = -(-length // page_size)
    total_pages = pages_per_req * batch
    k_cache = torch.randn(total_pages, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    v_cache = torch.randn(total_pages, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    indptr = torch.arange(0, total_pages + 1, pages_per_req, dtype=torch.int32, device=device)
    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    last = torch.full((batch,), page_size, dtype=torch.int32, device=device)
    meta = RaggedMetadata(
        kv_indptr=indptr, kv_indices=indices, kv_last_page_len=last, page_size=page_size
    )
    q = torch.randn(batch, kv_heads * group, 1, dim, dtype=DTYPE, device=device) * 0.2
    # The vendor arm needs dense KV.  Built by a copy, outside the timed region: it is the layout
    # the engine already holds when it does not page, so charging us for it would be unfair.
    dense_k = k_cache.reshape(batch, pages_per_req, page_size, kv_heads, dim)
    dense_k = dense_k.reshape(batch, length, kv_heads, dim).permute(0, 2, 1, 3).contiguous()
    dense_v = v_cache.reshape(batch, pages_per_req, page_size, kv_heads, dim)
    dense_v = dense_v.reshape(batch, length, kv_heads, dim).permute(0, 2, 1, 3).contiguous()
    return q, k_cache, v_cache, meta, dense_k, dense_v


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128, choices=[128, 256])
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--baseline", default="fia", choices=["fia", "none"])
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--json", default=None, help="also write the result to this path")
    args = parser.parse_args(argv)

    from ascend_attention import plan

    device = torch.device(args.device)
    q, k_cache, v_cache, meta, dense_k, dense_v = _build(
        args.batch, args.kv_heads, args.group, args.page_size, args.dim, args.length, device
    )
    p = plan(
        "paged_decode_attention",
        impl="tilelang.decode950",
        k_cache=k_cache,
        v_cache=v_cache,
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=args.page_size,
        num_qo_heads=args.kv_heads * args.group,
        num_kv_heads=args.kv_heads,
        head_dim=args.dim,
        batch=args.batch,
        dtype="bfloat16",
        device=device,
    )

    def ours():
        return p(q)

    def fia():
        import torch_npu

        return torch_npu.npu_fusion_attention(
            q, dense_k, dense_v, head_num=args.kv_heads * args.group, input_layout="BNSD",
            scale=args.dim**-0.5,
        )[0]

    result: Dict[str, Any] = {
        "op": "paged_decode_attention",
        "impl": "tilelang.decode950",
        "shape": {
            "batch": args.batch,
            "length": args.length,
            "kv_heads": args.kv_heads,
            "group": args.group,
            "head_dim": args.dim,
            "page_size": args.page_size,
        },
        "dtype": "bfloat16",
        "device": args.device,
        "pairs": args.pairs,
        "iters": args.iters,
        "command": " ".join(sys.argv),
        "host": platform.node(),
        "date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "measured_ratio": None,
        "ratio_note": "",
    }

    if args.baseline == "none":
        kernel_ms = statistics.median([_time(ours, args.iters) for _ in range(3)]) * 1e3
        result["ms"] = {"kernel": round(kernel_ms, 4)}
        result["ratio_note"] = "absolute latency only; no comparable baseline was run"
    else:
        pair_ratios: List[float] = []
        timings: List[Dict[str, float]] = []
        ours()
        fia()  # warm both (kernel bind + vendor workspace)
        for index in range(max(4, args.pairs)):
            if index % 2 == 0:
                a, b = _time(ours, args.iters), _time(fia, args.iters)
            else:
                b, a = _time(fia, args.iters), _time(ours, args.iters)
            pair_ratios.append(b / a)
            timings.append({"ours_ms": round(a * 1e3, 4), "fia_ms": round(b * 1e3, 4)})
        ratio = statistics.median(pair_ratios)
        result["measured_ratio"] = round(ratio, 4)
        result["pair_ratios"] = [round(r, 4) for r in pair_ratios]
        result["timings"] = timings
        result["ms"] = {
            "ours": round(statistics.median(t["ours_ms"] for t in timings), 4),
            "fia": round(statistics.median(t["fia_ms"] for t in timings), 4),
        }
        result["ratio_note"] = (
            "op level: vendor FIA (dense KV) / our kernel (paged KV), in-process alternating "
            "pairs, median of per-pair ratios; end-to-end not measured by this script"
        )
        # Numerics are checked by tests/, but a silent mismatch would make the ratio meaningless.
        max_err = (ours().float() - fia().float()).abs().max().item()
        result["max_abs_diff_vs_fia"] = float(max_err)

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
