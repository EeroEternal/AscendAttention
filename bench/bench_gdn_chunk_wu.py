#!/usr/bin/env python
"""``gdn_chunk_wu`` A/B: fused kernel vs ``solve_tril`` + ``recompute_w_u`` (torch ports).

Scope discipline: the baseline is exactly the two ops this kernel replaces, not the whole five-op
chunk chain.  TileInfer's ledger records 45.6x "isolated", 1.91x for the full chain and 1.04x
end-to-end; a ratio without its comparison set is not a result, so this script writes the set down.

Usage::

    source scripts/env-ascend102.sh
    python bench/bench_gdn_chunk_wu.py --t 256 --hv 4 --dim 128
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

DT = torch.bfloat16


def _inputs(t_total: int, hk: int, hv: int, dim: int, seq_lens, device):
    torch.manual_seed(0)
    from ascend_attention.ops.fla.kkt import chunk_scaled_dot_kkt_fwd
    from ascend_attention.ops.fla.solve_tril import prepare_chunk_indices

    k = torch.randn(1, t_total, hk, dim, dtype=DT, device=device) * 0.2
    v = torch.randn(1, t_total, hv, dim, dtype=DT, device=device) * 0.2
    beta = torch.rand(1, t_total, hv, dtype=DT, device=device) * 0.5
    g = torch.randn(1, t_total, hv, dtype=torch.float32, device=device).cumsum(-1) * -0.01
    cu = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), dtype=torch.int64, device=device
    )
    chunk_indices = prepare_chunk_indices(cu, 64)
    a_kkt = chunk_scaled_dot_kkt_fwd(
        k, beta, g, cu_seqlens=cu, chunk_indices=chunk_indices, chunk_size=64,
        output_dtype=torch.float32,
    )
    return a_kkt, k, v, beta, g, cu, chunk_indices


def _time(fn, iters: int) -> float:
    torch.npu.synchronize()
    fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", type=int, default=256)
    parser.add_argument("--hk", type=int, default=4)
    parser.add_argument("--hv", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seq-lens", default="128,128")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from ascend_attention import plan

    device = torch.device(args.device)
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    assert sum(seq_lens) == args.t
    a_kkt, k, v, beta, g, cu, ci = _inputs(
        args.t, args.hk, args.hv, args.dim, seq_lens, device
    )
    spec = dict(
        t_total=args.t,
        hv_heads=args.hv,
        hk_heads=args.hk,
        dim=args.dim,
        num_seqs=len(seq_lens),
        nt_chunks=int(ci.shape[0]),
        dtype="bfloat16",
    )
    p_kernel = plan("gdn_chunk_wu", impl="tilelang.chunk_wu950", **spec)
    p_torch = plan("gdn_chunk_wu", impl="torch.solve_tril_recompute_wu", **spec)

    def run_kernel():
        return p_kernel(a_kkt, k, v, beta, g, cu, ci)

    def run_torch():
        return p_torch(a_kkt, k, v, beta, g, cu, ci)

    w_k, u_k = run_kernel()
    w_t, u_t = run_torch()
    torch.npu.synchronize()
    result: Dict[str, Any] = {
        "op": "gdn_chunk_wu",
        "shape": {
            "t_total": args.t, "hk": args.hk, "hv": args.hv, "dim": args.dim,
            "seq_lens": seq_lens, "nt_chunks": int(ci.shape[0]),
        },
        "dtype": "bfloat16",
        "device": args.device,
        "pairs": args.pairs,
        "iters": args.iters,
        "rel_err": {
            "w": float((w_k.float() - w_t.float()).abs().max())
            / (float(w_t.float().abs().max()) or 1.0),
            "u": float((u_k.float() - u_t.float()).abs().max())
            / (float(u_t.float().abs().max()) or 1.0),
        },
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
        "op level: (solve_tril + recompute_w_u) / fused kernel, in-process alternating pairs, "
        "median of pair ratios. Comparison set is those two ops only - TileInfer's 45.6x is the "
        "isolated form and its 1.91x is the five-op chain, so this number equals neither"
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
