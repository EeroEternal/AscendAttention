"""Device numerics for the fused GDN ``solve_tril + recompute_w_u`` kernel (``npu`` marker).

Compared against the two torch ports it replaces, per head and per chunk: ``w = A_inv @ (k·β·e^g)``
and ``u = A_inv @ (v·β)``.  A tail chunk (sequence length not a multiple of 64) is included because
the kernel parks invalid rows in a trash row, and the whole point of that trick is that valid rows
must stay bit-comparable.
"""

from __future__ import annotations

import pytest
import torch

from ascend_attention.ops.gdn import WU_KERNEL
from ascend_attention.utils import is_npu_available

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
]

DT = torch.bfloat16
T, HK, HV, DIM = 256, 4, 4, 128
SEQ_LENS = [128, 96, 32]  # the last two are tail chunks (96 = 64 + 32)
TOL = 3e-2


def _inputs(device):
    from ascend_attention.ops.fla.kkt import chunk_scaled_dot_kkt_fwd
    from ascend_attention.ops.fla.solve_tril import prepare_chunk_indices

    torch.manual_seed(0)
    t_total = sum(SEQ_LENS)
    k = torch.randn(1, t_total, HK, DIM, dtype=DT, device=device) * 0.2
    v = torch.randn(1, t_total, HV, DIM, dtype=DT, device=device) * 0.2
    beta = torch.rand(1, t_total, HV, dtype=DT, device=device) * 0.5
    g = torch.randn(1, t_total, HV, dtype=torch.float32, device=device).cumsum(-1) * -0.01
    cu = torch.tensor([0] + list(torch.tensor(SEQ_LENS).cumsum(0).tolist()),
                      dtype=torch.int64, device=device)
    ci = prepare_chunk_indices(cu, 64)
    a = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu, chunk_indices=ci, chunk_size=64,
                                 output_dtype=torch.float32)
    return a, k, v, beta, g, cu, ci


def test_chunk_wu_matches_the_two_torch_ports():
    pytest.importorskip("tilelang")
    from ascend_attention import plan

    device = torch.device("npu", 0)
    a, k, v, beta, g, cu, ci = _inputs(device)
    spec = dict(
        t_total=sum(SEQ_LENS), hv_heads=HV, hk_heads=HK, dim=DIM,
        num_seqs=len(SEQ_LENS), nt_chunks=int(ci.shape[0]), dtype="bfloat16",
    )
    p_kernel = plan("gdn_chunk_wu", impl="tilelang.chunk_wu950", **spec)
    p_torch = plan("gdn_chunk_wu", impl="torch.solve_tril_recompute_wu", **spec)
    w_k, u_k = p_kernel(a, k, v, beta, g, cu, ci)
    w_t, u_t = p_torch(a, k, v, beta, g, cu, ci)
    torch.npu.synchronize()

    for name, got, want in (("w", w_k, w_t), ("u", u_k, u_t)):
        assert tuple(got.shape) == tuple(want.shape) == (1, sum(SEQ_LENS), HV, DIM)
        assert got.dtype == want.dtype
        scale = float(want.float().abs().max()) or 1.0
        err = float((got.float() - want.float()).abs().max())
        assert err <= TOL * scale, f"{name}: max_abs={err:.3e} vs scale {scale:.3e}"


def test_chunk_wu_kernel_is_measured_but_opt_in():
    """12.31x against two of the chain's ops; the chain and the step are not this number."""
    assert WU_KERNEL.enabled_by_default is False
    assert WU_KERNEL.measured_ratio and WU_KERNEL.measured_ratio > 1.0
    assert "solve_tril" in WU_KERNEL.ratio_note
