#!/usr/bin/env python3
"""
Sanity-check custom rms_norm_ascend_c vs reference RMSNorm and torch_npu on NPU.

Run from repo root or this directory:
  export enable_custom_op=true
  python csrc/ascend_c/rms_norm/demo_rms_norm_accuracy.py
"""
from __future__ import annotations

import os
import sys

# Repo root (nano-vllm-ascend): .../csrc/ascend_c/rms_norm -> parents[3]
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    import torch

    try:
        import torch_npu
    except ImportError:
        print("torch_npu not available; need NPU for this demo.")
        return 1

    if not torch.npu.is_available():
        print("NPU not available.")
        return 1

    device = torch.device("npu:0")
    hidden = 4096
    eps = 1e-6
    torch.manual_seed(0)

    # Load built extension (same path as nanovllm.custom_op.rms_norm_custom_ascend_c)
    so_path = os.path.join(_REPO_ROOT, "csrc", "build", "rms_norm_ascend_c", "rms_norm_ascend_c.so")
    if not os.path.isfile(so_path):
        print(f"Missing {so_path}\nRun: python {_REPO_ROOT}/csrc/ascend_c/rms_norm/build_rms_norm_ascend_c.py")
        return 1
    torch.ops.load_library(so_path)

    from nanovllm.layers.layernorm import AscendCustomRMSNorm, RMSNorm, TorchNPURMSNorm

    x = torch.randn(3, 7, hidden, device=device, dtype=torch.float16)
    res = torch.randn_like(x)
    gamma = torch.ones(hidden, device=device, dtype=torch.float16)

    ref_mod = RMSNorm(hidden, eps=eps).to(device)
    ref_mod.weight.data.copy_(gamma)

    asc_mod = AscendCustomRMSNorm(hidden, eps=eps).to(device)
    asc_mod.weight.data.copy_(gamma)

    npu_mod = TorchNPURMSNorm(hidden, eps=eps).to(device)
    npu_mod.weight.data.copy_(gamma)

    # --- rms only ---
    y_ref = ref_mod.rms_forward(x.clone())
    y_asc = asc_mod.rms_forward(x.clone())
    y_npu = npu_mod.rms_forward(x.clone())
    d1 = _max_abs(y_ref, y_asc)
    d2 = _max_abs(y_ref, y_npu)
    print(f"[rms_forward] max|ref - custom|: {d1:.6g}  max|ref - torch_npu|: {d2:.6g}")
    if d1 > 0.02:
        print("WARN: custom rms_forward diverges from float reference (adjust if needed).")

    # --- add + rms (pre-norm path) ---
    xa, ra = x.clone(), res.clone()
    xb, rb = x.clone(), res.clone()
    xc, rc = x.clone(), res.clone()

    o_ref, r_out_ref = ref_mod.add_rms_forward(xa, ra)
    o_asc, r_out_asc = asc_mod.add_rms_forward(xb, rb)
    o_npu, r_out_npu = npu_mod.add_rms_forward(xc, rc)

    d3 = _max_abs(o_ref, o_asc)
    d4 = _max_abs(r_out_ref, r_out_asc)
    d5 = _max_abs(o_ref, o_npu)
    print(
        f"[add_rms_forward] max|ref - custom| out: {d3:.6g}  residual: {d4:.6g}  max|ref - npu|: {d5:.6g}"
    )
    if d3 > 0.05 or d4 > 0.01:
        print("FAIL: add_rms_forward mismatch (functional bug likely).")
        return 2

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
