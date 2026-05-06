#!/usr/bin/env python3
"""Dump real dz tensors from the SonicMoE bwd dGated path for iso32 audit.

Hooks ``sonicmoe.functional.__init__.gemm_dgated_kernel`` so that the first
N kernel invocations after warmup save their ``dz`` output to disk. We then
have authentic gradient-domain bf16 tensors with which to compare 1×32
vs 32×32 quant fidelity, including downstream-GEMM error.

Outputs: reports/iso32_dz_audit/dz_iter{i}.pt  (bf16 CUDA tensor pickled to CPU)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=8192)
    p.add_argument("--H", type=int, default=3072)
    p.add_argument("--I", type=int, default=1536)
    p.add_argument("--E", type=int, default=8)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--captures", type=int, default=3,
                   help="Number of dz captures to save (one per backward pass).")
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(_REPO, "reports", "iso32_dz_audit"))
    p.add_argument("--imbalance", type=str, default="none",
                   choices=["none", "skew", "extreme"],
                   help="Routing distribution.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import paddle
    paddle.compat.enable_torch_proxy(silent=True)
    import torch

    from sonicmoe.ernie_compat import (
        SonicMoEMlpNode, flush_native_grads, invalidate_weight_caches,
    )
    import sonicmoe.functional as functional

    # Patch the dGated kernel call site by wrapping the imported alias inside
    # ``sonicmoe.functional``.  We capture ``dz`` (the 3rd positional output
    # tensor written in-place by the kernel) right after the call returns.
    fmod = functional
    orig = fmod.gemm_dgated_kernel
    state = {"backward_idx": 0, "saved": 0}
    n_captures = args.captures

    def hooked(dout_fp8, w2_fp8_enk, dz, *args_, **kwargs):
        rv = orig(dout_fp8, w2_fp8_enk, dz, *args_, **kwargs)
        if state["saved"] < n_captures and state["backward_idx"] >= args.warmup:
            torch.cuda.synchronize()
            amax = dz.detach().abs().max().item()
            cpu_dtype = dz.detach().to(dtype=torch.float32, device="cpu").to(torch.bfloat16)
            import numpy as np
            np_path = os.path.join(args.out_dir,
                                   f"dz_{args.imbalance}_iter{state['backward_idx']}.npy")
            np.save(np_path,
                    dz.detach().to(dtype=torch.float32, device="cpu").numpy())
            print(f"  [capture] saved {np_path}  shape={tuple(cpu_dtype.shape)} "
                  f"dtype={cpu_dtype.dtype} amax={amax:.4e}")
            state["saved"] += 1
        return rv

    args_captures = args.captures  # legacy
    fmod.gemm_dgated_kernel = hooked

    functional._ALIGNMENT_ASSUMED = True
    functional._ALIGNMENT_STREAK = 100

    H, I, E, topk, T = args.H, args.I, args.E, args.topk, args.T

    class MockExpert:
        def __init__(self, h, i, seed):
            paddle.seed(seed)
            self.up_gate_proj = type("P", (), {
                "weight": paddle.randn([h, 2 * i], dtype="bfloat16") / math.sqrt(h),
            })()
            self.down_proj = type("P", (), {
                "weight": paddle.randn([i, h], dtype="bfloat16") / math.sqrt(i),
            })()
            self.up_gate_proj.weight.stop_gradient = False
            self.down_proj.weight.stop_gradient = False

    experts = [MockExpert(H, I, e) for e in range(E)]
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E,
                           hidden_size=H, intermediate_size=I)

    torch.manual_seed(0)
    if args.imbalance == "extreme":
        idx = torch.arange(topk, device="cuda", dtype=torch.int32)
        di = idx.unsqueeze(0).expand(T, topk).contiguous()
    elif args.imbalance == "skew":
        raw = torch.randn(T, E, device="cuda")
        hot = (torch.rand(T, device="cuda") < 0.8)
        raw[hot, 0] += 100.0
        _, top_e = raw.topk(topk, dim=-1)
        di = top_e.int()
    else:
        raw = torch.randn(T, E, device="cuda")
        _, top_e = raw.topk(topk, dim=-1)
        di = top_e.int()
    dp = torch.rand(T, topk, device="cuda") * 0.5 + 0.5
    dp = (dp / dp.sum(dim=1, keepdim=True)).float()
    tpe = [int((di == e).sum().item()) for e in range(E)]

    paddle.seed(0)
    # Use realistic-scale activations and gradients (matches Ernie-shape unit
    # variance after layernorm; unscaled randn so dz exercises a meaningful
    # exponent range rather than collapsing to ~3e-4).
    x = paddle.randn([T, H], dtype="bfloat16")
    grad_out = paddle.randn([T, H], dtype="bfloat16")

    n_iters = args.warmup + args.captures + 1
    print(f"Running {n_iters} fwd+bwd iters; warmup={args.warmup}, "
          f"captures={args.captures}")
    for it in range(n_iters):
        state["backward_idx"] = it
        out = node.forward(x, tpe, dispatched_indices=di, dispatched_probs=dp)
        out.backward(grad_out)
        if it < args.warmup:
            print(f"  warmup iter {it}")
        elif state["saved"] >= args.captures:
            break
    flush_native_grads()
    torch.cuda.synchronize()
    print(f"Done. Saved {state['saved']} dz captures to {args.out_dir}")


if __name__ == "__main__":
    main()
