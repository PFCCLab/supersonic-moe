"""ncu profiling for individual FP8 kernels.

Uses same uniform routing as nsys_profile_comprehensive.py.
Usage:
  ncu --set roofline --kernel-name regex:<pattern> -o reports/ncu_xxx \
      python tools/ncu_profile_kernels.py --mode fp8
"""
import argparse
import os
import sys

os.environ["USE_QUACK_GEMM"] = "1"
for _k in [
    "PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
    "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
    "PADDLE_ELASTIC_TIMEOUT",
]:
    os.environ.pop(_k, None)

import torch
import torch.cuda.nvtx as nvtx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T, H, I, E, K = 4096, 4096, 1024, 128, 8


def build_uniform_routing(device):
    tok = torch.arange(T, device=device).unsqueeze(1)
    off = torch.arange(K, device=device).unsqueeze(0)
    indices = ((tok * K + off) % E).to(torch.int32)
    scores = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
    return scores, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bf16", "fp8"], default="fp8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "fp8":
        os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    else:
        os.environ.pop("SONIC_MOE_FP8_MODE", None)

    from sonicmoe import MoE, enable_quack_gemm
    from sonicmoe.enums import ActivationType
    import sonicmoe.functional as F_mod
    import sonicmoe.quack_utils.sm100_funcs as sm100_funcs

    torch.manual_seed(42)
    moe = MoE(
        num_experts=E, num_experts_per_tok=K, hidden_size=H,
        intermediate_size=I, activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02,
    ).to("cuda", torch.bfloat16)
    enable_quack_gemm()

    x_base = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    # Uniform routing patch
    _scores, _indices = build_uniform_routing(x_base.device)

    class _UniformRouter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, router_logits, E_arg, K_arg):
            ctx.save_for_backward(_scores, _indices)
            ctx.E = E_arg; ctx.dtype = router_logits.dtype
            return _scores.clone(), _indices.clone()
        @staticmethod
        def backward(ctx, grad_scores, _grad_indices):
            scores, _ = ctx.saved_tensors
            return torch.zeros(scores.size(0), ctx.E, dtype=ctx.dtype, device=scores.device), None, None

    orig_router = F_mod.TC_Softmax_Topk_Router_Function
    F_mod.TC_Softmax_Topk_Router_Function = _UniformRouter
    if args.mode == "fp8":
        sm100_funcs._ALIGNMENT_ASSUMED = True

    def zero_grads():
        for p in moe.parameters():
            p.grad = None

    # Warmup
    for _ in range(args.warmup):
        zero_grads()
        x_ = x_base.clone().requires_grad_(True)
        out, _ = moe(x_)
        out.sum().backward()
    torch.cuda.synchronize()
    print(f"Warmup done ({args.warmup} iters), starting profiled iterations...")

    # Profiled iterations with sync barriers for ncu
    for i in range(args.iters):
        zero_grads()
        x_ = x_base.clone().requires_grad_(True)

        torch.cuda.synchronize()
        nvtx.range_push("forward")
        out, _ = moe(x_)
        torch.cuda.synchronize()
        nvtx.range_pop()

        nvtx.range_push("backward")
        out.sum().backward()
        torch.cuda.synchronize()
        nvtx.range_pop()

    print("Done.")


if __name__ == "__main__":
    main()
