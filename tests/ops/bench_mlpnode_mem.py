#!/usr/bin/env python
"""SonicMoEMlpNode 单次前反向显存基准

用法:
    CUDA_VISIBLE_DEVICES=0 python tests/ops/bench_mlpnode_mem.py

配置（默认值对应 ERNIE 真实业务规格）:
    H=3072  I=1536  K=8  E_LOCAL=8  EP_SIZE=32  SEQ_LEN=16384

精度策略：
    - 前向/actgrad/dgated：FP8 (USE_QUACK_GEMM=1)
    - wgrad (dw1/dw2)：FP8（auto-detect，threshold=0 → 全 I 开启）
    - z 保存格式：fp8（save_z_fp8=True，默认）
"""
import os
import sys

# ── 自动切换到 eb_venv ───────────────────────────────────────────────────────
_VENV = os.environ.get("SONIC_MOE_PADDLE_VENV", sys.prefix)
if os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    print(f"\033[33mSwitch venv: {_VENV}\033[0m")
    os.execv(f"{_VENV}/bin/python", [f"{_VENV}/bin/python", *sys.argv])

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
for _p in (_QUACK, _REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import math
import logging
import time

import numpy as np
import paddle
paddle.enable_compat()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

import sonicmoe
print(f"\033[33mUsing sonicmoe: {sonicmoe.__path__}\033[0m")
from sonicmoe.config import SonicMoEConfig
from sonicmoe.ernie_compat import (
    SonicMoEMlpNode,
    flush_native_grads,
    invalidate_weight_caches,
)

# ── 形状参数 ─────────────────────────────────────────────────────────────────
H        = 3072    # hidden size
I        = 1536    # expert intermediate size
K        = 8       # 全局 topk
E_LOCAL  = 32      # 本卡专家数
EP_SIZE  = 8       # expert parallel size
SEQ_LEN  = 4096    # 通信前每机 token 数


# ── 路由数据生成（真实 EP 路由分布，来自 test_moe_general_routing_low_level）──
def make_inputs(n_experts, hidden_size, topk, ep_size, seq_len):
    """模拟真实 EP dispatch 后本卡收到的 token 分布。

    每个 token 从 ep_size * n_experts 个全局专家中选 topk，
    只保留落到本卡（expert_id < n_experts）的部分。
    返回:
        x                : [N_recv, H] bfloat16，收到的 token
        dispatched_indices: [N_recv, topk] int32，本卡专家索引（-1 为无效）
        dispatched_probs  : [N_recv, topk] float32，路由概率
        tokens_per_expert : list[int]，每个专家收到的 token 数
    """
    tokens_per_expert = [0] * n_experts
    dispatched_indices = []
    dispatched_probs   = []

    for _ in range(ep_size * seq_len):
        global_logits = np.random.normal(size=ep_size * n_experts).astype("float32")
        global_logits -= global_logits.max()
        global_probs   = np.exp(global_logits)
        global_probs  /= global_probs.sum()

        topk_unsorted = np.argpartition(global_probs, -topk)[-topk:]
        topk_sorted   = topk_unsorted[np.argsort(global_probs[topk_unsorted])[::-1]]
        choices       = topk_sorted[np.random.permutation(topk)]
        choice_probs  = global_probs[choices]

        local_choices = []
        local_probs   = []
        for expert_id, prob in zip(choices.tolist(), choice_probs.tolist()):
            if expert_id < n_experts:
                tokens_per_expert[expert_id] += 1
                local_choices.append(expert_id)
                local_probs.append(prob)

        if local_choices:
            dispatched_indices.append(local_choices + [-1] * (topk - len(local_choices)))
            dispatched_probs.append(local_probs   + [0.0] * (topk - len(local_probs)))

    dispatched_indices = paddle.to_tensor(dispatched_indices, "int32")
    dispatched_probs = paddle.to_tensor(dispatched_probs,   "float32")
    x = paddle.randn([dispatched_indices.shape[0], hidden_size], "bfloat16")
    return x, dispatched_indices, dispatched_probs, tokens_per_expert


# ── MockExpert（最小专家模块，对齐 ERNIE per-expert 参数结构）────────────────
class MockExpert:
    """提供 up_gate_proj.weight [H, 2I] 和 down_proj.weight [I, H]。"""
    def __init__(self, h: int, i: int, seed: int):
        paddle.seed(seed)
        scale = 1.0 / math.sqrt(h)
        self.up_gate_proj = type("_W", (), {
            "weight": paddle.randn([h, 2 * i], dtype="bfloat16") * scale,
        })()
        self.down_proj = type("_W", (), {
            "weight": paddle.randn([i, h], dtype="bfloat16") * scale,
        })()
        self.up_gate_proj.weight.stop_gradient = False
        self.down_proj.weight.stop_gradient    = False


def _mem_mib() -> float:
    return paddle.device.cuda.memory_allocated() / 1024 / 1024

def _peak_mib() -> float:
    return paddle.device.cuda.max_memory_allocated() / 1024 / 1024

def _print_mem(tag: str):
    print(f"\033[31m[{tag}] allocated={_mem_mib():.0f} MiB peak={_peak_mib():.0f} MiB\033[0m")
    paddle.device.cuda.reset_max_memory_allocated()


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    paddle.seed(42)
    np.random.seed(0)
    paddle.device.cuda.reset_max_memory_allocated()

    # 1. 生成路由数据
    t0 = time.time()
    print(f"生成路由数据 (EP={EP_SIZE}, SEQ={SEQ_LEN}, E_local={E_LOCAL}, K={K}) ...")
    x, dispatched_indices, dispatched_probs, tokens_per_expert = make_inputs(
        n_experts=E_LOCAL,
        hidden_size=H,
        topk=K,
        ep_size=EP_SIZE,
        seq_len=SEQ_LEN,
    )
    N_recv   = x.shape[0]
    TK       = sum(tokens_per_expert)
    TK_pad   = sum((c + 127) // 128 * 128 for c in tokens_per_expert)
    pad_rows = TK_pad - TK
    print(f"  耗时 {time.time()-t0:.1f}s")
    print(f"  N_recv={N_recv}  TK={TK}  TK_padded={TK_pad}  pad_rows={pad_rows}")
    print(f"  tokens_per_expert={tokens_per_expert}")
    print(f"  x={list(x.shape)} dispatched_indices={list(dispatched_indices.shape)}")

    # 理论激活显存估算
    x_mib    = N_recv * H * 2 / 1024**2
    z_mib    = TK_pad * (2*I) * 1 / 1024**2          # fp8
    zsc_mib  = TK_pad * (2*I // 32) * 1 / 1024**2    # e8m0 scales
    out_mib  = N_recv * H * 2 / 1024**2
    print(f"\n  理论激活估算 (保存到反向的):")
    print(f"    x        [N_recv={N_recv}, H={H}]       bf16 = {x_mib:.1f} MiB")
    print(f"    z_fp8    [TK_pad={TK_pad}, 2I={2*I}]  fp8  = {z_mib:.1f} MiB")
    print(f"    z_scales [TK_pad={TK_pad}, 2I/32={2*I//32}] e8m0 = {zsc_mib:.1f} MiB")
    print(f"    output   [N_recv={N_recv}, H={H}]       bf16 = {out_mib:.1f} MiB")
    print(f"    合计激活 (不含权重/cache): {x_mib+z_mib+zsc_mib+out_mib:.1f} MiB")

    # 理论临时峰值（前向）
    y1_mib   = TK_pad * I * 2 / 1024**2      # bf16 y1
    y2_mib   = TK_pad * H * 2 / 1024**2      # bf16 y2 (DownProj输出)
    print(f"\n  理论前向额外临时峰值:")
    print(f"    y1 [TK_pad={TK_pad}, I={I}]  bf16 = {y1_mib:.1f} MiB  (UpProj→DownProj，用完即扔)")
    print(f"    y2 [TK_pad={TK_pad}, H={H}]  bf16 = {y2_mib:.1f} MiB  (scatter前，用完即扔)")

    # 理论临时峰值（反向）
    dz_mib   = TK_pad * (2*I) * 2 / 1024**2  # bf16 dz
    y1s_mib  = TK_pad * I * 2 / 1024**2      # bf16 y1s (GemmDGated重算)
    dxe_mib  = TK_pad * H * 2 / 1024**2      # bf16 dx_expanded
    dw1_mib  = (2*I) * H * E_LOCAL * 2 / 1024**2   # bf16 dw1
    dw2_mib  = H * I * E_LOCAL * 2 / 1024**2        # bf16 dw2
    ng1_mib  = E_LOCAL * (2*I) * H * 4 / 1024**2   # fp32 _NATIVE_W1_GRAD
    ng2_mib  = E_LOCAL * H * I * 4 / 1024**2        # fp32 _NATIVE_W2_GRAD
    print(f"\n  理论反向额外临时峰值:")
    print(f"    dz      [TK_pad={TK_pad}, 2I={2*I}]  bf16 = {dz_mib:.1f} MiB")
    print(f"    y1s     [TK_pad={TK_pad}, I={I}]   bf16 = {y1s_mib:.1f} MiB  (GemmDGated重算)")
    print(f"    dx_exp  [TK_pad={TK_pad}, H={H}]   bf16 = {dxe_mib:.1f} MiB  (actgrad→scatter前)")
    print(f"    dw1     [2I×H×E={2*I}×{H}×{E_LOCAL}]  bf16 = {dw1_mib:.1f} MiB")
    print(f"    dw2     [H×I×E={H}×{I}×{E_LOCAL}]   bf16 = {dw2_mib:.1f} MiB")
    print(f"    NATIVE_W1_GRAD (fp32 acc) = {ng1_mib:.1f} MiB")
    print(f"    NATIVE_W2_GRAD (fp32 acc) = {ng2_mib:.1f} MiB")

    x.stop_gradient = False
    _print_mem("数据就绪")

    # 2. 初始化权重和 MlpNode
    experts = [MockExpert(H, I, seed=e) for e in range(E_LOCAL)]
    node    = SonicMoEMlpNode(
        experts=experts,
        n_experts=E_LOCAL,
        hidden_size=H,
        intermediate_size=I,
    )
    invalidate_weight_caches()

    # 3. 精度配置：FP8 全路径（含 wgrad）
    cfg = SonicMoEConfig(
        use_fp8        = True,
        assume_aligned = True,
        save_z_fp8     = True,
        stagewise_memory = True,  # 开启阶段显存 log
    )

    # 4. 前向
    _print_mem("前向开始前")
    with cfg.activate():
        output = node.forward(x, tokens_per_expert, dispatched_indices, dispatched_probs)
    _print_mem("前向结束后（ctx 已保存，反向未开始）")
    print(f"  output: {list(output.shape)}  dtype={output.dtype}")

    # 5. 反向
    grad_out = paddle.randn_like(output)
    _print_mem("反向开始前")
    output.backward(grad_out)
    # flush_native_grads() 不再需要：wgrad 路径直接写 main_grad
    _print_mem("反向结束后（含 flush_native_grads）")

    # 6. 基本健全性检查
    assert not paddle.any(paddle.isnan(output)), "output NaN"
    assert x.grad is not None, "dx is None"
    assert not paddle.any(paddle.isnan(x.grad)), "dx NaN"
    for e_idx, exp in enumerate(experts):
        assert exp.up_gate_proj.weight.main_grad is not None, f"dw1[{e_idx}] None"
        assert exp.down_proj.weight.main_grad    is not None, f"dw2[{e_idx}] None"
    print("PASS")

    #############################################################################
    x.clear_gradient()
    dispatched_probs.clear_gradient()
    print("dispatched_probs.grad:", dispatched_probs.grad)
    paddle.device.reset_max_memory_allocated()
    print(f"\033[31m{'=' * 80}\033[0m")

    _print_mem("前向开始前")
    with cfg.activate():
        output = node.forward(x, tokens_per_expert, dispatched_indices, dispatched_probs)
    _print_mem("前向结束后（ctx 已保存，反向未开始）")

    # 5. 反向
    _print_mem("反向开始前")
    output.backward(grad_out)
    _print_mem("反向结束后（含 flush_native_grads）")
    print(f"\033[31m{'=' * 80}\033[0m")

    #############################################################################
    paddle.base.core.nvprof_start()

    for i in range(10):
        x.clear_gradient(False)
        dispatched_probs.clear_gradient(False)
        paddle.randn([1024, 1024, 1024])
        paddle.randn([1024, 1024, 1024])

        paddle.base.core.nvprof_nvtx_push("sonic_fw")
        with cfg.activate():
            output = node.forward(x, tokens_per_expert, dispatched_indices, dispatched_probs)
        paddle.base.core.nvprof_nvtx_pop()

        paddle.base.core.nvprof_nvtx_push("sonic_bw")
        output.backward(grad_out)
        paddle.base.core.nvprof_nvtx_pop()

    paddle.base.core.nvprof_stop()
    invalidate_weight_caches()


if __name__ == "__main__":
    main()
