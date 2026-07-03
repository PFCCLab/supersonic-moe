"""
Unit test for MoELayer single-card implementation,
adapted from PaddleFleet's moe_layer.py.

Three switchable expert compute paths:
  1. Original: moe_general_routing_inputs (argsort-based)
  2. deepep_metadata: CUDA metadata kernel + direct _Up/_DownProjection
  3. mlpnode_v2: SonicMoEMlpNode from ernie_compat (full ERNIE drop-in path)

Usage:
  python tests/test_moe_layer.py                    # correctness (all 3 paths)
  python tests/test_moe_layer.py --perf             # perf benchmark
  python tests/test_moe_layer.py --perf --H 3072 --I 1536  # match README shape
"""
import argparse
import gc
import sys
import time
from dataclasses import dataclass
from functools import partial

import paddle
from paddle import nn
import paddle.nn.functional as F

# ---------------------------------------------------------------------------
# Supersonic-MoE imports (via Paddle torch-proxy)
# ---------------------------------------------------------------------------
paddle.compat.enable_torch_proxy(
    scope={"sonicmoe", "quack", "triton"}, silent=True
)

from sonicmoe.enums import ActivationType
from sonicmoe.functional import (
    moe_general_routing_inputs,
    clear_all_fp8_weight_caches,
    _refresh_fp8_config,
)
from sonicmoe.functional.utils import enable_fp8

# Optimized paths
from sonicmoe.ernie_compat.deepep_metadata import deepep_topk_to_sonic_metadata
from sonicmoe.functional import _UpProjection, _DownProjection
from sonicmoe.ernie_compat.mlp_node_v2 import SonicMoEMlpNode, flush_native_grads, invalidate_weight_caches


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MoETestConfig:
    seq_len: int = 8192
    scoring_func: str = "softmax"
    hidden_size: int = 1536
    n_routed_experts: int = 8
    n_shared_experts: int = 0
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    hidden_act: str = "silu"
    gated_linear_unit: bool = True
    moe_intermediate_size: int = 1024
    use_bias: bool = False
    moe_grouped_gemm: bool = True
    router_aux_loss_coef: float = 0.0
    fp8: bool = True
    using_sonic_moe: bool = True

    # Expert path switch: "original" | "deepep_metadata" | "mlpnode_v2"
    expert_path: str = "original"


# ---------------------------------------------------------------------------
# Gate (simplified TopKRouter)
# ---------------------------------------------------------------------------
class Gate(paddle.nn.Layer):
    """Simplified top-K gate matching PaddleFleet's TopKRouter interface."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self._cast_to_low_precision = False

        if config.scoring_func == "softmax":
            self.gate_score_func = partial(F.softmax, axis=-1)
        elif config.scoring_func == "sigmoid":
            self.gate_score_func = F.sigmoid
        else:
            raise NotImplementedError(f"{config.scoring_func} not implemented")

        self.weight = self.create_parameter(
            shape=[self.num_experts, config.hidden_size],
            dtype="float32",
        )

    def forward(self, hidden_states):
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])

        with paddle.amp.auto_cast(False):
            # PaddleFleet stores weight as [E, H] and transposes before F.linear
            logits = F.linear(hidden_states.cast("float32"), self.weight.T)

        gates = self.gate_score_func(logits)

        top_gate, top_idx = paddle.topk(gates, self.num_experts_per_tok, axis=-1)

        # Build mask and gates_masked (same as PaddleFleet TopKRouter)
        mask = paddle.zeros_like(gates).put_along_axis(
            top_idx, paddle.to_tensor(1.0, dtype=gates.dtype), axis=1
        )
        gates_masked = gates * mask

        # Normalize topk probs
        if self.norm_topk_prob:
            denominator = top_gate.sum(axis=-1, keepdim=True) + 1e-20
            top_gate = top_gate / denominator

        if abs(self.routed_scaling_factor - 1.0) > 1e-6:
            top_gate = top_gate * self.routed_scaling_factor
            gates_masked = gates_masked * self.routed_scaling_factor

        # Aux loss (simplified — zero)
        aux_loss = paddle.zeros([1], dtype="float32")
        aux_loss.stop_gradient = False

        return (
            None,          # capacity
            top_gate,      # topk_weights  [T, K]
            top_idx,       # topk_indices  [T, K]
            gates_masked,  # gates_masked  [T, E]
            mask,          # routing_map   [T, E]
            None,          # priorities
            aux_loss,
            None,          # z_loss
        )


# ---------------------------------------------------------------------------
# GroupedMLPExpert — batched weight storage for single-card grouped GEMM
# ---------------------------------------------------------------------------
class GroupedMLPExpert(paddle.nn.Layer):
    """Stores stacked expert weights [E, ...] for grouped GEMM or SonicMoE."""

    def __init__(self, num_local_experts, config):
        super().__init__()
        self.config = config
        self.num_local_experts = num_local_experts
        self.using_sonic_moe = config.using_sonic_moe

        fc1_output_size = config.moe_intermediate_size
        if config.gated_linear_unit:
            fc1_output_size *= 2
        fc2_input_size = config.moe_intermediate_size

        initializer = paddle.nn.initializer.Uniform(-0.001, 0.001)
        dtype = "bfloat16"

        # moe_general_routing_inputs expects w1 as [2I, H, E], w2 as [H, I, E]
        if self.using_sonic_moe:
            w1_shape = [fc1_output_size, config.hidden_size, num_local_experts]
            w2_shape = [config.hidden_size, fc2_input_size, num_local_experts]
        else:
            w1_shape = [num_local_experts, config.hidden_size, fc1_output_size]
            w2_shape = [num_local_experts, fc2_input_size, config.hidden_size]

        self.weight1 = paddle.create_parameter(
            shape=w1_shape, dtype=dtype, default_initializer=initializer,
        )
        self.weight2 = paddle.create_parameter(
            shape=w2_shape, dtype=dtype, default_initializer=initializer,
        )

    def forward(self, permuted_hidden_states, tokens_per_expert):
        """Standard grouped GEMM forward (non-SonicMoE path)."""
        tokens_per_expert = tokens_per_expert.cpu().tolist()
        tokens_per_expert = [int(x) for x in tokens_per_expert]

        fc1_output = paddle.incubate.nn.functional.batched_gemm(
            permuted_hidden_states, self.weight1, tokens_per_expert
        )

        # Gated activation: SwiGLU
        x, gate = fc1_output.chunk(2, axis=-1)
        intermediate = F.silu(gate) * x

        fc2_output = paddle.incubate.nn.functional.batched_gemm(
            intermediate, self.weight2, tokens_per_expert
        )
        return fc2_output


# ---------------------------------------------------------------------------
# Utility: permute / unpermute (from PaddleFleet moe_utils)
# ---------------------------------------------------------------------------
def permute(tokens, routing_map, tokens_per_expert=None):
    """Reorder tokens so tokens for the same expert are contiguous."""
    routing_map = routing_map.astype("bool")
    # sorted_indices: for each (token, expert) pair that is active, the token index
    sorted_indices = paddle.where(routing_map.T)[1].cast("int64")
    permuted_tokens = tokens.index_select(sorted_indices, axis=0)
    return permuted_tokens, sorted_indices


def unpermute(permuted_tokens, sorted_indices, restore_shape, probs=None, routing_map=None):
    """Reverse permutation, applying routing probs as weights."""
    output = paddle.zeros(restore_shape, dtype=permuted_tokens.dtype)
    if probs is not None and routing_map is not None:
        routing_map = routing_map.astype("bool")
        # Gather weights in permuted order
        weight_map = probs * routing_map.astype(probs.dtype)
        weights = weight_map.T[routing_map.T].unsqueeze(-1)
        permuted_tokens = permuted_tokens * weights.cast(permuted_tokens.dtype)
    output = paddle.scatter(
        output, sorted_indices.reshape([-1]), permuted_tokens, overwrite=False
    )
    return output


# ---------------------------------------------------------------------------
# Padding helper — 128-alignment for FP8 kernels
# ---------------------------------------------------------------------------
BLOCK = 128


def _make_mock_experts(weight1, weight2, E, H, I):
    """Wrap stacked weights [2I,H,E]/[H,I,E] into per-expert objects for SonicMoEMlpNode."""
    experts = []
    for e in range(E):
        # SonicMoEMlpNode expects experts with .up_gate_proj.weight [H, 2I]
        # and .down_proj.weight [I, H].
        # weight1 is [2I, H, E] → expert e's up_gate is weight1[:, :, e].T = [H, 2I]
        # weight2 is [H, I, E] → expert e's down is weight2[:, :, e].T = [I, H]
        #
        # But SonicMoE interleaves gate/up in w1. The stacked weight1[:,:,e] is
        # already in interleaved [2I, H] layout. stack_ernie_w1 expects split-half
        # [H, 2I] ERNIE layout and converts to interleaved internally.
        # We need to reverse: interleaved [2I, H] → split-half [H, 2I].
        w1_e = weight1[:, :, e]  # [2I, H] interleaved
        w1_gate = w1_e[0::2, :]  # [I, H] gate (even rows)
        w1_up = w1_e[1::2, :]    # [I, H] up (odd rows)
        w1_split = paddle.concat([w1_gate, w1_up], axis=0).T  # [H, 2I] split-half

        w2_e = weight2[:, :, e].T  # [I, H]

        exp = type("MockExpert", (), {})()
        exp.up_gate_proj = type("P", (), {"weight": w1_split})()
        exp.down_proj = type("P", (), {"weight": w2_e})()
        exp.up_gate_proj.weight.stop_gradient = False
        exp.down_proj.weight.stop_gradient = False
        experts.append(exp)
    return experts


def _prepare_sonic_inputs(
    x: paddle.Tensor,
    dispatched_indices: paddle.Tensor,
    dispatched_probs: paddle.Tensor,
    n_local_experts: int,
    block: int = BLOCK,
):
    """Convert routing outputs to the format expected by moe_general_routing_inputs.

    Each expert with a non-block-aligned token count receives virtual (zero-weight)
    padding tokens so that every expert's count is a multiple of ``block`` (128).

    Returns
    -------
    x_padded       : [T + max_pad, H]               bfloat16
    token_indices  : [TK_valid + n_pad_pairs]        int32, sorted ascending
    expert_indices : [TK_valid + n_pad_pairs]        int32
    router_scores  : [TK_valid + n_pad_pairs]        float32
    """
    T = x.shape[0]

    # 1. Flatten valid (token, expert, score) triples
    tok_ids = paddle.arange(T, dtype="int32").unsqueeze(1).expand_as(dispatched_indices)
    valid = dispatched_indices >= 0
    tok_flat = tok_ids[valid]
    exp_flat = dispatched_indices[valid].cast("int32")
    scr_flat = dispatched_probs[valid].cast("float32")

    # 2. Per-expert token counts and required padding
    exp_counts = paddle.bincount(exp_flat, minlength=n_local_experts).cast("int32")
    pad_counts = (block - exp_counts % block) % block
    max_pad = int(pad_counts.max().item())

    if max_pad == 0:
        return x, tok_flat, exp_flat, scr_flat

    # 3. Build padding entries (vectorised)
    row_ids = paddle.arange(max_pad, dtype="int32").unsqueeze(1).expand([max_pad, n_local_experts])
    exp_ids = paddle.arange(n_local_experts, dtype="int32").unsqueeze(0).expand([max_pad, n_local_experts])
    active = row_ids < pad_counts.unsqueeze(0)

    pad_tok = (T + row_ids[active]).cast("int32")
    pad_exp = exp_ids[active].cast("int32")
    pad_scr = paddle.zeros([pad_tok.shape[0]], dtype="float32")

    # 4. Concatenate real + padding pairs
    token_indices = paddle.concat([tok_flat, pad_tok])
    expert_indices = paddle.concat([exp_flat, pad_exp])
    router_scores = paddle.concat([scr_flat, pad_scr])

    # 5. Append zero rows to x
    x_padded = paddle.concat([x, paddle.zeros([max_pad, x.shape[1]], dtype=x.dtype)], axis=0)

    return x_padded, token_indices, expert_indices, router_scores


# ---------------------------------------------------------------------------
# MoELayer — single-card implementation following PaddleFleet
# ---------------------------------------------------------------------------
class MoELayer(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.using_sonic_moe = config.using_sonic_moe
        self.fp8 = config.fp8
        self.moe_grouped_gemm = config.moe_grouped_gemm
        self.router_aux_loss_coef = config.router_aux_loss_coef
        # Single-card: all experts are local
        self.num_experts_per_device = self.num_experts

        # Gate / Router
        self.gate = Gate(config)

        # Experts
        if self.moe_grouped_gemm:
            self.grouped_gemm_experts = GroupedMLPExpert(self.num_experts, config)
        else:
            raise NotImplementedError("Only moe_grouped_gemm=True is supported in this test")

        # Shared experts (optional)
        self.shared_experts = None

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
        Returns:
            output: same shape as input
        """
        orig_shape = hidden_states.shape
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])

        # --- Gate must run under no_grad ---
        # Reason: if gate outputs (indices, weights) carry stop_gradient=False,
        # they propagate into _DownProjection PyLayer as grad-enabled inputs.
        # But _DownProjection.backward returns non-None at positions where
        # Paddle expects None (for stop_gradient=True metadata tensors),
        # causing: "backward should return None at position N because forward
        # Tensor's stopgradient is true" → ValueError / core dump.
        # This matches the router-grad design: router grad flows via aux_loss,
        # not through expert output.
        with paddle.no_grad():
            (
                capacity,
                topk_weights,
                topk_indices,
                gates_masked,
                mask,
                priorities,
                aux_loss,
                z_loss,
            ) = self.gate(hidden_states)

        # --- Expert computation (single-card) ---
        if self.moe_grouped_gemm:
            output = self._forward_single_card_grouped_gemm_moe(
                hidden_states, mask, gates_masked
            )
        else:
            raise NotImplementedError

        # --- Aux loss ---
        if self.training and self.router_aux_loss_coef:
            aux_loss = aux_loss * float(self.router_aux_loss_coef)
            output = AddAuxiliaryLoss.apply(output, aux_loss)

        output = output.reshape(orig_shape)
        return output

    def _forward_single_card_grouped_gemm_moe(
        self,
        hidden_states: paddle.Tensor,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
    ) -> paddle.Tensor:

        def _convert_routing_map_and_probs(routing_map, probs, topk):
            routing_map = routing_map.astype("bool")
            masked_probs = probs * routing_map.astype("float32")
            weights, indices = paddle.topk(masked_probs, k=topk, axis=-1)
            return indices, weights

        if self.using_sonic_moe:
            T = hidden_states.shape[0]
            E = self.num_experts_per_device
            path = self.config.expert_path

            selected_indices, topk_scores = _convert_routing_map_and_probs(
                routing_map, probs, self.num_experts_per_tok
            )

            if path == "mlpnode_v2":
                # ═══ Path 3: SonicMoEMlpNode (full ERNIE drop-in) ═══════
                if not hasattr(self, "_mlpnode"):
                    H = self.config.hidden_size
                    I = self.config.moe_intermediate_size
                    w1 = self.grouped_gemm_experts.weight1
                    w2 = self.grouped_gemm_experts.weight2
                    self._mock_experts = _make_mock_experts(w1, w2, E, H, I)
                    self._mlpnode = SonicMoEMlpNode(
                        experts=self._mock_experts,
                        n_experts=E,
                        hidden_size=H,
                        intermediate_size=I,
                    )
                # Compute tokens_per_expert
                valid = selected_indices >= 0
                tpe = paddle.bincount(
                    selected_indices[valid].cast("int64"), minlength=E
                ).tolist()
                out = self._mlpnode(
                    hidden_states, tpe,
                    dispatched_indices=selected_indices.cast("int32"),
                    dispatched_probs=topk_scores.cast("float32"),
                )
                return out

            elif path == "deepep_metadata":
                # ═══ Path 2: CUDA metadata kernel + direct Up/Down ══════
                valid = selected_indices >= 0
                tpe = paddle.bincount(
                    selected_indices[valid].cast("int64"), minlength=E
                ).tolist()
                (efo, x_gather, s_scatter, s_reverse, naept,
                 scores, TK_padded, _, _) = deepep_topk_to_sonic_metadata(
                    selected_indices.cast("int32"),
                    topk_scores.cast("float32"),
                    tpe, E, device="cuda",
                )
                w1 = self.grouped_gemm_experts.weight1
                w2 = self.grouped_gemm_experts.weight2

                class _Ctx:
                    def save_for_backward(self, *a): self._saved = a
                    def saved_tensor(self): return self._saved
                    def mark_non_differentiable(self, *a): pass
                    def set_materialize_grads(self, v): pass

                up_ctx = _Ctx()
                y1, z = _UpProjection.forward(
                    up_ctx, hidden_states, w1, None,
                    efo, TK_padded, None, 0,
                    x_gather, s_scatter, s_reverse, naept,
                    True, ActivationType.SWIGLU, False, False,
                )
                down_ctx = _Ctx()
                out = _DownProjection.forward(
                    down_ctx, y1, z, w2, None,
                    scores, s_scatter, efo,
                    T, None, 0,
                    x_gather, s_scatter, s_reverse, naept,
                    True, ActivationType.SWIGLU, None,
                )
                return out

            else:
                # ═══ Path 1: Original (argsort via moe_general_routing_inputs)
                x_padded, token_indices, expert_indices, router_scores = \
                    _prepare_sonic_inputs(
                        hidden_states, selected_indices, topk_scores, E
                    )
                w1 = self.grouped_gemm_experts.weight1
                w2 = self.grouped_gemm_experts.weight2
                out, expert_freq = moe_general_routing_inputs(
                    x_padded, router_scores, token_indices, expert_indices,
                    w1, None, w2, None, E, 0, ActivationType.SWIGLU,
                )
                return out[:T]
        else:
            # Standard grouped GEMM path (non-SonicMoE)
            tokens_per_expert = routing_map.sum(axis=0)
            permuted_hidden_states, sorted_indices = permute(
                hidden_states, routing_map, tokens_per_expert
            )
            grouped_expert_out = self.grouped_gemm_experts(
                permuted_hidden_states, tokens_per_expert
            )
            final_hidden_states = unpermute(
                grouped_expert_out,
                sorted_indices,
                restore_shape=hidden_states.shape,
                probs=probs,
                routing_map=routing_map,
            )
            return final_hidden_states.cast(hidden_states.dtype)


# ---------------------------------------------------------------------------
# AddAuxiliaryLoss (from PaddleFleet moe_utils)
# ---------------------------------------------------------------------------
class AddAuxiliaryLoss(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, loss):
        ctx.save_for_backward(loss)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (loss,) = ctx.saved_tensor()
        return grad_output, paddle.ones_like(loss)


# ---------------------------------------------------------------------------
# README baseline data (Session 53, nsys GPU-projection, Target GPU)
# Key: (T, H, I, E, K) → {"bf16_us": ..., "fp8_us": ...}
# ---------------------------------------------------------------------------
README_BASELINES = {
    (8192, 3072, 1536, 8, 8):   {"bf16_us": 3644,  "fp8_us": 2715},
    (16384, 3072, 1536, 8, 8):  {"bf16_us": 7953,  "fp8_us": 5227},
    (32768, 3072, 1536, 8, 8):  {"bf16_us": 16287, "fp8_us": 10652},
    (8192, 3072, 1536, 128, 8): {"bf16_us": 5009,  "fp8_us": 3897},
}


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _run_one_path(config, label, x, out_grad):
    """Run fwd+bwd for one path. Returns output tensor."""
    moe = MoELayer(config)
    paddle.amp.decorate(moe, level="O2", dtype="bfloat16",
                        master_weight=True, master_grad=True)
    clear_all_fp8_weight_caches()
    # Warmup (cold cache)
    with enable_fp8(config.fp8):
        _refresh_fp8_config()
        moe(x)
    # Measured pass
    with enable_fp8(config.fp8):
        out = moe(x)
    out.backward(out_grad)
    clear_all_fp8_weight_caches()
    if hasattr(moe, '_mlpnode'):
        flush_native_grads()
    print(f"  [{label}] out.shape={out.shape}  PASS")
    return out, moe


def _bench_path(config, x, out_grad, n_warmup=5, n_bench=12):
    """CUDA-events benchmark for fwd+bwd. Returns µs/iter."""
    import torch
    moe = MoELayer(config)
    paddle.amp.decorate(moe, level="O2", dtype="bfloat16",
                        master_weight=True, master_grad=True)
    clear_all_fp8_weight_caches()
    with enable_fp8(config.fp8):
        _refresh_fp8_config()
        for _ in range(n_warmup):
            o = moe(x); o.backward(out_grad)
    if hasattr(moe, '_mlpnode'):
        flush_native_grads()
    torch.cuda.synchronize()
    gc.collect(); torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with enable_fp8(config.fp8):
        for _ in range(n_bench):
            o = moe(x); o.backward(out_grad)
    end.record()
    torch.cuda.synchronize()
    if hasattr(moe, '_mlpnode'):
        flush_native_grads()
    clear_all_fp8_weight_caches()
    return start.elapsed_time(end) / n_bench * 1000  # µs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def test_moe_layer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true", help="Run performance benchmark")
    parser.add_argument("--T", type=int, default=None, help="Override seq_len*batch")
    parser.add_argument("--H", type=int, default=None, help="Override hidden_size")
    parser.add_argument("--I", type=int, default=None, help="Override intermediate_size")
    parser.add_argument("--E", type=int, default=None, help="Override n_routed_experts")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bench", type=int, default=12)
    args = parser.parse_args()

    config = MoETestConfig()
    if args.H: config.hidden_size = args.H
    if args.I: config.moe_intermediate_size = args.I
    if args.E: config.n_routed_experts = args.E; config.num_experts_per_tok = min(args.E, 8)

    batch_size = 2
    T = args.T or batch_size * config.seq_len

    print(f"Config: T={T} H={config.hidden_size} I={config.moe_intermediate_size} "
          f"E={config.n_routed_experts} K={config.num_experts_per_tok} fp8={config.fp8}")

    paddle.seed(42)
    x = paddle.randn([T, config.hidden_size], dtype="bfloat16")
    x.stop_gradient = False
    out_grad = paddle.randn_like(x) * 0.01

    if args.perf:
        # ── Performance benchmark ─────────────────────────────────────
        print("\n=== Performance Benchmark (CUDA events, µs/iter fwd+bwd) ===")
        results = {}
        for path in ["original", "deepep_metadata", "mlpnode_v2"]:
            config.expert_path = path
            us = _bench_path(config, x, out_grad, args.warmup, args.bench)
            results[path] = us
            print(f"  {path:20s}: {us:8.1f} µs/iter")

        # Compare against README baseline
        key = (T, config.hidden_size, config.moe_intermediate_size,
               config.n_routed_experts, config.num_experts_per_tok)
        if key in README_BASELINES:
            bl = README_BASELINES[key]
            print(f"\n  README baseline (nsys GPU-proj):")
            print(f"    BF16: {bl['bf16_us']} µs   FP8: {bl['fp8_us']} µs")
            for path, us in results.items():
                ratio = bl["fp8_us"] / us if us > 0 else 0
                print(f"    {path}: {us:.0f} µs  (vs README FP8: {ratio:.2f}x)")
        else:
            print(f"\n  (No README baseline for shape {key})")
        print("\n=== Benchmark done ===")

    else:
        # ── Correctness: run all 3 paths on SAME model, compare outputs ──
        print("\n=== Correctness Test ===")
        config.expert_path = "original"
        moe = MoELayer(config)
        paddle.amp.decorate(moe, level="O2", dtype="bfloat16",
                            master_weight=True, master_grad=True)

        outputs = {}
        for path in ["original", "deepep_metadata", "mlpnode_v2"]:
            moe.config.expert_path = path
            if hasattr(moe, '_mlpnode'):
                delattr(moe, '_mlpnode')  # force re-init for mlpnode_v2
            clear_all_fp8_weight_caches()
            # Warmup
            with enable_fp8(config.fp8):
                _refresh_fp8_config()
                moe(x)
            # Measured
            with enable_fp8(config.fp8):
                out = moe(x)
            out.backward(out_grad)
            if hasattr(moe, '_mlpnode'):
                flush_native_grads()
            clear_all_fp8_weight_caches()
            outputs[path] = out.detach()
            print(f"  [{path}] out.shape={out.shape}  PASS")

        # Pairwise comparison
        print("\n  Pairwise output comparison:")
        ref = outputs["original"]
        all_ok = True
        for path in ["deepep_metadata", "mlpnode_v2"]:
            diff = (ref - outputs[path]).abs()
            rrmse = float(diff.norm().item()) / (float(ref.norm().item()) + 1e-10)
            cos = float(paddle.nn.functional.cosine_similarity(
                ref.flatten().unsqueeze(0).cast("float32"),
                outputs[path].flatten().unsqueeze(0).cast("float32"),
            ).item())
            ok = cos > 0.999 and rrmse < 0.01
            all_ok &= ok
            print(f"    original vs {path:20s}: cos={cos:.8f} rrmse={rrmse:.6f} {'PASS' if ok else 'FAIL'}")

        print(f"\n==== {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'} ====")
        assert all_ok


if __name__ == "__main__":
    test_moe_layer()
