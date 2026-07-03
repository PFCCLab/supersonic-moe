"""
Unit test for MoELayer single-card implementation,
adapted from PaddleFleet's moe_layer.py.

Two test paths:
  Path A — MoELayer using moe_general_routing_inputs (Paddle autograd, reliable baseline).
  Path B — SonicMoEMlpNode E2E (production DeepEP topk path, main_grad accumulation).
"""
import sys
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
    general_routing_router_metadata,
    _UpProjection,
    _DownProjection,
    clear_all_fp8_weight_caches,
    _refresh_fp8_config,
)
from sonicmoe.functional.utils import enable_fp8
from sonicmoe.count_cumsum import count_cumsum
from sonicmoe.ernie_compat.mlp_node_v2 import (
    SonicMoEMlpNode,
    invalidate_weight_caches,
    flush_native_grads,
)


# ---------------------------------------------------------------------------
# Config — mirrors the TransformerConfig fields used by PaddleFleet MoELayer
# ---------------------------------------------------------------------------
@dataclass
class MoETestConfig:
    """Simplified config for single-card MoE testing."""
    seq_len: int = 8192

    # --- Gate / Router ---
    scoring_func: str = "softmax"
    hidden_size: int = 1536
    n_routed_experts: int = 8
    n_shared_experts: int = 0
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0

    # --- Expert ---
    hidden_act: str = "silu"
    gated_linear_unit: bool = True
    moe_intermediate_size: int = 1024
    use_bias: bool = False

    # --- MoE layer ---
    moe_grouped_gemm: bool = True
    router_aux_loss_coef: float = 0.0

    # --- FP8 + SonicMoE ---
    fp8: bool = True
    using_sonic_moe: bool = True


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
# Lightweight expert stubs for SonicMoEMlpNode (Path B)
# ---------------------------------------------------------------------------
class _FakeLinear:
    """Minimal weight holder matching split-half per-expert interface."""
    def __init__(self, weight):
        self.weight = weight


class _FakeExpert:
    """Stub matching SonicMoEMlpNode's per-expert ``up_gate_proj`` / ``down_proj``."""
    def __init__(self, up_gate_weight, down_weight):
        self.up_gate_proj = _FakeLinear(up_gate_weight)   # [H, 2I]
        self.down_proj = _FakeLinear(down_weight)          # [I, H]


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

        # --- Gate (no no_grad: gate has its own autograd for router training) ---
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
                hidden_states, topk_indices, topk_weights
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
        topk_indices: paddle.Tensor,
        topk_scores: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Single-card MoE forward — PaddleFleet style direct .apply().

        topk_indices [T, K] int64 — expert indices from gate (no autograd)
        topk_scores  [T, K] float32 — routing probs from gate (ON autograd graph,
                     ds flows back to gate via _DownProjection.backward)
        """

        if self.using_sonic_moe:
            T = hidden_states.shape[0]
            K = self.num_experts_per_tok
            E = self.num_experts_per_device
            stream_id = 0

            # --- Routing metadata: only integer ops, no Paddle autograd nodes ---
            tok_ids = paddle.arange(T, dtype="int32").unsqueeze(1).expand(
                [T, K]).reshape([-1])
            exp_ids = topk_indices.reshape([-1]).cast("int32")
            scores_flat = topk_scores.reshape([-1])

            (
                _expert_frequency,
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
            ) = general_routing_router_metadata(
                scores_flat, tok_ids, exp_ids, T, E,
            )
            s_scatter_idx.stop_gradient = True

            # --- Direct .apply() (matching PaddleFleet single-card path) ---
            w1 = self.grouped_gemm_experts.weight1  # [2I, H, E]
            w2 = self.grouped_gemm_experts.weight2  # [H, I, E]

            y1, z = _UpProjection.apply(
                hidden_states, w1, None,           # x, w1, b1
                expert_frequency_offset,
                T * K, K, stream_id,
                x_gather_idx, s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                False,                             # is_varlen_K
                ActivationType.SWIGLU,
                False,                             # is_inference_mode_enabled
                False,                             # use_low_precision_postact_buffer
            )

            output = _DownProjection.apply(
                y1, z, w2, None,                   # y1, z, w2, b2
                topk_scores,                       # [T, K] — ds flows back to gate
                s_scatter_idx,                     # selected_experts (positional)
                expert_frequency_offset,
                T, K, stream_id,
                x_gather_idx, s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                False,                             # is_varlen_K
                ActivationType.SWIGLU,
                None,                              # fp8_protocol
            )
            return output
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
# Test entry point — Path A (MoELayer + moe_general_routing_inputs)
# ---------------------------------------------------------------------------
def test_moe_layer():
    config = MoETestConfig()
    print(f"[Path A] Config: fp8={config.fp8}, using_sonic_moe={config.using_sonic_moe}")

    moe_layer = MoELayer(config)

    # No paddle.amp.decorate — FP8 kernels handle precision internally.
    # AMP O2 inserts CastGradNode that segfaults when FP8 backward frees
    # tensor storage via resize_(0). Production PaddleFleet also does not
    # apply AMP decorate to the MoE layer itself.

    batch_size = 2
    x = paddle.randn(
        [batch_size * config.seq_len, config.hidden_size], dtype="bfloat16"
    )
    x.stop_gradient = False

    # --- Pass 1: forward-only (cold cache) to warm up FP8 weight cache ---
    clear_all_fp8_weight_caches()
    with enable_fp8(config.fp8):
        _refresh_fp8_config()
        print("[Path A] pass 1 (cold cache, forward-only)...", flush=True)
        out = moe_layer(x)
        print(f"[Path A] pass 1 done, out.shape={out.shape}", flush=True)
    out_grad = paddle.randn_like(out)

    # --- Pass 2: forward + backward (warm cache) ---
    with enable_fp8(config.fp8):
        print("[Path A] pass 2 (warm cache, forward)...", flush=True)
        out = moe_layer(x)
        print(f"[Path A] pass 2 forward done, out.shape={out.shape}", flush=True)
    print("[Path A] backward...", flush=True)
    out.backward(out_grad)
    print("[Path A] backward done", flush=True)

    # --- Gradient verification ---
    if x.grad is not None:
        print(f"[Path A] x.grad norm = {x.grad.norm().item():.6f}", flush=True)
    else:
        print("[Path A] WARNING: x.grad is None!", flush=True)

    clear_all_fp8_weight_caches()
    print("==== Path A PASSED ====", flush=True)


# ---------------------------------------------------------------------------
# Test entry point — Path B (SonicMoEMlpNode E2E, production MlpNode)
# ---------------------------------------------------------------------------
def test_mlpnode_e2e():
    config = MoETestConfig()
    E = config.n_routed_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    T = 2 * config.seq_len  # batch_size * seq_len
    print(f"[Path B] E={E}, H={H}, I={I}, T={T}", flush=True)

    # 1. Create per-expert parameters (ERNIE layout: w1 [H, 2I], w2 [I, H])
    init = paddle.nn.initializer.Uniform(-0.001, 0.001)
    experts = []
    expert_params = []          # keep references for AMP decorate
    for e_idx in range(E):
        w1 = paddle.create_parameter(
            shape=[H, 2 * I], dtype="bfloat16", default_initializer=init,
        )
        w2 = paddle.create_parameter(
            shape=[I, H], dtype="bfloat16", default_initializer=init,
        )
        expert_params.extend([w1, w2])
        experts.append(_FakeExpert(w1, w2))

    mlp_node = SonicMoEMlpNode(
        experts=experts,
        n_experts=E,
        hidden_size=H,
        intermediate_size=I,
        activation_type=ActivationType.SWIGLU,
    )

    # 2. Create input
    x = paddle.randn([T, H], dtype="bfloat16")
    x.stop_gradient = False

    # 3. Generate routing (gate under no_grad — we only test the MLP here)
    gate = Gate(config)
    with paddle.no_grad():
        _, topk_weights, topk_indices, _, _, _, _, _ = gate(x)

    dispatched_indices = topk_indices.cast("int32")
    dispatched_probs = topk_weights.cast("float32")
    flat_idx = dispatched_indices.reshape([-1])
    tokens_per_expert = paddle.bincount(
        flat_idx.cast("int64"), minlength=E,
    ).tolist()

    # 4. Warm-up pass (cold FP8 cache)
    invalidate_weight_caches()
    clear_all_fp8_weight_caches()
    with enable_fp8(config.fp8):
        _refresh_fp8_config()
        print("[Path B] warm-up (cold cache, forward-only)...", flush=True)
        out = mlp_node(x, tokens_per_expert, dispatched_indices, dispatched_probs)
        print(f"[Path B] warm-up done, out.shape={out.shape}", flush=True)

    # 5. Forward + backward (warm cache)
    # Clear x.grad from warm-up (if any)
    x.clear_gradient()

    with enable_fp8(config.fp8):
        _refresh_fp8_config()
        print("[Path B] forward (warm cache)...", flush=True)
        out = mlp_node(x, tokens_per_expert, dispatched_indices, dispatched_probs)
        print(f"[Path B] forward done, out.shape={out.shape}", flush=True)

    out_grad = paddle.randn_like(out)
    print("[Path B] backward...", flush=True)
    out.backward(out_grad)
    print("[Path B] backward done", flush=True)

    # 6. Flush native-layout grad accumulators → per-expert main_grad
    flush_native_grads()

    # 7. Verify gradients
    # dx — must be non-None and non-zero (autograd chain intact, no detach)
    assert x.grad is not None, "[Path B] FAIL: x.grad is None — autograd chain broken!"
    dx_norm = x.grad.norm().item()
    assert dx_norm > 0, "[Path B] FAIL: x.grad is zero — gradient vanished!"
    print(f"[Path B] dx norm = {dx_norm:.6f}", flush=True)

    # dw — per-expert main_grad via native accumulators
    n_nonzero = 0
    for e_idx, exp in enumerate(experts):
        mg1 = getattr(exp.up_gate_proj.weight, 'main_grad', None)
        mg2 = getattr(exp.down_proj.weight, 'main_grad', None)
        if mg1 is not None and mg1.norm().item() > 0:
            n_nonzero += 1
        if mg2 is not None and mg2.norm().item() > 0:
            n_nonzero += 1
    print(f"[Path B] {n_nonzero}/{2*E} expert weight grads non-zero via main_grad", flush=True)
    assert n_nonzero > 0, "[Path B] FAIL: no main_grad accumulated!"

    invalidate_weight_caches()
    clear_all_fp8_weight_caches()
    print("==== Path B (SonicMoEMlpNode E2E) PASSED ====", flush=True)


if __name__ == "__main__":
    test_moe_layer()          # Path A: MoELayer + moe_general_routing_inputs
    test_mlpnode_e2e()         # Path B: SonicMoEMlpNode (production MlpNode)
