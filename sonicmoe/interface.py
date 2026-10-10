# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import paddle

from .enums import ActivationType
from .ernie_compat.deepep_metadata import (
    deepep_topk_to_sonic_metadata,
    deepep_topk_to_sonic_metadata_with_scales,
)
from .ernie_compat.mlp_node_v2 import (
    _differentiable_router_scores,
)
from .functional import (
    _DownProjection,
    _UpProjection,
    _coerce_activation_type,
    attach_preallocated_gated_outputs,
)
from .functional.utils import enable_fp8
from .functional.fp8_protocol import resolve_fp8_backend

try:
    from .quack_utils.blockscaled_fp8_gemm import (
        _scatter_router_scores_i32,
    )
except (ImportError, RuntimeError):
    _scatter_router_scores_i32 = None

def _log_stage_memory(stage: str) -> None:
    stage_mem = os.getenv("SONIC_MOE_STAGEWISE_MEMORY", "").lower() in {"1", "true", "yes", "on"}
    if not stage_mem:
        return
    paddle.cuda.synchronize()
    mib = 1024**2
    print(
        f"[stage-memory] {stage}: "
        f"alloc_mib={paddle.cuda.memory_allocated() / mib:.2f}, "
        f"reserved_mib={paddle.cuda.memory_reserved() / mib:.2f}, "
        f"peak_alloc_mib={paddle.cuda.max_memory_allocated() / mib:.2f}, "
        f"peak_reserved_mib={paddle.cuda.max_memory_reserved() / mib:.2f}"
    )


def _resolve_sonic_config_bool(config, name):
    if config is None:
        return False
    value = getattr(config, name, None)
    if value is not None:
        return bool(value)
    resolver = getattr(config, f"resolve_{name}", None)
    return bool(resolver()) if resolver is not None else False


def run_sonic_moe(
    hidden_states,
    topk_indices,
    topk_scores,
    K,
    E,
    w1,
    w2,
    fp8=False,
    tokens_per_expert=None,
    fp8_scale=None,
    fp8_combine_grad_handle=None,
    fp8_config=None,
    release_fp8_weights=False,
    activation_type=ActivationType.SWIGLU,
    sync_free_sizing=False,
):
    """Run one SonicMoE expert-compute forward.

    ``activation_type`` accepts an ``ActivationType`` member, a plain enum value
    (``"swiglu"``), or an encoded SiTU string (``"situ_glu:b=4.0:lb=25.0"``).
    Bare ``ActivationType.SITU_GLU`` picks up its betas from the active
    ``SonicMoEConfig`` (``situ_beta`` / ``situ_linear_beta``).  Defaults to
    SWIGLU so existing callers are unaffected.
    """
    T = hidden_states.shape[0]
    stream_id = paddle.device.current_stream()
    topk_indices_i32 = (
        topk_indices
        if topk_indices.dtype == paddle.int32
        else topk_indices.cast(paddle.int32)
    )

    if tokens_per_expert is None and not sync_free_sizing:
        valid = topk_indices >= 0
        valid_experts = topk_indices[valid].cast(paddle.int32)
        tokens_per_expert = paddle.bincount(valid_experts, minlength=E).cast(
            paddle.int32
        )

    fp8_scale_packed = None
    gated_outputs = ()
    backend_resolution = (
        resolve_fp8_backend(fp8_config, hidden_states.device) if fp8 else None
    )
    is_sm90 = backend_resolution is not None and backend_resolution.is_hopper
    if is_sm90:
        (
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            _router_scores,
            TK_padded,
            total_pad_rows,
            _N_recv,
            _score_src_idx,
        ) = deepep_topk_to_sonic_metadata(
            topk_indices_i32,
            topk_scores,
            tokens_per_expert,
            E,
            block=backend_resolution.routing_block_size,
            sync_free_sizing=sync_free_sizing,
        )
    elif (
        fp8
        and fp8_scale is not None
        and deepep_topk_to_sonic_metadata_with_scales is not None
    ):
        gated_n = int(w1.shape[1])
        gated_z_quant = _resolve_sonic_config_bool(
            fp8_config, "epilogue_quant"
        ) and _resolve_sonic_config_bool(fp8_config, "save_z_fp8")
        preallocate_gated_outputs = (
            hidden_states.dtype == paddle.float8_e4m3fn
            and gated_n % 256 == 0
            and _resolve_sonic_config_bool(fp8_config, "fuse_y1_quant")
        )
        if preallocate_gated_outputs:
            metadata_result = deepep_topk_to_sonic_metadata_with_scales(
                topk_indices_i32,
                topk_scores,
                tokens_per_expert,
                E,
                fp8_scale,
                int(hidden_states.shape[1]),
                block=backend_resolution.routing_block_size,
                gated_output_prototype=hidden_states,
                gated_n=gated_n,
                gated_preact_bf16=not gated_z_quant,
                gated_allocate_z_scale=gated_z_quant,
                sync_free_sizing=sync_free_sizing,
            )
        else:
            metadata_result = deepep_topk_to_sonic_metadata_with_scales(
                topk_indices_i32,
                topk_scores,
                tokens_per_expert,
                E,
                fp8_scale,
                int(hidden_states.shape[1]),
                block=backend_resolution.routing_block_size,
                sync_free_sizing=sync_free_sizing,
            )
        (
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            _router_scores,
            TK_padded,
            total_pad_rows,
            _N_recv,
            _score_src_idx,
            fp8_scale_packed,
        ) = metadata_result[:11]
        gated_outputs = tuple(metadata_result[11:])
    else:
        (
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            _router_scores,
            TK_padded,
            total_pad_rows,
            _N_recv,
            _score_src_idx,
        ) = deepep_topk_to_sonic_metadata(
            topk_indices_i32,
            topk_scores,
            tokens_per_expert,
            E,
            block=128 if fp8 else 1,
            sync_free_sizing=sync_free_sizing,
        )

    s_scatter_idx.stop_gradient = True
    # Caller-supplied activation (was hardcoded to "swiglu").  Normalised here so
    # encoded SiTU strings survive: ActivationType("situ_glu:b=4.0:lb=25.0")
    # would raise, since the enum value is the bare "situ_glu".
    activation_type = _coerce_activation_type(activation_type)

    total_expert_freq = TK_padded
    router_score_source = None
    router_score_src_idx = None
    router_scores_need_grad = (
        hasattr(topk_scores, "stop_gradient") and not topk_scores.stop_gradient
    )
    if not router_scores_need_grad:
        # No router edge is needed. Keep metadata scores forward-only and do not
        # pass the original source into _DownProjection: Paddle rejects any
        # tensor returned for a forward input whose stop_gradient is true.
        _router_scores.stop_gradient = True
        scores_for_down = _router_scores
    elif _score_src_idx is not None and _scatter_router_scores_i32 is not None:
        # DownProjection already computes metadata-order ds. Attach the original
        # [T, K] score tensor as the backward edge and scatter ds natively.
        _router_scores.stop_gradient = True
        scores_for_down = _router_scores
        router_score_source = topk_scores
        router_score_src_idx = _score_src_idx
    else:
        # The differentiable gather is the compatibility fallback when metadata
        # does not provide a source map or the native scatter is unavailable.
        scores_for_down = _differentiable_router_scores(
            topk_scores,
            topk_indices.cast(paddle.int32),
            num_activated_expert_per_token_offset,
            TK_padded - total_pad_rows,
            TK_padded,
            E,
            score_src_idx=_score_src_idx,
        )

    fp8_hidden_states = None
    if fp8_scale is not None:
        if fp8_scale_packed is not None:
            if gated_outputs:
                attach_preallocated_gated_outputs(
                    fp8_scale_packed, gated_outputs
                )
            fp8_hidden_states = (hidden_states, fp8_scale, fp8_scale_packed)
        else:
            fp8_hidden_states = (hidden_states, fp8_scale)

    # if fp8:
    #     w1_sonic = _make_sonic_fp8_weight_carrier(w1)
    #     w2_sonic = _make_sonic_fp8_weight_carrier(w2)
    # else:
    #     w1_sonic = w1.permute([1, 2, 0])
    #     w2_sonic = w2.permute([1, 2, 0])

    with enable_fp8(fp8):
        y1, z = _UpProjection.apply(
            hidden_states,
            w1,
            None,
            expert_frequency_offset,
            total_expert_freq,
            K,
            stream_id,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            True,  # is_varlen_k
            activation_type,
            is_inference_mode_enabled=False,
            use_low_precision_postact_buffer=False,
            prequant_activation_payload=fp8_hidden_states,
            fp8_config=fp8_config,
        )
        # SM90 weights are quantized and owned by PaddleFleet's moe_expert;
        # they must not be released per call.
        if release_fp8_weights and not fp8_config.recompute_z and not is_sm90:
            transposed_fp8 = getattr(w1, "transposed_fp8", None)
            if (
                transposed_fp8 is None
                or w1.fp8[0].data_ptr() != transposed_fp8[0].data_ptr()
            ):
                w1.fp8[0]._clear_to_zero_allocation()
            w1.fp8[1]._clear_to_zero_allocation()

        down_args = (
            y1,
            z,
            w2,
            None,
            scores_for_down,
            s_scatter_idx,
            expert_frequency_offset,
            T,
            K,
            stream_id,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            True,  # is_varlen_k
            activation_type,
            None,
            fp8_combine_grad_handle,
        )
        if router_score_source is not None:
            hidden_states = _DownProjection.apply(
                *down_args,
                fp8_config,
                router_score_source,
                router_score_src_idx,
            )
        else:
            hidden_states = _DownProjection.apply(
                *down_args, fp8_config=fp8_config
            )
        if release_fp8_weights and not is_sm90:
            transposed_fp8 = getattr(w2, "transposed_fp8", None)
            if (
                transposed_fp8 is None
                or w2.fp8[0].data_ptr() != transposed_fp8[0].data_ptr()
            ):
                w2.fp8[0]._clear_to_zero_allocation()
            w2.fp8[1]._clear_to_zero_allocation()

    # _log_stage_memory("MoE expert compute forward end")

    return hidden_states
