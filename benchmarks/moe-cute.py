# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import argparse
import contextlib
import io
import math
import os
import random
import re
import time
from typing import Tuple, Type

import cutlass
import torch
import torch.nn.functional as F
from rich import print as print0
from triton.testing import do_bench

from sonicmoe import MoE, get_default_fp8_protocol
from sonicmoe.enums import ActivationType, is_glu
from sonicmoe.functional import moe_TC_softmax_topk_layer
from sonicmoe.functional import clear_fp8_native_weight_cache


def swiglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return u * F.silu(g)


def geglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return F.gelu(g.float()).to(dtype=g.dtype) * u


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float()).to(dtype=x.dtype)


def reglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return (F.relu(g.float()) * u).to(dtype=g.dtype)


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def relu_sq(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of SonicMoE (TC top-K routing).")

    parser.add_argument(
        "--thiek",
        type=parse_comma_separated_ints,
        default=(32768, 4096, 1024, 128, 8),
        help="T, H, I, E, K dimensions (comma-separated)",
    )
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--activation", choices=["swiglu", "geglu", "reglu", "relu_sq", "relu", "silu", "gelu"], default="swiglu"
    )
    parser.add_argument(
        "--add_bias",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--fp8_protocol",
        choices=["none", "blackwell"],
        default="none",
        help="Enable the SM100 FP8 protocol path for the up-proj/down-proj activation boundary",
    )
    parser.add_argument(
        "--report_fp8_metrics",
        action="store_true",
        default=False,
        help="When fp8 is enabled, report bf16-baseline RMSE and peak memory deltas before timing runs",
    )
    parser.add_argument(
        "--report_stage_memory",
        action="store_true",
        default=False,
        help="Print stagewise CUDA memory stats for one real training-mode forward/backward pass",
    )
    parser.add_argument(
        "--report_fp8_analysis",
        action="store_true",
        default=False,
        help="Print theoretical weight/activation memory and FP8 boundary traffic analysis",
    )
    parser.add_argument(
        "--prefetch_fp8_weights",
        action="store_true",
        default=False,
        help="Pre-quantize supported FP8 weights before fp8 timing/metric runs so runtime measures static fp8 weights",
    )
    parser.add_argument(
        "--native_fp8_forward",
        action="store_true",
        default=False,
        help="(Legacy) Enable native FP8 tensor cores. Use --fp8_mode instead.",
    )
    parser.add_argument(
        "--fp8_mode",
        type=str,
        default=None,
        choices=["perf", "mem"],
        help="Full-pipeline FP8 mode: 'perf' (cached weights), 'mem' (no cache, min memory)",
    )
    args = parser.parse_args()

    if len(args.thiek) != 5:
        parser.error("--thiek must contain exactly 5 values")

    return args


def our_e2e_fwd_bwd_call(moe, x, dout):
    o = moe(x)[0]
    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    o.backward(dout, retain_graph=True)
    x.grad = w1.grad = w2.grad = router_w.grad = None


def _mib(numel: int, bytes_per_elem: int) -> float:
    return numel * bytes_per_elem / (1024**2)


def _print_fp8_theory_report(
    *,
    thiek: Tuple[int, int, int, int, int],
    fp8_enabled: bool,
) -> None:
    T, H, I, E, K = thiek
    TK = T * K
    scale_cols_128 = I // 128

    w1_mib = _mib(E * 2 * I * H, 2)
    w2_mib = _mib(E * I * H, 2)
    x_mib = _mib(T * H, 2)
    z_mib = _mib(TK * 2 * I, 2)
    postact_bf16_mib = _mib(TK * I, 2)
    postact_fp8_mib = _mib(TK * I, 1)
    scales_f32_mib = _mib(TK * scale_cols_128, 4)
    scales_e8m0_mib = _mib(TK * scale_cols_128, 1)
    weight_fp8_data_mib = _mib(E * H * (3 * I), 1)
    weight_fp8_scale_e8m0_mib = _mib(E * H * ((2 * I) // 128 + I // 128), 1)
    weight_fp8_total_mib = weight_fp8_data_mib + weight_fp8_scale_e8m0_mib

    stable_fp8_boundary_lower_bound_mib = z_mib + postact_fp8_mib + scales_f32_mib + postact_fp8_mib + scales_f32_mib + postact_bf16_mib
    stable_fp8_saved_payload_mib = postact_bf16_mib - (postact_fp8_mib + scales_f32_mib)
    direct_fp8_boundary_floor_mib = z_mib + postact_fp8_mib + scales_e8m0_mib
    direct_fp8_boundary_saved_mib = stable_fp8_boundary_lower_bound_mib - direct_fp8_boundary_floor_mib
    aggressive_weight_saved_mib = (w1_mib + w2_mib) - weight_fp8_total_mib
    aggressive_total_saved_mib = aggressive_weight_saved_mib + (postact_bf16_mib - (postact_fp8_mib + scales_e8m0_mib))

    print0(
        "[bold yellow]Theoretical memory / compute analysis[/bold yellow] "
        f"weights_mib(w1={w1_mib:.2f}, w2={w2_mib:.2f}, total={w1_mib + w2_mib:.2f}); "
        f"activations_mib(x={x_mib:.2f}, z={z_mib:.2f}, postact_bf16={postact_bf16_mib:.2f}, "
        f"postact_fp8={postact_fp8_mib:.2f}, scales_f32={scales_f32_mib:.2f}, scales_e8m0={scales_e8m0_mib:.2f}); "
        f"stable_fp8_boundary_lower_bound_mib={stable_fp8_boundary_lower_bound_mib:.2f}; "
        f"stable_fp8_saved_payload_mib={stable_fp8_saved_payload_mib:.2f}; "
        f"direct_fp8_boundary_floor_mib={direct_fp8_boundary_floor_mib:.2f}; "
        f"direct_fp8_boundary_saved_mib={direct_fp8_boundary_saved_mib:.2f}; "
        f"aggressive_weight_fp8_storage_mib={weight_fp8_total_mib:.2f}; "
        f"aggressive_weight_saved_mib={aggressive_weight_saved_mib:.2f}; "
        f"aggressive_total_saved_mib={aggressive_total_saved_mib:.2f}"
    )

    blockscaled_capacity = os.getenv("SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY")
    if fp8_enabled and os.getenv("SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ", "").lower() in {"1", "true", "yes", "on"}:
        if blockscaled_capacity is not None:
            capacity = int(blockscaled_capacity)
            grouped_out_mib = _mib(E * capacity * H, 2)
            grouped_a_fp8_mib = _mib(E * capacity * I, 1)
            grouped_scale_1x32_mib = _mib(E * capacity * (I // 32), 1)
            print0(
                "[bold yellow]Blockscaled static-layout analysis[/bold yellow] "
                f"capacity={capacity}; grouped_out_bf16_mib={grouped_out_mib:.2f}; "
                f"grouped_a_fp8_mib={grouped_a_fp8_mib:.2f}; grouped_a_scale_e8m0_mib={grouped_scale_1x32_mib:.2f}"
            )


_STAGE_MEMORY_PATTERN = re.compile(
    r"^\[stage-memory\] (?P<stage>.+?): "
    r"alloc_mib=(?P<alloc>[0-9.]+), "
    r"reserved_mib=(?P<reserved>[0-9.]+), "
    r"peak_alloc_mib=(?P<peak_alloc>[0-9.]+), "
    r"peak_reserved_mib=(?P<peak_reserved>[0-9.]+)$"
)


def _parse_stage_memory_output(output: str) -> dict[str, dict[str, float]]:
    stages: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        match = _STAGE_MEMORY_PATTERN.match(line.strip())
        if match is None:
            continue
        stages[match.group("stage")] = {
            "alloc_mib": float(match.group("alloc")),
            "reserved_mib": float(match.group("reserved")),
            "peak_alloc_mib": float(match.group("peak_alloc")),
            "peak_reserved_mib": float(match.group("peak_reserved")),
        }
    return stages


def _run_stage_memory_case(
    *,
    moe,
    x: torch.Tensor,
    dout: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    activation: ActivationType,
    protocol_config,
) -> tuple[dict[str, dict[str, float]], float]:
    previous_probe = os.environ.get("SONIC_MOE_STAGEWISE_MEMORY")
    os.environ["SONIC_MOE_STAGEWISE_MEMORY"] = "1"
    capture = io.StringIO()
    try:
        x_case = x.detach().clone().requires_grad_()
        dout_case = dout.detach().clone()
        for grad_tensor in [x, w1, w2, router_w]:
            grad_tensor.grad = None
        clear_fp8_native_weight_cache()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with contextlib.redirect_stdout(capture):
            o_case, _, _ = moe_TC_softmax_topk_layer(
                x_case,
                router_w,
                w1.permute(1, 2, 0),
                b1,
                w2.permute(1, 2, 0),
                b2,
                moe.top_k,
                moe.stream_id,
                activation,
                False,
                protocol_config,
            )
            o_case.backward(dout_case)
            torch.cuda.synchronize()
        final_peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
        for grad_tensor in [x_case, x, w1, w2, router_w]:
            grad_tensor.grad = None
    finally:
        if previous_probe is None:
            os.environ.pop("SONIC_MOE_STAGEWISE_MEMORY", None)
        else:
            os.environ["SONIC_MOE_STAGEWISE_MEMORY"] = previous_probe
    return _parse_stage_memory_output(capture.getvalue()), final_peak_mib


def _print_stage_memory_comparison(
    *,
    bf16_stages: dict[str, dict[str, float]],
    bf16_final_peak_mib: float,
    fp8_stages: dict[str, dict[str, float]],
    fp8_final_peak_mib: float,
) -> None:
    ordered_stages = list(dict.fromkeys([*bf16_stages.keys(), *fp8_stages.keys()]))
    summary_parts = []
    for stage in ordered_stages:
        bf16_stage = bf16_stages.get(stage)
        fp8_stage = fp8_stages.get(stage)
        if bf16_stage is None or fp8_stage is None:
            continue
        delta_peak_alloc = fp8_stage["peak_alloc_mib"] - bf16_stage["peak_alloc_mib"]
        delta_alloc = fp8_stage["alloc_mib"] - bf16_stage["alloc_mib"]
        summary_parts.append(
            f"{stage}(bf16_peak_alloc={bf16_stage['peak_alloc_mib']:.2f}, "
            f"fp8_peak_alloc={fp8_stage['peak_alloc_mib']:.2f}, "
            f"delta_peak_alloc={delta_peak_alloc:+.2f}, "
            f"delta_alloc={delta_alloc:+.2f})"
        )
    print0(
        "[bold magenta]Stagewise memory comparison (bf16 vs fp8)[/bold magenta] "
        + "; ".join(summary_parts)
        + f"; final_peak_mib(bf16={bf16_final_peak_mib:.2f}, fp8={fp8_final_peak_mib:.2f}, "
        f"delta={fp8_final_peak_mib - bf16_final_peak_mib:+.2f})"
    )


def run(
    thiek: Tuple[int, int, int, int, int],
    dtype: Type[cutlass.Numeric],
    skip_test: Type[bool],
    add_bias: Type[bool],
    activation: Type[str],
    fp8_protocol: Type[str],
    report_fp8_metrics: Type[bool],
    report_stage_memory: Type[bool],
    report_fp8_analysis: Type[bool],
    prefetch_fp8_weights: Type[bool],
    native_fp8_forward: bool = False,
    fp8_mode: str | None = None,
    **kwargs,
):
    torch_dtype = {cutlass.BFloat16: torch.bfloat16, cutlass.Float16: torch.float16}[dtype]

    activation = ActivationType(activation)
    fp8_protocol_config = get_default_fp8_protocol() if fp8_protocol == "blackwell" else None
    # Unpack parameters
    T, H, I, E, K = thiek
    TK = T * K
    print(f"T {T}, I {I}, H {H}, E {E}, K {K}")
    if report_fp8_analysis:
        _print_fp8_theory_report(thiek=thiek, fp8_enabled=fp8_protocol_config is not None)

    random.seed(1111)
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    # Create and permute tensor A/B/C

    moe = (
        MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            activation_function=activation,
            add_bias=add_bias,
            std=0.02,
        )
        .to(dtype=torch_dtype)
        .cuda()
    )

    x = 0.2 * torch.randn(T, H, device="cuda:0", dtype=torch_dtype, requires_grad=True)
    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    b1, b2 = moe.c_fc.bias, moe.c_proj.bias
    if add_bias:
        torch.nn.init.normal_(b1, 0, 0.01)
        torch.nn.init.normal_(b2, 0, 0.01)
    dout = 0.2 * torch.randn_like(x, requires_grad=True)

    # ── FP8 mode setup ──
    if fp8_mode is not None:
        os.environ["SONIC_MOE_FP8_MODE"] = fp8_mode
        print0(
            f"[bold cyan]Full-pipeline FP8 mode='{fp8_mode}' enabled[/bold cyan]"
        )
    elif native_fp8_forward:
        os.environ["SONIC_MOE_OPT_NATIVE_FP8_UPPROJ"] = "1"
        os.environ["SONIC_MOE_OPT_MIXED_DTYPE_DOWNPROJ_DW2"] = "1"
        print0(
            "[bold cyan]Native FP8 forward/backward enabled (legacy flags)[/bold cyan]"
        )

    static_fp8_prefetch_announced = False

    def prime_static_fp8_weights() -> None:
        nonlocal static_fp8_prefetch_announced
        prefetched = moe.prefetch_fp8_weights(fp8_protocol_config)
        torch.cuda.synchronize()
        if not static_fp8_prefetch_announced:
            downproj_weight_fp8, downproj_scales = prefetched["downproj"]
            print0(
                "[bold yellow]Static FP8 weight prefetch[/bold yellow] "
                f"downproj_fp8_shape={tuple(downproj_weight_fp8.shape)}, "
                f"downproj_scale_shape={tuple(downproj_scales.shape)}, "
                f"downproj_weight_dtype={downproj_weight_fp8.dtype}, "
                f"downproj_scale_dtype={downproj_scales.dtype}"
            )
            static_fp8_prefetch_announced = True

    static_fp8_weights_enabled = prefetch_fp8_weights

    if report_fp8_metrics and fp8_protocol_config is not None:
        def collect_grad_rmses(fp8_grads, bf16_grads):
            return {
                name: torch.sqrt(torch.mean((fp8_grads[name] - bf16_grads[name]) ** 2)).item() for name in bf16_grads
            }

        def run_metrics_case(protocol_config):
            moe.clear_fp8_weight_cache()
            clear_fp8_native_weight_cache()
            if protocol_config is not None and static_fp8_weights_enabled:
                prime_static_fp8_weights()
            x_case = x.detach().clone().requires_grad_()
            dout_case = dout.detach().clone()
            tracked_grad_tensors = {
                "dw1": w1,
                "dw2": w2,
                "drouter_w": router_w,
            }
            if add_bias:
                tracked_grad_tensors["db1"] = b1
                tracked_grad_tensors["db2"] = b2

            for grad_tensor in tracked_grad_tensors.values():
                grad_tensor.grad = None

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            o_case, _, _ = moe_TC_softmax_topk_layer(
                x_case,
                router_w,
                w1.permute(1, 2, 0),
                b1,
                w2.permute(1, 2, 0),
                b2,
                moe.top_k,
                moe.stream_id,
                activation,
                False,
                protocol_config,
            )
            o_case.backward(dout_case)
            torch.cuda.synchronize()
            peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
            loss_value = (o_case.float() * dout_case.float()).mean().item()
            output_value = o_case.detach().float().cpu()
            grad_values = {"dx": x_case.grad.detach().float().cpu()}
            grad_values.update(
                {name: grad_tensor.grad.detach().float().cpu() for name, grad_tensor in tracked_grad_tensors.items()}
            )

            for grad_tensor in [x_case, *tracked_grad_tensors.values()]:
                grad_tensor.grad = None

            return output_value, loss_value, peak_mib, grad_values

        bf16_output, bf16_loss, bf16_peak_mib, bf16_grads = run_metrics_case(None)
        fp8_output, fp8_loss, fp8_peak_mib, fp8_grads = run_metrics_case(fp8_protocol_config)

        output_rmse = torch.sqrt(torch.mean((fp8_output - bf16_output) ** 2)).item()
        loss_rmse = math.sqrt((fp8_loss - bf16_loss) ** 2)
        grad_rmses = collect_grad_rmses(fp8_grads, bf16_grads)
        grad_rmse_summary = ", ".join(f"{name}_rmse={rmse:.8f}" for name, rmse in grad_rmses.items())

        print0(
            "[bold cyan]FP8 metrics vs bf16 baseline[/bold cyan] "
            f"output_rmse={output_rmse:.8f}, loss_rmse={loss_rmse:.8f}, "
            f"bf16_peak_mib={bf16_peak_mib:.2f}, fp8_peak_mib={fp8_peak_mib:.2f}, "
            + grad_rmse_summary
        )

    if report_stage_memory:
        if fp8_protocol_config is None:
            moe.clear_fp8_weight_cache()
            clear_fp8_native_weight_cache()
            stages, final_peak_mib = _run_stage_memory_case(
                moe=moe,
                x=x,
                dout=dout,
                router_w=router_w,
                w1=w1,
                b1=b1,
                w2=w2,
                b2=b2,
                activation=activation,
                protocol_config=None,
            )
            print0(
                "[bold magenta]Stagewise memory probe complete[/bold magenta] "
                + "; ".join(
                    f"{stage}(peak_alloc={stats['peak_alloc_mib']:.2f}, alloc={stats['alloc_mib']:.2f})"
                    for stage, stats in stages.items()
                )
                + f"; final_peak_mib={final_peak_mib:.2f}"
            )
        else:
            moe.clear_fp8_weight_cache()
            clear_fp8_native_weight_cache()
            bf16_stages, bf16_final_peak_mib = _run_stage_memory_case(
                moe=moe,
                x=x,
                dout=dout,
                router_w=router_w,
                w1=w1,
                b1=b1,
                w2=w2,
                b2=b2,
                activation=activation,
                protocol_config=None,
            )
            moe.clear_fp8_weight_cache()
            clear_fp8_native_weight_cache()
            if static_fp8_weights_enabled:
                prime_static_fp8_weights()
            fp8_stages, fp8_final_peak_mib = _run_stage_memory_case(
                moe=moe,
                x=x,
                dout=dout,
                router_w=router_w,
                w1=w1,
                b1=b1,
                w2=w2,
                b2=b2,
                activation=activation,
                protocol_config=fp8_protocol_config,
            )
            _print_stage_memory_comparison(
                bf16_stages=bf16_stages,
                bf16_final_peak_mib=bf16_final_peak_mib,
                fp8_stages=fp8_stages,
                fp8_final_peak_mib=fp8_final_peak_mib,
            )

    # # Ref check
    if not skip_test:
        o, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
            x, router_w, w1.permute(1, 2, 0), b1, w2.permute(1, 2, 0), b2, moe.top_k, moe.stream_id, activation
        )
        if add_bias:
            dx, dw1, db1, dw2, db2, drouter_w = torch.autograd.grad(
                o, [x, w1, b1, w2, b2, router_w], grad_outputs=dout
            )
        else:
            dx, dw1, dw2, drouter_w = torch.autograd.grad(o, [x, w1, w2, router_w], grad_outputs=dout)

        logits = F.linear(x, router_w)
        ref_topk_logits, ref_topk_experts = logits.topk(K, dim=-1)
        ref_topk_scores = ref_topk_logits.softmax(dim=-1, dtype=torch.float32)

        ref_topk_expert_idx, ref_s_scatter_idx = ref_topk_experts.flatten().sort()
        ref_topk_expert_idx, ref_s_scatter_idx = ref_topk_expert_idx.int(), ref_s_scatter_idx.int()

        ref_expert_frequency = ref_topk_expert_idx.view(-1).bincount(minlength=E).int()
        torch.testing.assert_close(expert_frequency.int(), ref_expert_frequency.int())

        act_func = {
            ActivationType.SWIGLU: swiglu,
            ActivationType.GEGLU: geglu,
            ActivationType.REGLU: reglu,
            ActivationType.GELU: gelu,
            ActivationType.RELU: relu,
            ActivationType.SILU: silu,
            ActivationType.RELU_SQ: relu_sq,
        }[activation]

        with torch.autocast("cuda:0", torch.float32):
            ref_o = torch.zeros_like(x)

            for i in range(E):
                T_idx, E_idx = torch.argwhere(ref_topk_experts == i).split(1, dim=1)
                T_idx, E_idx = T_idx.squeeze(-1), E_idx.squeeze(-1)

                if T_idx.numel() > 0:
                    w1_out = F.linear(x[T_idx, :], w1[i, :, :].squeeze(), bias=(b1[i] if add_bias else None))
                    w1_out = act_func(w1_out)

                    w2_out = F.linear(w1_out, w2[i, :, :].squeeze(), bias=(b2[i] if add_bias else None))

                    ref_o[T_idx, :] += w2_out * ref_topk_scores[T_idx, E_idx, None]

            o_diff = (o.float() - ref_o).abs()

            print(f"max ref o val {ref_o.abs().max():.6f}")
            print(f"mean ref o val {ref_o.abs().mean():.6f}")
            print(f"max abs diff on o {o_diff.max():.6f}")
            print(f"mean rel diff on o {(o_diff / (ref_o.abs() + 1e-6)).mean():.6f}" + "\n")

            if add_bias:
                ref_dx, ref_dw1, ref_db1, ref_dw2, ref_db2, ref_drouter_w = torch.autograd.grad(
                    ref_o, [x, w1, b1, w2, b2, router_w], grad_outputs=dout
                )
                test_triple_list = [
                    ("dx", dx, ref_dx),
                    ("dw2", dw2, ref_dw2),
                    ("db2", db2, ref_db2),
                    ("dw1", dw1, ref_dw1),
                    ("db1", db1, ref_db1),
                    ("drouter_w", drouter_w, ref_drouter_w),
                ]
            else:
                ref_dx, ref_dw1, ref_dw2, ref_drouter_w = torch.autograd.grad(
                    ref_o, [x, w1, w2, router_w], grad_outputs=dout
                )
                test_triple_list = [
                    ("dx", dx, ref_dx),
                    ("dw2", dw2, ref_dw2),
                    ("dw1", dw1, ref_dw1),
                    ("drouter_w", drouter_w, ref_drouter_w),
                ]
            for n, our, ref in test_triple_list:
                print(f"max abs ref value {n} {ref.abs().max():.6f}")
                print(f"mean abs ref value {n} {ref.abs().mean():.6f}")
                print(f"max abs diff on {n} {(our - ref).abs().max():.6f}")
                print(f"mean rel diff on {n} {((our - ref).abs() / (ref.abs() + 1e-6)).mean():.6f}" + "\n")

    if is_glu(activation):
        flops = 6 * T * I * H * K
    else:
        flops = 4 * T * I * H * K

    repeats = 500
    warmup = 5

    time.sleep(0.5)

    moe.clear_fp8_weight_cache()
    if static_fp8_weights_enabled:
        prime_static_fp8_weights()

    # Warmup — populate all CuTe compile caches and Triton autotune
    moe_TC_softmax_topk_layer(
        x,
        router_w,
        w1.permute(1, 2, 0),
        b1,
        w2.permute(1, 2, 0),
        b2,
        moe.top_k,
        moe.stream_id,
        activation,
        True,
        fp8_protocol_config,
    )

    cuda_graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    # Redirect CuTe kernels to capture stream
    old_stream_id = moe.stream_id
    moe.stream_id = stream.cuda_stream

    # ── Inference mode, Forward only (with cudagraphs) ──
    with torch.cuda.stream(stream):
        with torch.cuda.graph(cuda_graph, stream=stream):
            o, _, _ = moe_TC_softmax_topk_layer(
                x,
                router_w,
                w1.permute(1, 2, 0),
                b1,
                w2.permute(1, 2, 0),
                b2,
                moe.top_k,
                moe.stream_id,
                activation,
                True,
                fp8_protocol_config,
            )

    moe.stream_id = old_stream_id  # restore

    fwd_timing = do_bench(lambda: cuda_graph.replay(), warmup=warmup, rep=repeats)
    tflops = flops / (fwd_timing * 1e9)
    print0(f" Cute-DSL Fwd (inference mode + cudagraph) Average time: {fwd_timing:.3f} ms, TFLOPS: {tflops:.1f}")

    # ── Training mode, Forward only ──
    def forward_only_training_mode():
        o, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
            x,
            router_w,
            w1.permute(1, 2, 0),
            b1,
            w2.permute(1, 2, 0),
            b2,
            moe.top_k,
            moe.stream_id,
            activation,
            False,
            fp8_protocol_config,
        )
        return o

    fwd_no_cg_timing = do_bench(forward_only_training_mode, warmup=warmup, rep=repeats)
    print0(f" Cute-DSL Fwd (training mode) Average time: {fwd_no_cg_timing:.3f} ms")

    if is_glu(activation):
        flops = 18 * T * I * H * K
    else:
        flops = 12 * T * I * H * K

    time.sleep(0.5)

    def forward_and_backward():
        o, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
            x,
            router_w,
            w1.permute(1, 2, 0),
            b1,
            w2.permute(1, 2, 0),
            b2,
            moe.top_k,
            moe.stream_id,
            activation,
            False,
            fp8_protocol_config,
        )
        o.backward(dout, retain_graph=True)
        x.grad = w1.grad = w2.grad = router_w.grad = None

    e2e_timing = do_bench(forward_and_backward, warmup=warmup, rep=repeats, grad_to_none=[x, w1, w2, router_w, dout])
    tflops = flops / (e2e_timing * 1e9)  # Convert to TFlops
    print0(f"[bold green][/bold green] Cute-DSL Fwd + Bwd Average time: {e2e_timing:.3f} ms, TFLOPS: {tflops:.1f}")

    if is_glu(activation):
        flops = 12 * T * I * H * K
    else:
        flops = 8 * T * I * H * K

    bwd_time = e2e_timing - fwd_timing
    tflops = flops / (bwd_time / 1e3) / 1e12
    print0(f"[bold green][/bold green] Cute-DSL Bwd Average time: {bwd_time:.3f} ms, TFLOPS: {tflops:.1f}")


if __name__ == "__main__":
    args = parse_arguments()
    run(
        args.thiek,
        args.dtype,
        args.skip_test,
        args.add_bias,
        args.activation,
        args.fp8_protocol,
        args.report_fp8_metrics,
        args.report_stage_memory,
        args.report_fp8_analysis,
        args.prefetch_fp8_weights,
        args.native_fp8_forward,
        fp8_mode=args.fp8_mode,
    )
    print("PASS")
