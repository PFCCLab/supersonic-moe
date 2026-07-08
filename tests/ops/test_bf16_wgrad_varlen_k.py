"""Isolated BF16 varlen-K wgrad tests and opt-in A35B tuning bench.

The A35B training path uses:
  up-proj   dw1: (M,N,E) = (7168,7168,20), A=x.T,    B=dz
  down-proj dw2: (M,N,E) = (7168,3584,20), A=dout.T, B=y1s

Both call ``bf16_wgrad_gemm_varlen_k*``.  This file keeps a small always-on
correctness smoke test, a large-shape auto-config regression against the
historical ``tile_m=128,tile_n=128,cluster=1x1`` reference, and an opt-in timing
sweep for tuning.

Usage:
  CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
    python -m pytest tests/ops/test_bf16_wgrad_varlen_k.py -v -s

  CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
    SONIC_MOE_RUN_BF16_WGRAD_TUNE=1 \
    SONIC_MOE_RUN_A35B_BF16_WGRAD_BENCH=1 \
    SONIC_MOE_BF16_WGRAD_BENCH_TOKENS_PER_EXPERT=2048 \
    python -m pytest tests/ops/test_bf16_wgrad_varlen_k.py::test_bf16_wgrad_config_sweep -v -s
"""

from __future__ import annotations

import json
import os
import statistics

import pytest
import torch

from tests.ops.conftest import requires_blackwell, requires_quack


pytestmark = [requires_blackwell, requires_quack]

DEFAULT_CONFIG = "128,128,1,1,1,8"
DEFAULT_CANDIDATES = [
    DEFAULT_CONFIG,
    "128,192,1,1,1,8",
    "128,256,1,1,1,8",
    "128,128,2,1,1,8",
    "128,192,2,1,1,8",
    "128,256,2,1,1,8",
    "256,192,2,1,1,8",
    "256,256,2,1,1,8",
    "256,256,2,1,0,8",
]


def _clear_fast_paths() -> None:
    from sonicmoe.quack_utils import bf16_wgrad_gemm as wgrad_mod

    wgrad_mod._GEMM_FAST_PATH_BF16_VK.clear()
    wgrad_mod._GEMM_FAST_PATH_BF16_VK_ACCUM.clear()
    wgrad_mod._GEMM_FAST_PATH_BF16_VK_TMA_ADD.clear()


def _make_config(config: str):
    from quack.gemm_config import GemmConfig
    from quack.cute_dsl_utils import get_device_capacity

    fields = [int(part.strip()) for part in config.split(",")]
    assert len(fields) == 6
    tile_m, tile_n, cluster_m, cluster_n, use_clc, max_swizzle = fields
    cap = get_device_capacity(torch.device("cuda"))[0]
    return GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        pingpong=False,
        swap_ab=False,
        is_dynamic_persistent=bool(use_clc),
        max_swizzle_size=max_swizzle,
        device_capacity=cap,
    )


def _make_case(
    *,
    M: int,
    N: int,
    E: int,
    tokens_per_expert: int,
    source_tokens: int,
    layout: str,
    seed: int = 20260708,
) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    total_K = E * tokens_per_expert

    # Real callers pass x.T / dout.T.  That view has stride (1, M), so an
    # arbitrary token gather still lands on a 16B boundary when M is divisible
    # by 8 bf16 elements.  A contiguous (M, T) tensor would only guarantee 2B
    # alignment after gather and CuTe correctly rejects the async copy.
    A_base = (0.02 * torch.randn(source_tokens, M, dtype=dtype, device=device)).contiguous()
    A = A_base.T
    B = (0.02 * torch.randn(total_K, N, dtype=dtype, device=device)).contiguous()
    cu = torch.arange(0, total_K + 1, tokens_per_expert, dtype=torch.int32, device=device)

    # Deterministic per-expert gather.  Values are sorted inside each expert slice,
    # matching the access pattern after routing metadata sort.
    idx_parts = []
    base = torch.arange(tokens_per_expert, dtype=torch.int32, device=device)
    for expert in range(E):
        idx_parts.append(((base + expert * 17) % source_tokens).sort()[0])
    A_idx = torch.cat(idx_parts).contiguous()
    return {
        "M": M,
        "N": N,
        "E": E,
        "total_K": total_K,
        "A": A,
        "A_base": A_base,
        "B": B,
        "cu": cu,
        "A_idx": A_idx,
        "layout": layout,
        "dtype": dtype,
        "device": device,
    }


def _make_case_from_counts(
    *,
    M: int,
    N: int,
    expert_counts: list[int],
    source_tokens: int,
    layout: str,
    seed: int = 20260708,
) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    total_K = sum(expert_counts)
    E = len(expert_counts)

    A_base = (0.02 * torch.randn(source_tokens, M, dtype=dtype, device=device)).contiguous()
    A = A_base.T
    B = (0.02 * torch.randn(total_K, N, dtype=dtype, device=device)).contiguous()
    cu_host = [0]
    for count in expert_counts:
        cu_host.append(cu_host[-1] + count)
    cu = torch.tensor(cu_host, dtype=torch.int32, device=device)

    idx_parts = []
    for expert, count in enumerate(expert_counts):
        if count == 0:
            continue
        base = torch.arange(count, dtype=torch.int32, device=device)
        idx_parts.append(((base + expert * 17) % source_tokens).sort()[0])
    A_idx = (
        torch.cat(idx_parts).contiguous()
        if idx_parts else torch.empty((0,), dtype=torch.int32, device=device)
    )
    return {
        "M": M,
        "N": N,
        "E": E,
        "total_K": total_K,
        "A": A,
        "A_base": A_base,
        "B": B,
        "cu": cu,
        "A_idx": A_idx,
        "layout": layout,
        "dtype": dtype,
        "device": device,
        "expert_counts": expert_counts,
    }


def _make_out(case: dict):
    E, M, N = case["E"], case["M"], case["N"]
    if case["layout"] == "upproj_transposed":
        base = torch.empty((E, N, M), dtype=case["dtype"], device=case["device"])
        return base.permute(0, 2, 1), base
    out = torch.empty((E, M, N), dtype=case["dtype"], device=case["device"])
    return out, out


def _run(case: dict, out: torch.Tensor) -> torch.Tensor:
    from sonicmoe.quack_utils.bf16_wgrad_gemm import bf16_wgrad_gemm_varlen_k

    return bf16_wgrad_gemm_varlen_k(
        case["A"],
        case["B"],
        case["cu"],
        case["A_idx"],
        out=out,
        M=case["M"],
        N=case["N"],
        total_K=case["total_K"],
        num_experts=case["E"],
        device=case["device"],
    )


def _run_with_config(case: dict, out: torch.Tensor, config: str) -> torch.Tensor:
    from sonicmoe.quack_utils import bf16_wgrad_gemm as wgrad_mod

    return wgrad_mod._run_bf16_wgrad_varlen_k(
        case["A"],
        case["B"],
        case["cu"],
        case["A_idx"],
        out=out,
        C=None,
        M=case["M"],
        N=case["N"],
        total_K=case["total_K"],
        num_experts=case["E"],
        device=case["device"],
        variant="bf16_vk",
        compile_cache=wgrad_mod._COMPILE_CACHE_BF16_VK,
        fast_cache=wgrad_mod._GEMM_FAST_PATH_BF16_VK,
        epi_args=wgrad_mod.GemmDefaultSm100.EpilogueArguments(),
        config=_make_config(config),
    )


def _gold(case: dict) -> torch.Tensor:
    A = case["A"].float()
    B = case["B"].float()
    A_idx = case["A_idx"].long()
    cu = case["cu"]
    E, M, N = case["E"], case["M"], case["N"]
    out = torch.empty((E, M, N), dtype=torch.bfloat16, device=case["device"])
    for expert in range(E):
        start = int(cu[expert].item())
        end = int(cu[expert + 1].item())
        gathered = A[:, A_idx[start:end]]
        out[expert] = (gathered @ B[start:end]).to(torch.bfloat16)
    return out


def _assert_byte_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.numel() > 64 * 1024 * 1024 and actual.ndim > 0:
        for index in range(actual.shape[0]):
            _assert_byte_equal(actual[index], expected[index])
        return
    a = actual.contiguous().view(torch.uint8)
    b = expected.contiguous().view(torch.uint8)
    mismatch = int((a != b).sum().item())
    assert mismatch == 0, f"byte mismatch: {mismatch}/{a.numel()} bytes"


def _time_config(case: dict, config: str, warmup: int, repeats: int) -> dict:
    out, _ = _make_out(case)
    _clear_fast_paths()
    for _ in range(warmup):
        _run_with_config(case, out, config)
    torch.cuda.synchronize()

    first, _ = _make_out(case)
    second, _ = _make_out(case)
    _run_with_config(case, first, config)
    _run_with_config(case, second, config)
    torch.cuda.synchronize()
    deterministic = torch.equal(first, second)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(repeats):
        start.record()
        _run_with_config(case, out, config)
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)

    times_sorted = sorted(times_us)
    return {
        "config": config,
        "median_us": statistics.median(times_us),
        "p20_us": times_sorted[max(0, int(0.20 * (len(times_sorted) - 1)))],
        "p80_us": times_sorted[min(len(times_sorted) - 1, int(0.80 * (len(times_sorted) - 1)))],
        "deterministic": deterministic,
        "out": out.detach(),
    }


@pytest.mark.parametrize(
    "M,N,E,tokens_per_expert,source_tokens,layout",
    [
        pytest.param(256, 512, 4, 128, 256, "contiguous", id="dw2-contiguous-smoke"),
        pytest.param(384, 768, 4, 128, 256, "upproj_transposed", id="dw1-transposed-smoke"),
    ],
)
def test_bf16_wgrad_default_correctness_and_determinism(
    M: int,
    N: int,
    E: int,
    tokens_per_expert: int,
    source_tokens: int,
    layout: str,
) -> None:
    case = _make_case(
        M=M,
        N=N,
        E=E,
        tokens_per_expert=tokens_per_expert,
        source_tokens=source_tokens,
        layout=layout,
    )
    first, _ = _make_out(case)
    second, _ = _make_out(case)
    _run_with_config(case, first, DEFAULT_CONFIG)
    _run_with_config(case, second, DEFAULT_CONFIG)
    torch.cuda.synchronize()

    _assert_byte_equal(first, second)
    torch.testing.assert_close(first.float(), _gold(case).float(), atol=3e-2, rtol=2e-2)


def test_bf16_wgrad_auto_large_shape_matches_reference_config() -> None:
    case = _make_case(
        M=4096,
        N=2048,
        E=2,
        tokens_per_expert=32,
        source_tokens=64,
        layout="contiguous",
    )
    reference, _ = _make_out(case)
    _clear_fast_paths()
    _run_with_config(case, reference, DEFAULT_CONFIG)

    first, _ = _make_out(case)
    second, _ = _make_out(case)
    _clear_fast_paths()
    _run(case, first)
    _run(case, second)
    torch.cuda.synchronize()

    _assert_byte_equal(first, second)
    _assert_byte_equal(first, reference)


@pytest.mark.parametrize(
    "M,N",
    [
        pytest.param(4100, 2048, id="unaligned-a-stride"),
        pytest.param(4096, 2052, id="unaligned-b-d-stride"),
    ],
)
def test_bf16_wgrad_rejects_unaligned_dynamic_strides(M: int, N: int) -> None:
    case = _make_case(
        M=M,
        N=N,
        E=2,
        tokens_per_expert=32,
        source_tokens=64,
        layout="contiguous",
    )
    out, _ = _make_out(case)
    _clear_fast_paths()
    with pytest.raises(RuntimeError, match="16-byte alignment"):
        _run_with_config(case, out, "256,256,2,1,1,8")


def _assert_config_matches_reference(case: dict, config: str) -> None:
    reference, _ = _make_out(case)
    _clear_fast_paths()
    _run_with_config(case, reference, DEFAULT_CONFIG)

    actual, _ = _make_out(case)
    _clear_fast_paths()
    _run_with_config(case, actual, config)
    torch.cuda.synchronize()

    _assert_byte_equal(actual, reference)
    del actual, reference
    torch.cuda.empty_cache()


def test_bf16_wgrad_2cta_large_and_skew_safety() -> None:
    if os.getenv("SONIC_MOE_RUN_BF16_WGRAD_SAFETY", "0") != "1":
        pytest.skip("Set SONIC_MOE_RUN_BF16_WGRAD_SAFETY=1 to run large safety cases")

    cases = [
        _make_case(
            M=7168,
            N=7168,
            E=20,
            tokens_per_expert=8,
            source_tokens=64,
            layout="upproj_transposed",
        ),
        _make_case(
            M=7168,
            N=3584,
            E=20,
            tokens_per_expert=8,
            source_tokens=64,
            layout="contiguous",
        ),
        _make_case_from_counts(
            M=7168,
            N=3584,
            expert_counts=[1] * 19 + [512],
            source_tokens=512,
            layout="contiguous",
        ),
    ]
    for case in cases:
        _assert_config_matches_reference(case, "256,256,2,1,1,8")


def test_bf16_wgrad_2cta_sanitizer_safety_smoke() -> None:
    if os.getenv("SONIC_MOE_RUN_BF16_WGRAD_SANITIZER", "0") != "1":
        pytest.skip("Set SONIC_MOE_RUN_BF16_WGRAD_SANITIZER=1 under compute-sanitizer")

    cases = [
        _make_case(
            M=4096,
            N=2048,
            E=4,
            tokens_per_expert=16,
            source_tokens=64,
            layout="contiguous",
        ),
        _make_case_from_counts(
            M=4096,
            N=2048,
            expert_counts=[1] * 7 + [128],
            source_tokens=128,
            layout="contiguous",
        ),
    ]
    for case in cases:
        _assert_config_matches_reference(case, "256,256,2,1,1,8")


def test_bf16_wgrad_config_sweep() -> None:
    if os.getenv("SONIC_MOE_RUN_BF16_WGRAD_TUNE", "0") != "1":
        pytest.skip("Set SONIC_MOE_RUN_BF16_WGRAD_TUNE=1 to run the tuning sweep")

    candidates = os.getenv("SONIC_MOE_BF16_WGRAD_TUNE_CONFIGS", "")
    configs = [x.strip() for x in candidates.split(";") if x.strip()] or DEFAULT_CANDIDATES
    warmup = int(os.getenv("SONIC_MOE_BF16_WGRAD_TUNE_WARMUP", "5"))
    repeats = int(os.getenv("SONIC_MOE_BF16_WGRAD_TUNE_REPEATS", "20"))

    if os.getenv("SONIC_MOE_RUN_A35B_BF16_WGRAD_BENCH", "0") == "1":
        tokens_per_expert = int(os.getenv("SONIC_MOE_BF16_WGRAD_BENCH_TOKENS_PER_EXPERT", "2048"))
        source_tokens = int(os.getenv("SONIC_MOE_BF16_WGRAD_BENCH_SOURCE_TOKENS", str(tokens_per_expert)))
        shapes = [
            ("a35b_upproj_dw1", 7168, 7168, 20, tokens_per_expert, source_tokens, "upproj_transposed"),
            ("a35b_downproj_dw2", 7168, 3584, 20, tokens_per_expert, source_tokens, "contiguous"),
        ]
    else:
        shapes = [
            ("smoke_upproj_dw1", 1024, 1024, 8, 256, 512, "upproj_transposed"),
            ("smoke_downproj_dw2", 1024, 512, 8, 256, 512, "contiguous"),
        ]

    report = []
    for name, M, N, E, tpe, src_t, layout in shapes:
        case = _make_case(
            M=M,
            N=N,
            E=E,
            tokens_per_expert=tpe,
            source_tokens=src_t,
            layout=layout,
        )
        shape_rows = []
        default_out = None
        default_time = None
        for config in configs:
            try:
                result = _time_config(case, config, warmup, repeats)
                if config == DEFAULT_CONFIG:
                    default_out = result["out"].clone()
                    default_time = result["median_us"]
                    bit_exact = True
                    speedup = 1.0
                else:
                    bit_exact = default_out is not None and torch.equal(result["out"], default_out)
                    speedup = (default_time / result["median_us"]) if default_time else None
                shape_rows.append({
                    "shape": name,
                    "config": config,
                    "median_us": round(result["median_us"], 3),
                    "p20_us": round(result["p20_us"], 3),
                    "p80_us": round(result["p80_us"], 3),
                    "speedup_vs_default": None if speedup is None else round(speedup, 4),
                    "deterministic": bool(result["deterministic"]),
                    "bit_exact_to_default": bool(bit_exact),
                })
                assert result["deterministic"], f"{name} {config} is not deterministic"
                assert bit_exact, f"{name} {config} is not byte-exact to {DEFAULT_CONFIG}"
            except AssertionError:
                raise
            except Exception as exc:
                shape_rows.append({
                    "shape": name,
                    "config": config,
                    "error": repr(exc),
                })
        assert any("median_us" in row for row in shape_rows), f"no config succeeded for {name}"
        report.extend(shape_rows)

    print(json.dumps(report, indent=2, sort_keys=True))
