"""Shared fixtures, precision helpers, gold references, and shape constants for FP8 op tests."""
import os
import math

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

def _has_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def _has_quack() -> bool:
    return os.getenv("USE_QUACK_GEMM", "0") == "1"


requires_blackwell = pytest.mark.skipif(
    not _has_blackwell(), reason="Requires SM100+ GPU (SM100+)"
)
requires_quack = pytest.mark.skipif(
    not _has_quack(), reason="Requires USE_QUACK_GEMM=1"
)


# ---------------------------------------------------------------------------
# Precision helpers
# ---------------------------------------------------------------------------

def rrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative root-mean-square error: ||a-b|| / ||b||."""
    a = actual.float().flatten()
    b = expected.float().flatten()
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def cosine_sim(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Cosine similarity between flattened tensors."""
    a = actual.float().flatten()
    b = expected.float().flatten()
    return ((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def assert_fp8_tolerance(actual, expected, rrmse_max=0.10, cosine_min=0.99):
    r = rrmse(actual, expected)
    c = cosine_sim(actual, expected)
    assert r < rrmse_max, f"RRMSE {r:.6f} >= {rrmse_max}"
    assert c > cosine_min, f"cosine {c:.6f} <= {cosine_min}"


def assert_bf16_close(actual, expected, atol=1.4e-2):
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=atol, rtol=0.0
    )


def assert_byte_exact(actual: torch.Tensor, expected: torch.Tensor):
    a = actual.contiguous().view(torch.uint8)
    b = expected.contiguous().view(torch.uint8)
    mismatches = (a != b).sum().item()
    total = a.numel()
    assert mismatches == 0, (
        f"Byte mismatch: {mismatches}/{total} bytes differ "
        f"({mismatches/total*100:.2f}%)"
    )


# ---------------------------------------------------------------------------
# E8M0 gold reference (pure torch, no triton)
# Ported from tests/test_cute_blockscaled.py:27-47
# ---------------------------------------------------------------------------

GROUP_SIZE = 32


def _e8m0_quant_groups(src_f32: torch.Tensor, group_dim: int, num_groups: int):
    """Core E8M0 quantization loop. Returns (fp8_out, raw_scales_u8).

    src_f32: float32 tensor
    group_dim: axis along which groups of 32 are taken
    num_groups: number of groups along group_dim
    """
    shape = src_f32.shape
    fp8_out = torch.empty(shape, dtype=torch.float8_e4m3fn, device=src_f32.device)

    if group_dim == 1:
        # Row-wise: groups along dim=1 (columns), amax per-row within group
        scale_shape = (shape[0], num_groups)
        scale_out = torch.empty(scale_shape, dtype=torch.uint8, device=src_f32.device)
        for g in range(num_groups):
            c_start = g * GROUP_SIZE
            c_end = min(c_start + GROUP_SIZE, shape[1])
            group_data = src_f32[:, c_start:c_end]
            amax = group_data.abs().amax(dim=1)  # per-row amax
            amax = amax.clamp(min=1e-12)

            amax_bits = amax.view(torch.int32)
            biased_exp = (amax_bits >> 23) & 0xFF
            mantissa = amax_bits & 0x7FFFFF
            carry = (mantissa > 0x600000).int()
            e8m0 = biased_exp - 8 + carry
            e8m0 = e8m0.clamp(min=0)

            qexp = (254 - e8m0).clamp(1, 254)
            quant_scale = (qexp.int() << 23).view(torch.float32)

            quantized = (group_data * quant_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            fp8_out[:, c_start:c_end] = quantized
            scale_out[:, g] = e8m0.to(torch.uint8)
    else:
        # Col-wise: groups along dim=0 (rows), amax per-column within group
        scale_shape = (shape[1], num_groups)
        scale_out = torch.empty(scale_shape, dtype=torch.uint8, device=src_f32.device)
        for g in range(num_groups):
            r_start = g * GROUP_SIZE
            r_end = min(r_start + GROUP_SIZE, shape[0])
            group_data = src_f32[r_start:r_end, :]
            amax = group_data.abs().amax(dim=0)  # per-column amax
            amax = amax.clamp(min=1e-12)

            amax_bits = amax.view(torch.int32)
            biased_exp = (amax_bits >> 23) & 0xFF
            mantissa = amax_bits & 0x7FFFFF
            carry = (mantissa > 0x600000).int()
            e8m0 = biased_exp - 8 + carry
            e8m0 = e8m0.clamp(min=0)

            qexp = (254 - e8m0).clamp(1, 254)
            quant_scale = (qexp.int() << 23).view(torch.float32)

            quantized = (group_data * quant_scale.unsqueeze(0)).to(torch.float8_e4m3fn)
            fp8_out[r_start:r_end, :] = quantized
            scale_out[:, g] = e8m0.to(torch.uint8)

    return fp8_out, scale_out


def gold_e8m0_row_quant(src_bf16: torch.Tensor, group: int = 32):
    """Row-wise E8M0 blockscaled quant (groups along columns).

    Returns (fp8 (M,K), raw_scales_u8 (M, K//group)).
    """
    assert group == GROUP_SIZE
    M, K = src_bf16.shape
    num_groups = math.ceil(K / group)
    return _e8m0_quant_groups(src_bf16.float(), group_dim=1, num_groups=num_groups)


def gold_e8m0_col_quant(src_bf16: torch.Tensor, group: int = 32):
    """Col-wise E8M0 blockscaled quant (groups along rows).

    Returns (fp8 (TK,dim), raw_scales_u8 (dim, TK//group)).
    """
    assert group == GROUP_SIZE
    TK, dim = src_bf16.shape
    num_groups = math.ceil(TK / group)
    return _e8m0_quant_groups(src_bf16.float(), group_dim=0, num_groups=num_groups)


def gold_e8m0_iso32_quant(src_bf16: torch.Tensor, group: int = 32):
    """32x32 isotropic blockscaled quant (all 32 rows share same scale per group).

    Returns (fp8 (M,K), raw_scales_u8 (M, K//group)).
    """
    assert group == GROUP_SIZE
    src_f32 = src_bf16.float()
    M, K = src_f32.shape
    num_groups = math.ceil(K / group)

    fp8_out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=src_f32.device)
    scale_out = torch.empty(M, num_groups, dtype=torch.uint8, device=src_f32.device)

    # Process in 32-row blocks; within each block, amax is over all 32 rows
    num_row_blocks = math.ceil(M / 32)
    for rb in range(num_row_blocks):
        r_start = rb * 32
        r_end = min(r_start + 32, M)
        for g in range(num_groups):
            c_start = g * GROUP_SIZE
            c_end = min(c_start + GROUP_SIZE, K)
            block = src_f32[r_start:r_end, c_start:c_end]
            amax = block.abs().amax()  # single scalar for the whole 32x32 block
            amax = amax.clamp(min=1e-12)

            amax_bits = amax.view(torch.int32)
            biased_exp = (amax_bits >> 23) & 0xFF
            mantissa = amax_bits & 0x7FFFFF
            carry = (mantissa > 0x600000).int()
            e8m0 = biased_exp - 8 + carry
            e8m0 = e8m0.clamp(min=0)

            qexp = (254 - e8m0).clamp(1, 254)
            quant_scale = (qexp.int() << 23).view(torch.float32)

            quantized = (block * quant_scale).to(torch.float8_e4m3fn)
            fp8_out[r_start:r_end, c_start:c_end] = quantized
            scale_out[r_start:r_end, g] = e8m0.to(torch.uint8)

    return fp8_out, scale_out


def gold_dequant(fp8: torch.Tensor, scales_u8: torch.Tensor, group: int = 32):
    """Dequantize blockscaled FP8 to bf16: fp8.float() * 2^(scale) per group.

    fp8: (M, K) float8_e4m3fn
    scales_u8: (M, K//group) uint8 — raw E8M0 scale bytes
    """
    assert group == GROUP_SIZE
    M, K = fp8.shape
    num_groups = scales_u8.shape[1]
    fp8_f32 = fp8.float()
    out = torch.empty(M, K, dtype=torch.bfloat16, device=fp8.device)
    for g in range(num_groups):
        c_start = g * group
        c_end = min(c_start + group, K)
        e8m0 = scales_u8[:, g].to(torch.int32)
        scale_f32 = (e8m0 << 23).view(torch.float32)
        out[:, c_start:c_end] = (fp8_f32[:, c_start:c_end] * scale_f32.unsqueeze(1)).to(torch.bfloat16)
    return out


# ---------------------------------------------------------------------------
# ISA scale unpack helper
# ---------------------------------------------------------------------------

def unpack_isa_scales(packed: torch.Tensor, rows: int, cols: int):
    """Unpack ISA-layout E8M0 scales back to raw (rows, num_groups) uint8.

    Uses the same index mapping as pack_blockscaled_1x32_scales but in reverse.
    """
    SF_VEC_SIZE = 32
    SF_TILE_M = 128
    SF_TILE_K = 128
    SF_TILE_STORAGE = SF_TILE_M * (SF_TILE_K // SF_VEC_SIZE)

    scale_cols = math.ceil(cols / SF_VEC_SIZE)
    k_tiles = math.ceil(cols / SF_TILE_K)

    packed_u8 = packed.contiguous().view(torch.uint8).flatten()

    row_ids = torch.arange(rows, device=packed.device, dtype=torch.int64).unsqueeze(1)
    scale_block_ids = torch.arange(scale_cols, device=packed.device, dtype=torch.int64).unsqueeze(0)

    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    k_tiles_idx = scale_block_ids // (SF_TILE_K // SF_VEC_SIZE)
    k_in_tile = scale_block_ids % (SF_TILE_K // SF_VEC_SIZE)

    tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    index = (tile_base + row_base + k_in_tile).reshape(-1)

    raw = packed_u8[index].reshape(rows, scale_cols)
    return raw


# ---------------------------------------------------------------------------
# FP8 → BF16 dequant + compare template
# ---------------------------------------------------------------------------

def dequant_isa_and_compare(
    fp8_data: torch.Tensor,
    isa_packed_scales: torch.Tensor,
    expected_bf16: torch.Tensor,
    *,
    rrmse_max: float = 0.10,
    cosine_min: float = 0.99,
    label: str = "",
):
    """Unpack ISA scales → dequant FP8 → compare against expected BF16 reference.

    This is the standard template for comparing an FP8 kernel output against a
    BF16 reference. Steps:
      1. Unpack ISA-packed E8M0 scales to raw (rows, num_groups) uint8
      2. Dequantize: fp8.float() * 2^(e8m0) per group → bf16
      3. Assert RRMSE and cosine similarity within tolerance

    Parameters
    ----------
    fp8_data : (M, K) float8_e4m3fn
    isa_packed_scales : (1, packed_size) float8_e8m0fnu
    expected_bf16 : (M, K) bfloat16 — trusted reference
    rrmse_max : max acceptable RRMSE
    cosine_min : min acceptable cosine similarity
    label : optional label for error messages
    """
    M, K = fp8_data.shape
    raw_scales = unpack_isa_scales(isa_packed_scales, M, K)
    dequant_bf16 = gold_dequant(fp8_data, raw_scales)
    r = rrmse(dequant_bf16, expected_bf16)
    c = cosine_sim(dequant_bf16, expected_bf16)
    prefix = f"[{label}] " if label else ""
    assert r < rrmse_max, f"{prefix}RRMSE {r:.6f} >= {rrmse_max}"
    assert c > cosine_min, f"{prefix}cosine {c:.6f} <= {cosine_min}"
    return r, c


def dequant_raw_and_compare(
    fp8_data: torch.Tensor,
    raw_scales_u8: torch.Tensor,
    expected_bf16: torch.Tensor,
    *,
    rrmse_max: float = 0.10,
    cosine_min: float = 0.99,
    label: str = "",
):
    """Dequant FP8 with raw scales → compare against expected BF16 reference.

    Same as dequant_isa_and_compare but for raw (non-ISA-packed) scales.

    Parameters
    ----------
    fp8_data : (M, K) float8_e4m3fn
    raw_scales_u8 : (M, K//32) uint8 or float8_e8m0fnu
    expected_bf16 : (M, K) bfloat16 — trusted reference
    """
    scales = raw_scales_u8
    if scales.dtype != torch.uint8:
        scales = scales.view(torch.uint8)
    dequant_bf16 = gold_dequant(fp8_data, scales)
    r = rrmse(dequant_bf16, expected_bf16)
    c = cosine_sim(dequant_bf16, expected_bf16)
    prefix = f"[{label}] " if label else ""
    assert r < rrmse_max, f"{prefix}RRMSE {r:.6f} >= {rrmse_max}"
    assert c > cosine_min, f"{prefix}cosine {c:.6f} <= {cosine_min}"
    return r, c


# ---------------------------------------------------------------------------
# Shape constants
# ---------------------------------------------------------------------------

QUANT_SHAPES = [
    pytest.param(128, 128, id="smoke"),
    pytest.param(1024, 1536, id="aligned"),
    pytest.param(384, 1536, id="unaligned-TK"),
    pytest.param(65536, 1536, id="large-TK"),
    pytest.param(8192, 3072, id="production"),
]

GEMM_SHAPES = [
    pytest.param(256, 768, 384, 8, 8, id="smoke"),
    pytest.param(1024, 3072, 1536, 8, 8, id="aligned"),
    pytest.param(512, 1536, 768, 4, 4, id="small-E"),
    pytest.param(2048, 3072, 1536, 32, 8, id="large-E"),
    pytest.param(8192, 3072, 1536, 8, 8, id="production"),
]

SWIGLU_SHAPES = [
    pytest.param(128, 128, id="smoke"),
    pytest.param(1024, 768, id="medium"),
    pytest.param(384, 1536, id="unaligned-TK"),
    pytest.param(8192, 1536, id="production"),
    pytest.param(2048, 3072, id="wide-I"),
]

SEEDS = [42, 123, 777]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_cuda_memory():
    """Flush CUDA memory and CUTLASS caches between tests to prevent workspace corruption.

    Root cause: CUTLASS blockscaled GEMM (quack-kernels 0.3.7) retains internal
    state tied to GPU memory addresses from prior kernel launches.  When PyTorch's
    CUDA allocator reclaims and reuses that memory for a different problem shape,
    the stale state causes subsequent GEMMs to produce garbage output (RRMSE ≈ √2).

    Fix: clear all CUTLASS plan/fast-path caches AND flush the CUDA allocator
    before each test, so every test starts with a clean GPU state.
    """
    import gc
    import sys
    # Clear CUTLASS kernel caches to avoid stale compiled-object references
    bfp8 = sys.modules.get("sonicmoe.quack_utils.blockscaled_fp8_gemm")
    if bfp8 is not None:
        if hasattr(bfp8, "_GEMM_FAST_PATH"):
            bfp8._GEMM_FAST_PATH.clear()
        if hasattr(bfp8, "_GEMM_FAST_PATH_VK"):
            bfp8._GEMM_FAST_PATH_VK.clear()
        if hasattr(bfp8, "_COMPILE_CACHE"):
            bfp8._COMPILE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    yield


@pytest.fixture(params=SEEDS, ids=[f"seed{s}" for s in SEEDS])
def seed(request):
    s = request.param
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    return s
