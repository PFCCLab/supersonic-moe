# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Triton kernels for SonicMoE expert weight layout conversion."""

from functools import cache

import paddle
import triton
import triton.language as tl
from triton.runtime.driver import _create_driver, driver


@triton.jit
def _grouped_w1_to_sonic_kernel(
    src,
    dst,
    total: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    two_i: tl.constexpr = intermediate_size * 2
    expert_stride: tl.constexpr = hidden_size * two_i
    rem = offsets % expert_stride
    expert = offsets // expert_stride
    sonic_col = rem // hidden_size
    hidden = rem - sonic_col * hidden_size
    src_col = sonic_col // 2 + (sonic_col % 2) * intermediate_size
    src_offsets = expert * expert_stride + hidden * two_i + src_col
    tl.store(dst + offsets, tl.load(src + src_offsets, mask=mask), mask=mask)


@triton.jit
def _sonic_w1_to_grouped_kernel(
    src,
    dst,
    total: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    two_i: tl.constexpr = intermediate_size * 2
    expert_stride: tl.constexpr = hidden_size * two_i
    rem = offsets % expert_stride
    expert = offsets // expert_stride
    hidden = rem // two_i
    grouped_col = rem - hidden * two_i
    src_col = tl.where(
        grouped_col < intermediate_size,
        grouped_col * 2,
        (grouped_col - intermediate_size) * 2 + 1,
    )
    src_offsets = expert * expert_stride + src_col * hidden_size + hidden
    tl.store(dst + offsets, tl.load(src + src_offsets, mask=mask), mask=mask)


@triton.jit
def _transpose_last2_kernel(
    src,
    dst,
    total: tl.constexpr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    expert_stride: tl.constexpr = rows * cols
    rem = offsets % expert_stride
    expert = offsets // expert_stride
    dst_col = rem // rows
    dst_row = rem - dst_col * rows
    src_offsets = expert * expert_stride + dst_row * cols + dst_col
    tl.store(dst + offsets, tl.load(src + src_offsets, mask=mask), mask=mask)


@triton.jit
def _grouped_w1_to_sonic_tiled_kernel(
    src,
    dst,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_h = tl.program_id(1)
    expert = tl.program_id(2)
    hidden = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    inter = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    two_i: tl.constexpr = intermediate_size * 2
    expert_stride: tl.constexpr = hidden_size * two_i

    src_mask = (hidden[:, None] < hidden_size) & (
        inter[None, :] < intermediate_size
    )
    src_base = expert * expert_stride + hidden[:, None] * two_i
    gate = tl.load(src + src_base + inter[None, :], mask=src_mask, other=0.0)
    up = tl.load(
        src + src_base + intermediate_size + inter[None, :],
        mask=src_mask,
        other=0.0,
    )

    dst_mask = (inter[:, None] < intermediate_size) & (
        hidden[None, :] < hidden_size
    )
    dst_base = expert * expert_stride + hidden[None, :]
    tl.store(
        dst + dst_base + (inter[:, None] * 2) * hidden_size,
        tl.trans(gate),
        mask=dst_mask,
    )
    tl.store(
        dst + dst_base + (inter[:, None] * 2 + 1) * hidden_size,
        tl.trans(up),
        mask=dst_mask,
    )


@triton.jit
def _sonic_w1_to_grouped_tiled_kernel(
    src,
    dst,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_h = tl.program_id(1)
    expert = tl.program_id(2)
    inter = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    hidden = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    two_i: tl.constexpr = intermediate_size * 2
    expert_stride: tl.constexpr = hidden_size * two_i

    src_mask = (inter[:, None] < intermediate_size) & (
        hidden[None, :] < hidden_size
    )
    src_base = expert * expert_stride + hidden[None, :]
    gate = tl.load(
        src + src_base + (inter[:, None] * 2) * hidden_size,
        mask=src_mask,
        other=0.0,
    )
    up = tl.load(
        src + src_base + (inter[:, None] * 2 + 1) * hidden_size,
        mask=src_mask,
        other=0.0,
    )

    dst_mask = (hidden[:, None] < hidden_size) & (
        inter[None, :] < intermediate_size
    )
    dst_base = expert * expert_stride + hidden[:, None] * two_i
    tl.store(dst + dst_base + inter[None, :], tl.trans(gate), mask=dst_mask)
    tl.store(
        dst + dst_base + intermediate_size + inter[None, :],
        tl.trans(up),
        mask=dst_mask,
    )


@triton.jit
def _transpose_last2_tiled_kernel(
    src,
    dst,
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.program_id(2)
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    expert_stride: tl.constexpr = rows * cols
    mask = (row[:, None] < rows) & (col[None, :] < cols)
    values = tl.load(
        src + expert * expert_stride + row[:, None] * cols + col[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(
        dst + expert * expert_stride + col[:, None] * rows + row[None, :],
        tl.trans(values),
        mask=tl.trans(mask),
    )


@cache
def _paddle_triton_driver():
    if not paddle.is_compiled_with_cuda() or not hasattr(
        paddle, "use_compat_guard"
    ):
        return None
    try:
        with paddle.use_compat_guard(enable=True, silent=True):
            return _create_driver()
    except Exception:
        return None


def _launch(kernel, grid, *args, **meta):
    paddle_driver = _paddle_triton_driver()
    if paddle_driver is None:
        kernel[grid](*args, **meta)
        return

    driver.set_active(paddle_driver)
    try:
        kernel[grid](*args, **meta)
    finally:
        driver.reset_active()


def _launch_1d(kernel, total, *args):
    block = 256
    grid = (triton.cdiv(total, block),)
    _launch(kernel, grid, *args, BLOCK=block, num_warps=4)


def fused_grouped_w1_to_sonic(weight):
    """Convert W1 [E, H, 2I] grouped layout to [E, 2I, H] Sonic layout."""
    assert len(weight.shape) == 3, f"expected rank-3 weight, got {weight.shape}"
    assert weight.shape[2] % 2 == 0, (
        f"W1 last dim must be even, got {weight.shape}"
    )
    num_experts, hidden_size, two_i = [int(v) for v in weight.shape]
    intermediate_size = two_i // 2
    out = paddle.empty([num_experts, two_i, hidden_size], dtype=weight.dtype)
    block_h, block_i = 64, 32
    grid = (
        triton.cdiv(intermediate_size, block_i),
        triton.cdiv(hidden_size, block_h),
        num_experts,
    )
    _launch(
        _grouped_w1_to_sonic_tiled_kernel,
        grid,
        weight,
        out,
        hidden_size,
        intermediate_size,
        BLOCK_H=block_h,
        BLOCK_I=block_i,
        num_warps=4,
    )
    return out


def fused_sonic_w1_to_grouped(weight):
    """Convert W1 [E, 2I, H] Sonic layout to [E, H, 2I] grouped layout."""
    assert len(weight.shape) == 3, f"expected rank-3 weight, got {weight.shape}"
    assert weight.shape[1] % 2 == 0, (
        f"W1 second dim must be even, got {weight.shape}"
    )
    num_experts, two_i, hidden_size = [int(v) for v in weight.shape]
    intermediate_size = two_i // 2
    out = paddle.empty([num_experts, hidden_size, two_i], dtype=weight.dtype)
    block_i, block_h = 32, 32
    grid = (
        triton.cdiv(intermediate_size, block_i),
        triton.cdiv(hidden_size, block_h),
        num_experts,
    )
    _launch(
        _sonic_w1_to_grouped_tiled_kernel,
        grid,
        weight,
        out,
        hidden_size,
        intermediate_size,
        BLOCK_I=block_i,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


def fused_transpose_w2_layout(weight):
    """Transpose expert W2 between [E, M, N] and [E, N, M]."""
    assert len(weight.shape) == 3, f"expected rank-3 weight, got {weight.shape}"
    num_experts, rows, cols = [int(v) for v in weight.shape]
    out = paddle.empty([num_experts, cols, rows], dtype=weight.dtype)
    block_m, block_n = 64, 32
    grid = (
        triton.cdiv(rows, block_m),
        triton.cdiv(cols, block_n),
        num_experts,
    )
    _launch(
        _transpose_last2_tiled_kernel,
        grid,
        weight,
        out,
        rows,
        cols,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out
