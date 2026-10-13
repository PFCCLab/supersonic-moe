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

from __future__ import annotations

import gc
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import paddle
import pytest


pytestmark = pytest.mark.skipif(
    not paddle.is_compiled_with_cuda(), reason="CUDA is required"
)

_LARGE_TEST_ENV = "SONIC_MOE_RUN_LARGE_LAYOUT_TESTS"


def _load_weight_layout_fusion():
    path = (
        Path(__file__).resolve().parents[2]
        / "sonicmoe"
        / "ernie_compat"
        / "weight_layout_fusion.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_sonicmoe_weight_layout_fusion_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


weight_layout_fusion = _load_weight_layout_fusion()


def _set_cuda_device():
    paddle.set_device(os.environ.get("SONIC_MOE_LAYOUT_TEST_DEVICE", "gpu:0"))


def _cleanup_cuda():
    gc.collect()
    if hasattr(paddle.device, "cuda") and hasattr(
        paddle.device.cuda, "empty_cache"
    ):
        paddle.device.cuda.empty_cache()


def _sync_device():
    if hasattr(paddle.device, "synchronize"):
        paddle.device.synchronize()
    else:
        paddle.device.cuda.synchronize()


def _make_weight(shape, dtype):
    total = math.prod(shape)
    values = paddle.arange(total, dtype="int64").reshape(shape)
    values = (values % 4096) - 2048
    return values.cast(dtype)


def _assert_bit_exact(actual, expected):
    assert list(actual.shape) == list(expected.shape)
    assert actual.dtype == expected.dtype
    _sync_device()
    actual_bytes = actual.numpy().reshape(-1).view(np.uint8)
    expected_bytes = expected.numpy().reshape(-1).view(np.uint8)
    if np.array_equal(actual_bytes, expected_bytes):
        return
    mismatch = int(np.count_nonzero(actual_bytes != expected_bytes))
    raise AssertionError(
        f"bit-exact mismatch: {mismatch}/{actual_bytes.size} bytes differ; "
        f"actual shape={list(actual.shape)}, dtype={actual.dtype}"
    )


def _expect_scalar(tensor, value, dtype):
    expected = paddle.full([], value, dtype=dtype)
    _assert_bit_exact(tensor, expected)


def _reference_grouped_w1_to_sonic(weight):
    num_experts, hidden_size, two_i = [int(v) for v in weight.shape]
    intermediate_size = two_i // 2
    gate = weight[:, :, :intermediate_size].transpose([0, 2, 1])
    up = weight[:, :, intermediate_size:].transpose([0, 2, 1])
    return paddle.stack([gate, up], axis=2).reshape(
        [num_experts, two_i, hidden_size]
    )


def _reference_sonic_w1_to_grouped(weight):
    _num_experts, two_i, _hidden_size = [int(v) for v in weight.shape]
    intermediate_size = two_i // 2
    gate = weight[:, 0::2, :].transpose([0, 2, 1])
    up = weight[:, 1::2, :].transpose([0, 2, 1])
    return paddle.concat([gate, up], axis=2)


def _reference_transpose_w2(weight):
    return weight.transpose([0, 2, 1]).contiguous()


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 2),
        (2, 7, 10),
        (3, 63, 62),
        (4, 64, 64),
        (3, 65, 66),
        (8, 128, 256),
    ],
)
def test_grouped_w1_to_sonic_bit_exact(dtype, shape):
    _set_cuda_device()
    weight = _make_weight(shape, dtype)
    actual = weight_layout_fusion.fused_grouped_w1_to_sonic(weight)
    expected = _reference_grouped_w1_to_sonic(weight)
    _assert_bit_exact(actual, expected)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 1),
        (2, 10, 7),
        (3, 62, 63),
        (4, 64, 64),
        (3, 66, 65),
        (8, 256, 128),
    ],
)
def test_sonic_w1_to_grouped_bit_exact(dtype, shape):
    _set_cuda_device()
    weight = _make_weight(shape, dtype)
    actual = weight_layout_fusion.fused_sonic_w1_to_grouped(weight)
    expected = _reference_sonic_w1_to_grouped(weight)
    _assert_bit_exact(actual, expected)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1),
        (2, 7, 9),
        (3, 63, 31),
        (4, 64, 32),
        (3, 65, 33),
        (8, 257, 129),
    ],
)
def test_transpose_w2_bit_exact(dtype, shape):
    _set_cuda_device()
    weight = _make_weight(shape, dtype)
    actual = weight_layout_fusion.fused_transpose_w2_layout(weight)
    expected = _reference_transpose_w2(weight)
    _assert_bit_exact(actual, expected)


@pytest.mark.skipif(
    os.environ.get(_LARGE_TEST_ENV) != "1",
    reason=f"set {_LARGE_TEST_ENV}=1 to run >4GB layout tests",
)
def test_grouped_w1_to_sonic_large_tensor_crosses_int32_boundary():
    _set_cuda_device()
    dtype = "bfloat16"
    num_experts, hidden_size, intermediate_size = 2, 16384, 32769
    two_i = intermediate_size * 2
    shape = (num_experts, hidden_size, two_i)
    assert math.prod(shape) > 2**31
    assert math.prod(shape) * 2 > 4 * 1024**3

    weight = paddle.zeros(shape, dtype=dtype)
    sentinels = [
        (0, 0, 0, 1.0),
        (1, hidden_size - 1, two_i - 1, -2.0),
        (1, hidden_size // 2, intermediate_size, 3.5),
    ]
    for expert, hidden, col, value in sentinels:
        weight[expert, hidden, col] = paddle.full([], value, dtype=dtype)

    out = weight_layout_fusion.fused_grouped_w1_to_sonic(weight)
    for expert, hidden, col, value in sentinels:
        if col < intermediate_size:
            sonic_col = col * 2
        else:
            sonic_col = (col - intermediate_size) * 2 + 1
        _expect_scalar(out[expert, sonic_col, hidden], value, dtype)

    del weight, out
    _cleanup_cuda()


@pytest.mark.skipif(
    os.environ.get(_LARGE_TEST_ENV) != "1",
    reason=f"set {_LARGE_TEST_ENV}=1 to run >4GB layout tests",
)
def test_sonic_w1_to_grouped_large_tensor_crosses_int32_boundary():
    _set_cuda_device()
    dtype = "bfloat16"
    num_experts, hidden_size, intermediate_size = 2, 16384, 32769
    two_i = intermediate_size * 2
    shape = (num_experts, two_i, hidden_size)
    assert math.prod(shape) > 2**31
    assert math.prod(shape) * 2 > 4 * 1024**3

    weight = paddle.zeros(shape, dtype=dtype)
    sentinels = [
        (0, 0, 0, 1.0),
        (1, two_i - 1, hidden_size - 1, -2.0),
        (1, 1, hidden_size // 2, 3.5),
    ]
    for expert, sonic_col, hidden, value in sentinels:
        weight[expert, sonic_col, hidden] = paddle.full([], value, dtype=dtype)

    out = weight_layout_fusion.fused_sonic_w1_to_grouped(weight)
    for expert, sonic_col, hidden, value in sentinels:
        if sonic_col % 2 == 0:
            grouped_col = sonic_col // 2
        else:
            grouped_col = intermediate_size + sonic_col // 2
        _expect_scalar(out[expert, hidden, grouped_col], value, dtype)

    del weight, out
    _cleanup_cuda()


@pytest.mark.skipif(
    os.environ.get(_LARGE_TEST_ENV) != "1",
    reason=f"set {_LARGE_TEST_ENV}=1 to run >4GB layout tests",
)
def test_transpose_w2_large_tensor_crosses_int32_boundary():
    _set_cuda_device()
    dtype = "bfloat16"
    num_experts, rows, cols = 2, 32768, 32769
    shape = (num_experts, rows, cols)
    assert math.prod(shape) > 2**31
    assert math.prod(shape) * 2 > 4 * 1024**3

    weight = paddle.zeros(shape, dtype=dtype)
    sentinels = [
        (0, 0, 0, 1.0),
        (1, rows - 1, cols - 1, -2.0),
        (1, rows // 2, cols // 2, 3.5),
    ]
    for expert, row, col, value in sentinels:
        weight[expert, row, col] = paddle.full([], value, dtype=dtype)

    out = weight_layout_fusion.fused_transpose_w2_layout(weight)
    for expert, row, col, value in sentinels:
        _expect_scalar(out[expert, col, row], value, dtype)

    del weight, out
    _cleanup_cuda()
