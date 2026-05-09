import pytest
pytestmark = pytest.mark.skip(reason="Requires native PyTorch torch._dynamo/torch.compile, not available in Paddle compat proxy")

# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch

from sonicmoe import KernelBackendMoE, MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType

from .test_commons import TestCommons


_SEED = 42
_BLACKWELL_SMOKE_SHAPE = (256, 768, 256, 128, 8)

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch._functorch.config.donated_buffer = False


class BlackwellMoETest(TestCommons):
    def test_moe_blackwell_quack(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)

        T, H, I, E, K = _BLACKWELL_SMOKE_SHAPE
        dtype = torch.bfloat16
        device = torch.device("cuda")

        with torch.device(device):
            moe = MoE(
                num_experts=E,
                num_experts_per_tok=K,
                hidden_size=H,
                intermediate_size=I,
                activation_function=ActivationType.SWIGLU,
                add_bias=False,
                std=0.02,
            ).to(dtype=dtype)

        x_torch = 0.02 * torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
        x_kernel = x_torch.clone().detach().requires_grad_()
        dy_torch = 0.02 * torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
        dy_kernel = dy_torch.clone().detach().requires_grad_()

        weights = list(moe.parameters())

        with torch.autocast(device.type, torch.float32):
            with enable_quack_gemm(True):
                y_kernel = moe(x_kernel, kernel_backend_moe=KernelBackendMoE.sonicmoe)[0]
                kernel_grads = torch.autograd.grad(
                    y_kernel,
                    [x_kernel] + weights,
                    grad_outputs=dy_kernel,
                    retain_graph=True,
                )

            y_torch = moe(x_torch, kernel_backend_moe=KernelBackendMoE.torch)[0]
            torch_grads = torch.autograd.grad(
                y_torch,
                [x_torch] + weights,
                grad_outputs=dy_torch,
                retain_graph=True,
            )

        self.assert_equal_tensors(
            y_kernel.float(),
            y_torch.float(),
            False,
            atol_bfloat16=1.4e-2,
            rtol_bfloat16=2e-2,
            dtype=dtype,
        )

        for torch_grad, kernel_grad in zip(torch_grads, kernel_grads):
            self.assert_equal_tensors(
                kernel_grad.float(),
                torch_grad.float(),
                False,
                atol_bfloat16=2e-2,
                rtol_bfloat16=2e-2,
                dtype=dtype,
            )

        torch.cuda.empty_cache()
