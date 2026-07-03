"""NCU profiling script specifically for colwise_quantize_and_pack.
This is the actual bottleneck kernel called in FP8 wgrad path.
"""
import torch

torch.manual_seed(42)

# reference shape
TK = 65536
I_dim = 1536
H_dim = 3072

dz_bf16 = torch.randn(TK, I_dim, dtype=torch.bfloat16, device='cuda')
x_bf16 = torch.randn(TK, H_dim, dtype=torch.bfloat16, device='cuda')

from sonicmoe.quack_utils.blockscaled_fp8_gemm import colwise_quantize_and_pack

# Warmup
for _ in range(3):
    colwise_quantize_and_pack(dz_bf16, I_dim, TK)
torch.cuda.synchronize()

# Profile colwise on dz shape (TK=65536, I=1536)
torch.cuda.nvtx.range_push("colwise_dz")
for _ in range(3):
    colwise_quantize_and_pack(dz_bf16, I_dim, TK)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

# Profile colwise on x shape (TK=65536, H=3072)
for _ in range(3):
    colwise_quantize_and_pack(x_bf16, H_dim, TK)
torch.cuda.synchronize()

torch.cuda.nvtx.range_push("colwise_x")
for _ in range(3):
    colwise_quantize_and_pack(x_bf16, H_dim, TK)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print("Done. Shapes: dz=(65536,1536), x=(65536,3072)")
