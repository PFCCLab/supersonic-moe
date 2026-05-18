"""Worker for iso32 precision audit. Runs a single (shape, mode) on assigned GPU."""
import torch, os, gc, json, sys, numpy as np
import paddle

MODE = os.environ["__MODE__"]
T = int(os.environ["__T__"])
E = int(os.environ["__E__"])
K = int(os.environ["__K__"])
H = int(os.environ["__H__"])
I = int(os.environ["__I__"])
OUT_FILE = os.environ["__OUT_FILE__"]

os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
if MODE == "bf16":
    os.environ["SONIC_MOE_FP8_MODE"] = ""
    os.environ["SONIC_MOE_FP8_ISO32_WEIGHT"] = "0"
elif MODE == "baseline":
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    os.environ["SONIC_MOE_FP8_ISO32_WEIGHT"] = "0"
else:
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    os.environ["SONIC_MOE_FP8_ISO32_WEIGHT"] = "1"

sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
os.chdir("/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")

torch.cuda.init()
_ = torch.empty(1, device="cuda")

from tests.fp8_frontier_stress_test import _make_node_and_data
from sonicmoe.functional import _refresh_fp8_config
_refresh_fp8_config()

node, x, grad_out, tpe, d_idx, d_probs = _make_node_and_data(T, E, K, H, I, seed=42)

for _ in range(4):
    out = node.forward(x, tpe, dispatched_indices=d_idx, dispatched_probs=d_probs)
    out.backward(grad_out)
torch.cuda.synchronize()

out = node.forward(x, tpe, dispatched_indices=d_idx, dispatched_probs=d_probs)
torch.cuda.synchronize()
np.save(OUT_FILE, out.detach().float().cpu().numpy())
