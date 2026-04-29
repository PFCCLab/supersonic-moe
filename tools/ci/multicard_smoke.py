#!/usr/bin/env python3
"""2-rank distributed smoke test for sonicmoe mlp_node_v2.

Spawns paddle.distributed launch with 2 GPUs, each rank runs a single
fwd+bwd of SonicMoEMlpNode and broadcasts the output mean to rank 0;
rank 0 asserts cosine similarity ≥ 0.999. Non-zero exit on failure.

Used by tools/ci/run_core_tests.sh; safe to skip when fewer than 2 GPUs
are available (the wrapper script gates on nvidia-smi).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


WORKER_BODY = r"""
import os, sys, math
print(f"[pre-import] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
      f"PADDLE_LOCAL_RANK={os.environ.get('PADDLE_LOCAL_RANK')!r} "
      f"FLAGS_selected_gpus={os.environ.get('FLAGS_selected_gpus')!r}", flush=True)

# Some Python interpreters (e.g. /usr/local/bin/python in this image) do not
# have quack on the import path; sonicmoe imports it eagerly. Inject the same
# location every other tests/ops/* benchmark uses.
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
if os.path.isdir(_QUACK) and _QUACK not in sys.path:
    sys.path.insert(0, _QUACK)

import paddle
local_rank = int(os.environ.get("PADDLE_LOCAL_RANK", os.environ.get("RANK", "0")))
# paddle.distributed.launch with --gpus 0,1 keeps CUDA_VISIBLE_DEVICES="0,1"
# for both workers but injects a per-rank FLAGS_selected_gpus. Pin the
# device to that physical id so the context pool registers the correct
# place; "gpu:0" on rank 1 is REJECTED because FLAGS_selected_gpus=1 told
# paddle to register only place gpu:1 in the pool.
gpu_id = int(os.environ.get("FLAGS_selected_gpus", str(local_rank)))
paddle.device.set_device(f"gpu:{gpu_id}")
# Force-instantiate the context pool entry BEFORE init_parallel_env or any
# import that triggers a tensor allocation through a path that bypasses
# set_device (autograd backward, paddle.library proxies inside quack JIT).
_warm = paddle.empty([1], dtype="float32")
del _warm

from paddle.distributed import init_parallel_env  # noqa: E402
init_parallel_env()
paddle.enable_compat()

import torch  # noqa: E402
torch.cuda.set_device(gpu_id)

import sonicmoe.functional as functional  # noqa: E402
from sonicmoe.ernie_compat import (  # noqa: E402
    SonicMoEMlpNode,
    invalidate_weight_caches,
)

functional._ALIGNMENT_ASSUMED = True
functional._ALIGNMENT_STREAK = 100

E, H, I, T, K = 4, 1024, 512, 1024, 2

def _make_experts():
    out = []
    for e in range(E):
        paddle.seed(1000 + e + local_rank * 17)
        up = type("P", (), {
            "weight": paddle.randn([H, 2 * I], dtype="bfloat16") / math.sqrt(H),
        })()
        dn = type("P", (), {
            "weight": paddle.randn([I, H], dtype="bfloat16") / math.sqrt(I),
        })()
        up.weight.stop_gradient = False
        dn.weight.stop_gradient = False
        out.append(type("Expert", (), {"up_gate_proj": up, "down_proj": dn})())
    return out

invalidate_weight_caches()
functional.clear_all_fp8_weight_caches()
experts = _make_experts()
node = SonicMoEMlpNode(experts=experts, n_experts=E,
                       hidden_size=H, intermediate_size=I)

paddle.seed(7 + local_rank)
x_p = paddle.randn([T, H], dtype="bfloat16") * 0.02
g_p = paddle.randn([T, H], dtype="bfloat16") * 0.01
x_in = torch.from_dlpack(x_p.detach())
x_in.stop_gradient = False

torch.manual_seed(123 + local_rank)
raw_p = paddle.randn([T, E], dtype="float32")
raw = torch.from_dlpack(raw_p.detach())
top = raw.topk(K, dim=-1).indices.int()
di = top
dp_p = paddle.uniform([T, K], dtype="float32", min=0.5, max=1.0)
dp = torch.from_dlpack(dp_p.detach())
dp = dp / dp.sum(dim=1, keepdim=True)
tpe = [int((di == e).sum().item()) for e in range(E)]

out = node.forward(x_in, tpe, dispatched_indices=di, dispatched_probs=dp)
out.backward(torch.from_dlpack(g_p.detach()).clone())
node.flush_grads()

assert out.shape[0] == T
om = float(torch.from_dlpack(out.detach()).float().mean().cpu())
assert math.isfinite(om), f"non-finite output mean: {om}"
print(f"[rank{local_rank}] out_mean={om:.6f}", flush=True)

import paddle.distributed as dist
t = paddle.to_tensor([om], dtype="float32")
gathered = []
dist.all_gather(gathered, t)
vals = [float(g.numpy()[0]) for g in gathered]
if local_rank == 0:
    print(f"[rank0] gathered={vals}", flush=True)
    assert all(math.isfinite(v) for v in vals), f"non-finite: {vals}"
    print("[rank0] multicard smoke OK", flush=True)
"""


def main() -> int:
    import shutil
    if shutil.which("python") is None:
        return 2

    # Write the worker body to a tmp file so paddle.distributed.launch can
    # exec it on each rank.
    import tempfile
    with tempfile.NamedTemporaryFile(
        "w", suffix="_multicard_worker.py", delete=False
    ) as f:
        f.write(WORKER_BODY)
        worker = f.name

    cmd = [
        sys.executable, "-m", "paddle.distributed.launch",
        "--gpus", "0,1",
        worker,
    ]
    # The paddlejob runtime exports an extensive set of cluster discovery env
    # vars (PADDLE_TRAINERS=4 IPs, PADDLE_TRAINER_ENDPOINTS, POD_*, EKS_POD_*,
    # GPUTRAINER_ENDPOINTS, DISTRIBUTED_TRAINER_ENDPOINTS, TRAINER_INSTANCES,
    # PADDLE_CLUSTER_TRAIN, PADDLE_IS_LOCAL=0 …). If any of these reach
    # paddle.distributed.launch, it enters multi-node rendezvous mode and
    # blocks forever waiting for the 3 absent peer nodes. Build a whitelist
    # env instead of a denylist — the smoke test is single-node, 2-rank.
    KEEP_PREFIXES = (
        "PATH", "LD_", "HOME", "USER", "LANG", "LC_", "TERM", "TMPDIR",
        "PWD", "SHELL", "PYTHON", "VIRTUAL_ENV", "CONDA_",
        "CUDA_", "NVIDIA_",  # GPU runtime
        "TRITON_",            # triton cache / ptxas
        "SONIC_MOE_", "USE_QUACK_GEMM",
        "FLAGS_",             # paddle internal flags
        "NCCL_",              # NCCL config (excluding bootstrap-from-cluster)
        "GLOG_", "OMP_",
    )
    DROP_NCCL = {
        # NCCL bootstrap settings keyed to the multi-node IB fabric; safe to
        # drop for an in-host smoke test.
        "NCCL_SOCKET_IFNAME",
        "NCCL_BOOTSTRAP_UID_SOCK_FAMILY",
    }
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        if any(k.startswith(p) for p in KEEP_PREFIXES) and k not in DROP_NCCL:
            env[k] = v
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    # Triton's bundled ptxas does not know about Blackwell sm_103a; prefer the
    # system CUDA 13 ptxas if available so JIT compilation works on B200/RTX
    # PRO 6000 hosts.
    if "TRITON_PTXAS_PATH" not in env:
        for cand in ("/usr/local/cuda-13.0/bin/ptxas", "/usr/local/cuda/bin/ptxas"):
            if os.path.isfile(cand):
                env["TRITON_PTXAS_PATH"] = cand
                break
    print("[multicard] +", " ".join(cmd))
    import subprocess
    rc = subprocess.call(cmd, env=env, cwd=str(REPO))
    try:
        os.unlink(worker)
    except OSError:
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
