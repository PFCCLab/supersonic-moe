#!/usr/bin/env python3
# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import argparse
import os
import subprocess
import sys
from pathlib import Path


_SHARDS = [
    ["tests/fp8_protocol_test.py"],
    ["tests/moe_blackwell_test.py"],
    ["tests/moe_test.py"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SM100 test shards on separate GPUs.")
    parser.add_argument(
        "--gpus",
        default="0,1,2",
        help="Comma-separated GPU ids to use. The first three entries are mapped to the three SM100 test shards.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the shard-to-GPU mapping without launching pytest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]

    if len(gpu_ids) < len(_SHARDS):
        raise SystemExit(f"Need at least {len(_SHARDS)} GPUs for the predefined shards, got {len(gpu_ids)}")

    repo_root = Path(__file__).resolve().parents[1]
    processes = []

    for shard, gpu_id in zip(_SHARDS, gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["USE_QUACK_GEMM"] = "1"
        cmd = [sys.executable, "-m", "pytest", "-q", *shard]
        print(f"[blackwell-shard] gpu={gpu_id} tests={' '.join(shard)}", flush=True)
        if args.dry_run:
            continue
        processes.append((gpu_id, shard, subprocess.Popen(cmd, cwd=repo_root, env=env)))

    exit_code = 0
    if args.dry_run:
        return exit_code

    for gpu_id, shard, process in processes:
        result = process.wait()
        if result != 0:
            print(f"[blackwell-shard] gpu={gpu_id} failed: {' '.join(shard)}", flush=True)
            exit_code = result

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
