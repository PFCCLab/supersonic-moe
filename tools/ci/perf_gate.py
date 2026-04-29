#!/usr/bin/env python3
"""CI perf gate: drive bench_mlpnode_topk_nsys via nsys, assert µs/iter ≤ budget.

Used by tools/ci/run_core_tests.sh. Stand-alone usage:

    python tools/ci/perf_gate.py --shape T8192-H3072-I1536-E8-K8 \
                                 --budget-ms 4.5 --iters 12

Exits 0 on pass, 1 on fail. Prints GPU-projection µs/iter and budget.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "tests" / "ops" / "bench_mlpnode_topk_nsys.py"

_SHAPE_RE = re.compile(
    r"T(?P<T>\d+)-H(?P<H>\d+)-I(?P<I>\d+)-E(?P<E>\d+)-K(?P<K>\d+)$"
)


def parse_shape(s: str) -> dict[str, int]:
    m = _SHAPE_RE.match(s)
    if not m:
        raise SystemExit(f"bad --shape '{s}', expected like T8192-H3072-I1536-E8-K8")
    return {k: int(v) for k, v in m.groupdict().items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="T8192-H3072-I1536-E8-K8")
    p.add_argument("--budget-ms", type=float, default=4.5,
                   help="Per-iter wall-clock budget in milliseconds.")
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--imbalance", default="none",
                   choices=["none", "skew", "extreme"])
    p.add_argument("--keep-trace", action="store_true",
                   help="Don't delete the nsys .nsys-rep / .sqlite outputs.")
    args = p.parse_args()

    if shutil.which("nsys") is None:
        print("[perf_gate] nsys not found; skipping (treated as pass)")
        return 0
    if not BENCH.exists():
        print(f"[perf_gate] missing {BENCH}", file=sys.stderr)
        return 2

    sh = parse_shape(args.shape)

    tmp = Path(args.keep_trace and "/tmp/sonicmoe_ci_perf"
               or tempfile.mkdtemp(prefix="sonicmoe_ci_perf_"))
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / "trace"
    rep = out.with_suffix(".nsys-rep")
    sql = out.with_suffix(".sqlite")
    for f in (rep, sql):
        if f.exists():
            f.unlink()

    nsys_cmd = [
        "nsys", "profile", "--trace=cuda,nvtx", "--force-overwrite=true",
        "--output", str(out),
        sys.executable, str(BENCH),
        "--T", str(sh["T"]), "--E", str(sh["E"]), "--I", str(sh["I"]),
        "--topk", str(sh["K"]),
        "--warmup", str(args.warmup), "--iters", str(args.iters),
        "--imbalance", args.imbalance,
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print("[perf_gate] +", " ".join(nsys_cmd))
    rc = subprocess.call(nsys_cmd, env=env)
    if rc != 0:
        print(f"[perf_gate] nsys profile rc={rc}")
        return 1

    rc = subprocess.call(
        ["nsys", "export", "--type=sqlite",
         "--force-overwrite=true", "--output", str(sql), str(rep)]
    )
    if rc != 0:
        print(f"[perf_gate] nsys export rc={rc}")
        return 1

    extract = subprocess.run(
        [sys.executable, str(BENCH), "--extract", str(sql),
         "--iters", str(args.iters)],
        capture_output=True, text=True,
    )
    print(extract.stdout)
    if extract.returncode != 0:
        print(extract.stderr, file=sys.stderr)
        return 1

    m = re.search(r"GPU-projection:\s+([\d.]+)\s+µs/iter", extract.stdout)
    if not m:
        print("[perf_gate] could not parse µs/iter from extract output",
              file=sys.stderr)
        return 1
    us = float(m.group(1))
    budget_us = args.budget_ms * 1000.0
    pct = 100.0 * us / budget_us
    msg = (f"[perf_gate] shape={args.shape} GPU-projection={us:.1f} µs/iter "
           f"budget={budget_us:.1f} µs ({pct:.1f}% of budget)")
    print(msg)

    if not args.keep_trace:
        shutil.rmtree(tmp, ignore_errors=True)

    return 0 if us <= budget_us else 1


if __name__ == "__main__":
    sys.exit(main())
