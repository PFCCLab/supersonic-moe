#!/usr/bin/env python3
"""Sweep nsys profiles for SonicMoE FP8 frontier MlpNode (Paddle compat path).

For each shape config:
  1. Launch `bench_mlpnode_topk_nsys.py` under nsys (sqlite export).
  2. Parse GPU-projection µs/iter from BENCH NVTX range.
  3. Extract per-kernel breakdown inside BENCH range (top-N).
  4. Emit a markdown report comparing to S53 baseline.

Compare against:
  reports/session53_breakdown.md
    T=8192,E=8,I=1536,K=8 — BF16=3644 / FP8=2715 µs (pure-torch frontier).
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(os.environ.get("SONIC_MOE_REPO", Path(__file__).resolve().parents[1])).resolve()
QUACK = Path(os.environ.get("SONIC_MOE_QUACK_PATH", REPO / "third_party/quack"))
NSYS = os.environ.get("SONIC_MOE_NSYS", "nsys")
PYTHON = os.environ.get("SONIC_MOE_PADDLE_PYTHON", sys.executable)
BENCH = REPO / "tests/ops/bench_mlpnode_topk_nsys.py"
OUT_ROOT = Path(os.environ.get("SONIC_MOE_NSYS_OUTPUT_DIR", REPO / "reports" / "nsys"))


@dataclass
class Shape:
    name: str
    T: int
    E: int
    I: int = 1536
    K: int = 8
    imbalance: str = "none"
    s53_bf16: Optional[float] = None
    s53_fp8: Optional[float] = None
    notes: str = ""


# Order: baseline first (matches S53 anchor), then variants.
SHAPES: list[Shape] = [
    # Anchor: matches reports/session53_breakdown.md head-line numbers.
    Shape("baseline_T8K_E8",     8192,  8, 1536, 8, "none",    s53_bf16=3644, s53_fp8=2715, notes="S53 anchor"),
    # production-like reference EP=32 shape (HANDOFF Session 65).
    Shape("reference_T8K_E32",   8192, 32, 1536, 8, "none",                                  notes="reference EP=32"),
    # Large-N: stress Target GPU / activates many wgrad blocks (was the bug-class regime).
    Shape("large_T16K_E8",      16384,  8, 1536, 8, "none",    s53_bf16=7953, s53_fp8=5227, notes="2× tokens"),
    Shape("large_T16K_E32",     16384, 32, 1536, 8, "none",                                  notes="2× tokens, EP=32"),
    # Tiny / edge — quant overhead dominated.
    Shape("tiny_T512_E8",         512,  8, 1536, 4, "none",                                  notes="small N"),
    # Pathological hot-expert: all tokens to first K experts.
    Shape("imbal_extreme_T8K",   8192,  8, 1536, 8, "extreme", notes="all tokens to E0..K-1"),
    # Realistic skew: 80% tokens to E0.
    Shape("imbal_skew_T8K",      8192,  8, 1536, 8, "skew",    notes="80% to E0"),
    # Wide-I sanity (stronger fp8 wgrad win at I=3072).
    Shape("wide_I3072_T8K_E8",   8192,  8, 3072, 8, "none",    s53_fp8=4150, notes="I=3072"),
]


def run_nsys(shape: Shape, gpu: int, warmup: int, iters: int, out_dir: Path) -> Path:
    """Run one nsys profile. Returns sqlite path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{shape.name}_gpu{gpu}"
    sqlite_path = base.with_suffix(".sqlite")
    if sqlite_path.exists():
        sqlite_path.unlink()
    nsysrep = base.with_suffix(".nsys-rep")
    if nsysrep.exists():
        nsysrep.unlink()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{QUACK}:{REPO}:" + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["USE_QUACK_GEMM"] = "1"
    env["SONIC_MOE_FP8_MODE"] = "perf"
    env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    # Frontier knobs (epilogue maingrad add + fused swiglu quant + fp8 wgrad on by default).
    env.setdefault("SONIC_MOE_FP8_FUSED_SWIGLU_QUANT", "1")
    env.setdefault("SONIC_MOE_FP8_SAVE_Z_FP8", "1")
    env.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
    # Defensive: drop paddle distributed envs.
    for k in ("PADDLE_ELASTIC_JOB_ID", "PADDLE_TRAINER_ENDPOINTS",
              "DISTRIBUTED_TRAINER_ENDPOINTS", "FLAGS_START_PORT",
              "PADDLE_ELASTIC_TIMEOUT"):
        env.pop(k, None)

    cmd = [
        NSYS, "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--backtrace=none",
        "--resolve-symbols=false",
        "--export=sqlite",
        f"--output={base}",
        "-f", "true",
        PYTHON, str(BENCH),
        "--T", str(shape.T),
        "--E", str(shape.E),
        "--I", str(shape.I),
        "--topk", str(shape.K),
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--imbalance", shape.imbalance,
    ]
    print(f"\n[nsys] {shape.name} gpu={gpu} -> {sqlite_path}")
    print("  $ " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    log = out_dir / f"{shape.name}_gpu{gpu}.log"
    log.write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    print(f"  done in {dt:.1f}s rc={proc.returncode}; log={log}")
    if proc.returncode != 0 or not sqlite_path.exists():
        print("  STDERR (tail):")
        print("\n".join(proc.stderr.splitlines()[-30:]))
        raise RuntimeError(f"nsys run failed for {shape.name}")
    return sqlite_path


def gpu_projection_us(sqlite_path: Path, n_iters: int) -> float:
    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute("SELECT start, end FROM NVTX_EVENTS WHERE text='BENCH'").fetchall()
    if not rows:
        conn.close()
        raise RuntimeError(f"no BENCH NVTX in {sqlite_path}")
    bench_start, bench_end = rows[0]
    krows = conn.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start >= ? AND end <= ? ORDER BY start",
        (bench_start, bench_end),
    ).fetchall()
    conn.close()
    if not krows:
        raise RuntimeError(f"no kernels inside BENCH in {sqlite_path}")
    merged = []
    cs, ce = krows[0]
    for s, e in krows[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return sum(e - s for s, e in merged) / 1000.0 / n_iters  # us/iter


def kernel_breakdown(sqlite_path: Path, n_iters: int, top_n: int = 20) -> list[tuple[str, float, int]]:
    """Per-kernel µs/iter inside BENCH, summed.
    Returns list of (demangled_name_or_id, us_per_iter, count_per_iter)."""
    conn = sqlite3.connect(str(sqlite_path))
    bench_start, bench_end = conn.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text='BENCH'"
    ).fetchone()
    # Try StringIds for demangled names
    rows = conn.execute(
        """
        SELECT s.value, SUM(k.end - k.start) AS dur, COUNT(*) AS n
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON s.id = k.demangledName
        WHERE k.start >= ? AND k.end <= ?
        GROUP BY s.value
        ORDER BY dur DESC
        LIMIT ?
        """,
        (bench_start, bench_end, top_n),
    ).fetchall()
    conn.close()
    out = []
    for name, dur_ns, cnt in rows:
        out.append((name or "<unknown>", (dur_ns or 0) / 1000.0 / n_iters, cnt / n_iters))
    return out


def short_kname(s: str) -> str:
    """Best-effort short kernel name for a row in the breakdown."""
    if not s:
        return "<unknown>"
    # CUTLASS / cute templates: keep up to first '<'
    s2 = re.sub(r"^void\s+", "", s)
    s2 = s2.split("(")[0]
    # If too long, keep the leaf identifier
    if len(s2) > 90:
        leaf = s2.split("::")[-1]
        s2 = (".../" + leaf)[:90]
    return s2[:90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default="2,3", help="comma-list of GPU ids")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--shapes", type=str, default="all", help="comma list of shape names, or 'all'")
    ap.add_argument("--out", type=str, default=str(OUT_ROOT / "frontier_sweep"))
    ap.add_argument("--report", type=str, default=str(REPO / "reports/frontier_perf_sweep.md"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    selected = SHAPES if args.shapes == "all" else [
        s for s in SHAPES if s.name in set(args.shapes.split(","))
    ]
    if not selected:
        print("No shapes selected", file=sys.stderr)
        sys.exit(2)

    results: dict[str, dict] = {}
    for shape in selected:
        per_gpu = {}
        for gpu in gpus:
            try:
                sqlite_path = run_nsys(shape, gpu, args.warmup, args.iters, out_dir)
                us = gpu_projection_us(sqlite_path, args.iters)
                bd = kernel_breakdown(sqlite_path, args.iters)
                per_gpu[gpu] = {"us_per_iter": us, "sqlite": str(sqlite_path), "breakdown": bd}
                print(f"  {shape.name} gpu{gpu}: {us:.1f} µs/iter")
            except Exception as e:
                print(f"  FAILED {shape.name} gpu{gpu}: {e}", file=sys.stderr)
                per_gpu[gpu] = {"error": str(e)}
        results[shape.name] = {"shape": shape, "runs": per_gpu}

    # ---- Markdown report ----
    lines = []
    lines.append(f"# SonicMoE FP8 Frontier (MlpNode) Performance Sweep\n")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    lines.append("")
    lines.append("**Path under test**: `SonicMoEMlpNode` (Paddle compat) with FP8 frontier "
                 "(fused swiglu+quant epilogue, FP8 wgrad, TMA reduce-add wgrad accumulation).\n")
    lines.append("**Method**: nsys 2026 sqlite GPU-projection inside `BENCH` NVTX range, "
                 "merged kernel intervals / iters.\n")
    lines.append(f"Warmup={args.warmup}, iters={args.iters}, GPUs={gpus}.\n")
    lines.append("")
    lines.append("## Per-shape µs/iter (GPU-projection)\n")
    lines.append("| Shape | T | E | K | I | imbalance | "
                 + " | ".join(f"GPU{g} (µs)" for g in gpus)
                 + " | mean | CV% | S53 BF16 | S53 FP8 | Δ vs S53 FP8 | notes |")
    lines.append("|" + "|".join(["---"] * (12 + len(gpus))) + "|")

    summary_csv = []
    for name, info in results.items():
        sh: Shape = info["shape"]
        per_gpu_us = []
        per_gpu_str = []
        for g in gpus:
            r = info["runs"].get(g, {})
            if "us_per_iter" in r:
                per_gpu_us.append(r["us_per_iter"])
                per_gpu_str.append(f"{r['us_per_iter']:.1f}")
            else:
                per_gpu_str.append("FAIL")
        if per_gpu_us:
            mean = sum(per_gpu_us) / len(per_gpu_us)
            if len(per_gpu_us) > 1:
                var = sum((x - mean) ** 2 for x in per_gpu_us) / len(per_gpu_us)
                cv = (var ** 0.5) / mean * 100
            else:
                cv = 0.0
            mean_s = f"{mean:.1f}"
            cv_s = f"{cv:.2f}"
            delta = ""
            if sh.s53_fp8 is not None:
                d = (mean - sh.s53_fp8) / sh.s53_fp8 * 100
                delta = f"{d:+.1f}%"
        else:
            mean_s, cv_s, delta = "FAIL", "-", "-"
        bf16 = f"{sh.s53_bf16}" if sh.s53_bf16 else "-"
        fp8 = f"{sh.s53_fp8}" if sh.s53_fp8 else "-"
        lines.append("| " + " | ".join([
            sh.name, str(sh.T), str(sh.E), str(sh.K), str(sh.I), sh.imbalance,
            *per_gpu_str, mean_s, cv_s, bf16, fp8, delta, sh.notes
        ]) + " |")
        summary_csv.append((sh.name, mean_s, cv_s))

    lines.append("")
    lines.append("## Per-kernel breakdown (top 12 inside BENCH, GPU0 of run, µs/iter)\n")
    for name, info in results.items():
        sh = info["shape"]
        # Use first gpu's breakdown
        gpu0 = gpus[0]
        r = info["runs"].get(gpu0, {})
        bd = r.get("breakdown") or []
        lines.append(f"### {sh.name} (T={sh.T} E={sh.E} K={sh.K} I={sh.I} imbalance={sh.imbalance})\n")
        if not bd:
            lines.append("_no breakdown (run failed)_\n")
            continue
        lines.append("| kernel | µs/iter | calls/iter |")
        lines.append("|---|---:|---:|")
        total_top = 0.0
        for kname, us, cnt in bd[:12]:
            total_top += us
            lines.append(f"| `{short_kname(kname)}` | {us:.2f} | {cnt:.1f} |")
        lines.append(f"\n_top-12 sum ≈ {total_top:.1f} µs/iter_\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"\nReport: {report_path}")
    for name, mean, cv in summary_csv:
        print(f"  {name:30s} {mean:>8s} µs/iter  CV={cv}")


if __name__ == "__main__":
    main()
