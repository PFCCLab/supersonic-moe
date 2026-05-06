#!/usr/bin/env python3
"""S79 MFU sweep — drive bench_mlpnode_topk_nsys for many shapes, compute MFU.

Hardware: NVIDIA B30Z (Blackwell GB300-class, sm_103). FP8 dense tensor-core peak
= 4500 TFLOPS (per reports/cross_framework_report.md, matches B200/B300 spec).

Per shape we measure GPU-projection µs/iter (sum of merged kernel-busy intervals
inside the NVTX `BENCH` range — i.e. what an asynchronous training loop with
multi-microbatch + multi-layer pipelining converges to once Python-side launch
overhead is fully hidden).

MFU formula (single MoE-MLP layer, fwd+bwd, topk dispatch):

    matmul_FLOPs / iter = 18 × TK × H × I            (TK = T × topk)
                        = 6 × TK × H × I  (fwd: up + down)
                        + 12 × TK × H × I (bwd: 2× dgrad + 2× wgrad)

    MFU = matmul_FLOPs / (busy_time_s × 4.5e15)

This is the standard "Model FLOPs Utilisation" — counts only useful matmul work
in the numerator and treats the GPU's busy time at peak as the denominator.
Routing/quant/scatter kernels run inside `busy_time` but produce no useful FLOPs;
that drag is exactly what MFU exposes.

Usage:
    python tools/mfu_sweep_s79.py                         # full sweep
    python tools/mfu_sweep_s79.py --shapes T8192-H3072-I1536-E8-K8 ...   # subset

Outputs:
    reports/mfu_s79/sweep.json     - raw measurements
    reports/mfu_s79/sweep.csv      - same as table
    reports/mfu_s79/mfu_*.png      - seaborn figures
    reports/mfu_s79/README.md      - methodology + headline numbers
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "tests" / "ops" / "bench_mlpnode_topk_nsys.py"
OUT = REPO / "reports" / "mfu_s79"

PEAK_TFLOPS_FP8 = 4500.0          # B30Z dense tensor-core FP8 peak (TFLOP/s)
PEAK_FLOPS_FP8 = PEAK_TFLOPS_FP8 * 1e12

SHAPE_RE = re.compile(r"T(?P<T>\d+)-H(?P<H>\d+)-I(?P<I>\d+)-E(?P<E>\d+)-K(?P<K>\d+)$")


@dataclasses.dataclass
class ShapeSpec:
    T: int
    H: int
    I: int
    E: int
    K: int

    @property
    def label(self) -> str:
        return f"T{self.T}-H{self.H}-I{self.I}-E{self.E}-K{self.K}"

    @property
    def TK(self) -> int:
        return self.T * self.K

    @property
    def matmul_flops_per_iter(self) -> float:
        # fwd: up = 2 * TK * H * 2I = 4 TK H I; down = 2 * TK * I * H = 2 TK H I
        # bwd: dgrad up + dgrad down + wgrad up + wgrad down = 4 + 2 + 4 + 2 = 12 TK H I
        return 18.0 * self.TK * self.H * self.I


def parse_shape(s: str) -> ShapeSpec:
    m = SHAPE_RE.match(s)
    if not m:
        raise SystemExit(f"bad shape '{s}', expected T<...>-H<...>-I<...>-E<...>-K<...>")
    return ShapeSpec(**{k: int(v) for k, v in m.groupdict().items()})


# Default shape sweep — Ernie + larger model dimensions.
DEFAULT_SHAPES = [
    # Ernie-shape: vary token count (sequence/microbatch scaling)
    "T1024-H3072-I1536-E8-K8",
    "T2048-H3072-I1536-E8-K8",
    "T4096-H3072-I1536-E8-K8",
    "T8192-H3072-I1536-E8-K8",   # canonical Ernie production
    "T16384-H3072-I1536-E8-K8",
    # Vary expert count at fixed T (router-overhead scaling)
    "T8192-H3072-I1536-E16-K8",
    "T8192-H3072-I1536-E32-K8",
    # Wider model (more matmul work per token)
    "T4096-H4096-I2048-E8-K8",
    "T8192-H4096-I2048-E8-K8",
    "T8192-H4096-I4096-E8-K8",
    # Very large (stress)
    "T8192-H6144-I2048-E8-K8",
]


def gpu_projection_us(sqlite_path: Path, n_iters: int) -> float:
    conn = sqlite3.connect(str(sqlite_path))
    nvtx = conn.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text = 'BENCH'"
    ).fetchall()
    if not nvtx:
        raise RuntimeError(f"no BENCH NVTX range in {sqlite_path}")
    bench_start, bench_end = nvtx[0]
    rows = conn.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start >= ? AND end <= ? ORDER BY start",
        (bench_start, bench_end),
    ).fetchall()
    conn.close()
    if not rows:
        raise RuntimeError(f"no kernels in BENCH range of {sqlite_path}")
    merged = []
    cs, ce = rows[0]
    for s, e in rows[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    busy_ns = sum(e - s for s, e in merged)
    span_ns = bench_end - bench_start
    return busy_ns / 1000.0 / n_iters, span_ns / 1000.0 / n_iters


def run_one(shape: ShapeSpec, iters: int, warmup: int, outdir: Path) -> dict:
    """Profile one shape. Returns measurement dict."""
    if shutil.which("nsys") is None:
        raise SystemExit("nsys not found in PATH")

    tag = shape.label
    out_base = outdir / f"trace_{tag}"
    rep = out_base.with_suffix(".nsys-rep")
    sql = out_base.with_suffix(".sqlite")
    log = outdir / f"bench_{tag}.log"

    for p in (rep, sql):
        if p.exists():
            p.unlink()

    env = os.environ.copy()
    env["USE_QUACK_GEMM"] = "1"
    env["SONIC_MOE_FP8_MODE"] = "perf"
    env["SONIC_MOE_FP8_WGRAD"] = "1"
    env["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
    env["TRITON_PTXAS_PATH"] = env.get("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    nsys_cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "--sample=none", "--backtrace=none",
        "--resolve-symbols=false",
        "--force-overwrite=true",
        "--output", str(out_base),
        sys.executable, str(BENCH),
        "--T", str(shape.T),
        "--H", str(shape.H),
        "--I", str(shape.I),
        "--E", str(shape.E),
        "--topk", str(shape.K),
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--imbalance", "none",
    ]
    print(f"\n[mfu_sweep] {tag}  -> {' '.join(shlex.quote(c) for c in nsys_cmd[:5])} ...")
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(nsys_cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    print(f"  nsys rc={rc} wall={wall:.1f}s")
    if rc != 0 or not rep.exists():
        return {
            "shape": tag, "T": shape.T, "H": shape.H, "I": shape.I,
            "E": shape.E, "K": shape.K, "rc": rc, "wall_s": wall,
            "busy_us_per_iter": None, "span_us_per_iter": None,
            "matmul_flops_per_iter": shape.matmul_flops_per_iter,
            "mfu": None, "sm_util_busy_over_span": None,
            "log": str(log),
        }

    # Export sqlite
    rc2 = subprocess.call([
        "nsys", "export", "--type=sqlite",
        "--force-overwrite", "true",
        "--output", str(sql), str(rep)
    ], env=env)
    if rc2 != 0 or not sql.exists():
        return {
            "shape": tag, "T": shape.T, "H": shape.H, "I": shape.I,
            "E": shape.E, "K": shape.K, "rc": rc, "wall_s": wall,
            "busy_us_per_iter": None, "span_us_per_iter": None,
            "matmul_flops_per_iter": shape.matmul_flops_per_iter,
            "mfu": None, "sm_util_busy_over_span": None,
            "log": str(log), "export_rc": rc2,
        }

    busy_us, span_us = gpu_projection_us(sql, iters)
    busy_s = busy_us * 1e-6
    mfu = shape.matmul_flops_per_iter / (busy_s * PEAK_FLOPS_FP8)
    sm_util = busy_us / span_us if span_us > 0 else None
    print(f"  busy={busy_us:.1f} µs/iter  span={span_us:.1f} µs/iter  "
          f"matmul-FLOPs={shape.matmul_flops_per_iter/1e9:.1f}G  "
          f"MFU={mfu*100:.2f}%  SM-util={sm_util*100:.1f}%")
    return {
        "shape": tag, "T": shape.T, "H": shape.H, "I": shape.I,
        "E": shape.E, "K": shape.K, "rc": rc, "wall_s": wall,
        "busy_us_per_iter": busy_us, "span_us_per_iter": span_us,
        "matmul_flops_per_iter": shape.matmul_flops_per_iter,
        "matmul_tflops_achieved": shape.matmul_flops_per_iter / busy_s / 1e12,
        "mfu": mfu,
        "sm_util_busy_over_span": sm_util,
        "log": str(log),
    }


def render_plots(rows: list[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame([r for r in rows if r.get("mfu") is not None])
    if df.empty:
        print("[mfu_sweep] no successful rows; skip plots")
        return

    df["MFU_%"] = df["mfu"] * 100.0
    df["TFLOPS"] = df["matmul_tflops_achieved"]
    df["TK"] = df["T"] * df["K"]
    df["model_size_HxI"] = df.apply(lambda r: f"H={r['H']} I={r['I']}", axis=1)

    sns.set_theme(style="whitegrid", context="talk", palette="deep")

    # 1. MFU vs T at Ernie shape, varying E
    ernie = df[(df.H == 3072) & (df.I == 1536)].sort_values(["E", "T"])
    if not ernie.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=ernie, x="T", y="MFU_%", hue="E", marker="o",
                     palette="viridis", ax=ax, linewidth=2.2, markersize=10)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Tokens per layer per microbatch (T)")
        ax.set_ylabel("MFU (%)")
        ax.set_title(
            "FP8 Frontier MFU vs Token Count — Ernie shape (H=3072, I=1536, K=8)\n"
            "B30Z dense FP8 peak 4500 TFLOPS · busy-time projection (async-overlap, no-bubble)"
        )
        ax.set_ylim(0, max(50, ernie["MFU_%"].max() * 1.15))
        for _, row in ernie.iterrows():
            ax.annotate(f"{row['MFU_%']:.1f}%",
                        xy=(row['T'], row["MFU_%"]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / "mfu_vs_T_ernie.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {outdir/'mfu_vs_T_ernie.png'}")

    # 2. MFU vs model width (H,I) at T=8192
    big = df[df.T == 8192].copy().sort_values(["H", "I"])
    if not big.empty:
        big["shape_label"] = big.apply(lambda r: f"H{r['H']}/I{r['I']}/E{r['E']}", axis=1)
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=big, x="shape_label", y="MFU_%", hue="E",
                    palette="rocket", ax=ax)
        ax.set_xlabel("Model width (per-layer)")
        ax.set_ylabel("MFU (%)")
        ax.set_title(
            "FP8 Frontier MFU vs Model Width @ T=8192 — single B30Z, FP8 peak 4500 TFLOPS"
        )
        for p in ax.patches:
            v = p.get_height()
            if v > 0:
                ax.annotate(f"{v:.1f}%", (p.get_x() + p.get_width()/2, v),
                            ha="center", va="bottom", fontsize=10)
        plt.xticks(rotation=15)
        fig.tight_layout()
        fig.savefig(outdir / "mfu_vs_width_T8192.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {outdir/'mfu_vs_width_T8192.png'}")

    # 3. Busy µs/iter vs matmul FLOPs (roofline-ish)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="matmul_flops_per_iter", y="busy_us_per_iter",
                    hue="E", size="T", sizes=(60, 350), palette="mako", ax=ax)
    # ideal (peak) line: t_us = flops / peak_flops * 1e6
    import numpy as np
    fmax = df.matmul_flops_per_iter.max() * 1.2
    fmin = df.matmul_flops_per_iter.min() * 0.8
    xs = np.geomspace(fmin, fmax, 50)
    ax.plot(xs, xs / PEAK_FLOPS_FP8 * 1e6, "--", color="crimson", lw=2,
            label=f"Ideal @ {PEAK_TFLOPS_FP8:.0f} TFLOPS")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Matmul FLOPs / iter (fwd+bwd)")
    ax.set_ylabel("GPU-projection busy time / iter (µs)")
    ax.set_title("Roofline view — distance to ideal line ⇔ (1/MFU)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(outdir / "roofline.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {outdir/'roofline.png'}")

    # 4. SM utilisation (busy/span) heatmap
    if {"H", "I", "E", "T"}.issubset(df.columns) and len(df) > 4:
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.scatterplot(data=df, x="MFU_%", y=df["sm_util_busy_over_span"]*100,
                        hue="E", size="TK", sizes=(80, 400), palette="flare", ax=ax)
        for _, r in df.iterrows():
            ax.annotate(r['shape'], (r["MFU_%"], r["sm_util_busy_over_span"]*100),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("MFU (%) — useful matmul / peak")
        ax.set_ylabel("SM-busy / span (%) — async-overlap headroom")
        ax.set_title("MFU vs Single-iter SM utilisation\n"
                     "(High span/busy ratio ⇒ launch-overhead gap → reclaimable by multi-microbatch async pipelining)")
        fig.tight_layout()
        fig.savefig(outdir / "mfu_vs_smutil.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {outdir/'mfu_vs_smutil.png'}")


def write_readme(rows: list[dict], outdir: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sweep.csv", index=False)
    with open(outdir / "sweep.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    ok = df[df.mfu.notnull()].copy()
    ok["MFU_%"] = ok.mfu * 100
    ok["TFLOPS_achieved"] = ok.matmul_tflops_achieved
    ok = ok[["shape", "T", "H", "I", "E", "K", "busy_us_per_iter",
             "span_us_per_iter", "TFLOPS_achieved", "MFU_%"]]

    md = []
    md.append("# FP8 Frontier MFU Sweep — S79\n")
    md.append("> Generated by `tools/mfu_sweep_s79.py`. Hardware: NVIDIA B30Z "
              "(Blackwell GB300-class, sm_103). FP8 dense tensor-core peak = "
              f"**{PEAK_TFLOPS_FP8:.0f} TFLOPS**.\n")
    md.append("## Methodology (rigorous)\n")
    md.append(
        "1. Each shape profiled with `nsys profile --trace=cuda,nvtx --sample=none "
        "--backtrace=none --resolve-symbols=false` (no symbol DL — see env.md).\n"
        "2. nsys sqlite parsed: kernels strictly inside the NVTX `BENCH` range; "
        "overlapping intervals merged → **GPU-projection busy time / iter**. This is "
        "what an async multi-microbatch + multi-layer training loop converges to once "
        "Python-side launch overhead is fully hidden by overlap (no bubbles).\n"
        "3. Useful work / iter = single MoE-MLP-layer matmul-FLOPs:\n\n"
        "    matmul_FLOPs = 18 × TK × H × I,  TK = T × topk\n\n"
        "    breakdown:\n"
        "    - fwd up-proj   2·TK·H·(2I) = 4 TK H I\n"
        "    - fwd down-proj 2·TK·I·H    = 2 TK H I\n"
        "    - bwd dgrad up  4 TK H I    | dgrad down 2 TK H I\n"
        "    - bwd wgrad up  4 TK H I    | wgrad down 2 TK H I\n\n"
        "4. **MFU** = matmul_FLOPs / (busy_time × 4.5 × 10¹⁵). This is the standard "
        "Megatron-style MFU (only matmul work in the numerator); routing/quant/scatter "
        "kernels run inside `busy_time` but produce no useful FLOPs — that drag is "
        "exactly what MFU exposes.\n"
        "5. Multi-layer / multi-microbatch invariance: both numerator and denominator "
        "scale linearly with L · MB. Therefore the per-layer MFU measured here IS the "
        "training-loop MFU (under the no-bubble assumption ②).\n\n"
    )
    md.append("## Results\n\n")
    md.append("| shape | T | H | I | E | K | busy µs/iter | span µs/iter | TFLOPS achieved | MFU |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in ok.iterrows():
        md.append(f"| {r['shape']} | {r['T']} | {r['H']} | {r['I']} | {r['E']} | {r['K']} | "
                  f"{r['busy_us_per_iter']:.1f} | {r['span_us_per_iter']:.1f} | "
                  f"{r['TFLOPS_achieved']:.0f} | **{r['MFU_%']:.2f}%** |\n")
    if len(ok):
        md.append(f"\n**Headline:** best MFU = {ok['MFU_%'].max():.2f}% "
                  f"({ok.loc[ok['MFU_%'].idxmax(),'shape']}); "
                  f"worst MFU = {ok['MFU_%'].min():.2f}% "
                  f"({ok.loc[ok['MFU_%'].idxmin(),'shape']}).\n")
    md.append("\n## Figures\n")
    md.append("- `mfu_vs_T_ernie.png` — MFU scaling vs token count, Ernie shape, hue=E.\n")
    md.append("- `mfu_vs_width_T8192.png` — MFU vs model width at T=8192.\n")
    md.append("- `roofline.png` — log-log busy-time vs matmul FLOPs against the FP8 ideal line.\n")
    md.append("- `mfu_vs_smutil.png` — MFU vs single-iter SM utilisation (per-shape headroom for async overlap).\n")
    (outdir / "README.md").write_text("".join(md))
    print(f"  wrote {outdir/'README.md'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", default=None,
                   help="Shape labels (e.g. T8192-H3072-I1536-E8-K8). Default: full sweep.")
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--outdir", default=str(OUT))
    p.add_argument("--skip-bench", action="store_true",
                   help="Re-use existing sqlite/nsys-rep in --outdir, only re-render plots/README.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shape_strs = args.shapes if args.shapes else DEFAULT_SHAPES
    shapes = [parse_shape(s) for s in shape_strs]

    rows: list[dict] = []
    if args.skip_bench:
        for shape in shapes:
            sql = outdir / f"trace_{shape.label}.sqlite"
            if not sql.exists():
                print(f"[mfu_sweep] {shape.label}: no sqlite, skip")
                continue
            try:
                busy_us, span_us = gpu_projection_us(sql, args.iters)
                busy_s = busy_us * 1e-6
                mfu = shape.matmul_flops_per_iter / (busy_s * PEAK_FLOPS_FP8)
                rows.append({
                    "shape": shape.label, "T": shape.T, "H": shape.H, "I": shape.I,
                    "E": shape.E, "K": shape.K, "rc": 0,
                    "busy_us_per_iter": busy_us, "span_us_per_iter": span_us,
                    "matmul_flops_per_iter": shape.matmul_flops_per_iter,
                    "matmul_tflops_achieved": shape.matmul_flops_per_iter / busy_s / 1e12,
                    "mfu": mfu,
                    "sm_util_busy_over_span": busy_us / span_us if span_us else None,
                    "log": "<reused>",
                })
            except Exception as e:
                print(f"[mfu_sweep] {shape.label}: parse err {e}")
    else:
        for shape in shapes:
            try:
                rows.append(run_one(shape, args.iters, args.warmup, outdir))
            except Exception as e:
                print(f"[mfu_sweep] {shape.label} FAILED: {e}")
                rows.append({"shape": shape.label, "error": str(e)})

    write_readme(rows, outdir)
    render_plots(rows, outdir)

    print("\n=== summary ===")
    for r in rows:
        if r.get("mfu") is not None:
            print(f"  {r['shape']:<35s} busy={r['busy_us_per_iter']:7.1f} µs/iter  "
                  f"MFU={r['mfu']*100:5.2f}%  TFLOPS={r['matmul_tflops_achieved']:6.0f}")
        else:
            print(f"  {r['shape']:<35s} <fail>")


if __name__ == "__main__":
    main()
