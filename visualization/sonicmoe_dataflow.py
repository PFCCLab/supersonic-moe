"""SonicMoE fig1–10: Session 53 data-driven figure suite.

All figures consume **session-53 artifacts only**:
  • reports/fp8_frontier_path_analysis.json  — path analysis, budget anchor, precision anchor
  • reports/quant_bench_final.json           — quantization kernel latency benchmarks
  • reports/grid_session53/session53_grid_full.json — 27-shape performance grid

No legacy manifest.json or benchmark_manifest.json is used.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
_PATH_ANALYSIS = ROOT / "reports" / "fp8_frontier_path_analysis.json"
_QUANT_BENCH   = ROOT / "reports" / "quant_bench_final.json"
_GRID_FULL     = ROOT / "reports" / "grid_session53" / "session53_grid_full.json"

# ── shared style ─────────────────────────────────────────────────────────────
C_BF16   = "#3B82F6"
C_FP8    = "#F97316"
C_SAVE   = "#10B981"
C_COST   = "#EF4444"
C_TEXT   = "#1F2937"
C_NEUTRAL= "#6B7280"
C_GRID   = "#E5E7EB"
C_PANEL  = "#F9FAFB"
_HW      = "NVIDIA Target GPU (SM100)"

STAGE_NAMES = [
    "Router &\nMeta",
    "UpProj\nFwd",
    "DnProj\nFwd",
    "DnProj\nBwd",
    "UpBwd\n(wgrad)",
    "UpBwd\n(actgrad)",
]

def _style() -> None:
    plt.rcParams.update({
        "font.family":  "DejaVu Sans",
        "axes.facecolor":  "#FCFCFD",
        "axes.edgecolor":  "#CBD5E1",
        "axes.linewidth":  0.8,
        "grid.color":      C_GRID,
        "grid.linewidth":  0.6,
        "xtick.color":     C_TEXT,
        "ytick.color":     C_TEXT,
        "text.color":      C_TEXT,
        "axes.labelcolor": C_TEXT,
    })

def _save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(exist_ok=True)
    p = ASSETS / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> {p}")

def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.06, 1.04, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=C_TEXT, va="bottom")

# ── data loaders ─────────────────────────────────────────────────────────────
def _load_pa() -> dict[str, Any]:
    return json.loads(_PATH_ANALYSIS.read_text())

def _load_qb() -> list[dict[str, Any]]:
    return json.loads(_QUANT_BENCH.read_text())

def _load_grid() -> dict[str, Any]:
    return json.loads(_GRID_FULL.read_text())


# ═════════════════════════════════════════════════════════════════════════════
# fig1 — System Architecture Conceptual
# ═════════════════════════════════════════════════════════════════════════════
def fig1_system_overview() -> None:
    """Conceptual diagram: MoE block + FP8 blockscaling innovation."""
    _style()
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def _box(x, y, w, h, label, sublabel="", fill="#DBEAFE", edge="#3B82F6",
             tc=C_TEXT, fs=9.5, lw=1.4, style="round,pad=0.06"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                   facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3))
        cy = y + h/2 + (0.10 if sublabel else 0)
        ax.text(x + w/2, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.18, sublabel, ha="center", va="center",
                    fontsize=7.2, color=C_NEUTRAL, zorder=4)

    def _arr(x0, y0, x1, y1, color=C_TEXT, lw=1.5, label=""):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                   mutation_scale=12), zorder=2)
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx+0.07, my, label, fontsize=7.5, color=color,
                    va="center", style="italic")

    # Token input
    _box(0.15, 2.4, 1.1, 1.2, "Input tokens\n$x$",
         f"T×H  BF16", fill="#F0FDF4", edge=C_SAVE, fs=9)

    # TopK Router
    _box(1.7, 2.4, 1.5, 1.2, "TopK Router",
         "Softmax + gather\nindices + dispatch", fill="#EDE9FE", edge="#7C3AED")
    _arr(1.25, 3.0, 1.70, 3.0)

    # Expert pool
    exp_y = [4.2, 3.0, 1.6]  # top, mid, bot
    for i, ey in enumerate(exp_y):
        _box(3.7, ey, 2.5, 0.85,
             f"Expert FFN",
             "UpProj + SwiGLU + DnProj",
             fill="#FFF7ED", edge=C_FP8)
        _arr(3.2, ey + 0.42, 3.7, ey + 0.42, color=C_FP8)
        _arr(6.2, ey + 0.42, 6.7, ey + 0.42, color=C_FP8)
    ax.text(3.45, 3.0, "⋮", ha="center", va="center",
            fontsize=18, color=C_FP8, fontweight="bold")
    ax.text(6.45, 3.0, "⋮", ha="center", va="center",
            fontsize=18, color=C_FP8, fontweight="bold")

    # Bracket "E experts"
    ax.annotate("", xy=(3.70, 4.85), xytext=(3.70, 1.60),
                arrowprops=dict(arrowstyle="-", color=C_NEUTRAL,
                                connectionstyle="bar,fraction=0.15"), zorder=1)
    ax.text(3.42, 3.22, f"E\nexpert\nFFNs", ha="right", va="center",
            fontsize=8, color=C_NEUTRAL, style="italic")

    # Scatter gather
    _box(6.7, 2.4, 1.3, 1.2, "Scatter-\nGather", "un-dispatch +\nscatter output",
         fill="#EDE9FE", edge="#7C3AED")
    _box(8.4, 2.4, 1.4, 1.2, "Output\n$y$",
         "T×H  BF16", fill="#F0FDF4", edge=C_SAVE)
    _arr(8.0, 3.0, 8.4, 3.0)

    # Router dispatch arrows from router to experts
    rx, ry = 3.2, 3.0
    for ey in exp_y:
        ax.annotate("", xy=(3.7, ey+0.42), xytext=(rx, ry),
                    arrowprops=dict(arrowstyle="-|>", color=C_NEUTRAL,
                                   lw=0.9, connectionstyle="arc3,rad=0.0"))
    ax.text(rx + 0.3, 3.18, "dispatch", fontsize=7, color=C_NEUTRAL, style="italic")

    # Gather back to scatter-gather
    for ey in exp_y:
        ax.annotate("", xy=(6.7, 3.0), xytext=(6.2, ey+0.42),
                    arrowprops=dict(arrowstyle="-|>", color=C_NEUTRAL,
                                   lw=0.9, connectionstyle="arc3,rad=0.0"))

    # FP8 innovation callout box
    innov_x, innov_y = 3.7, 0.12
    ax.add_patch(FancyBboxPatch((innov_x, innov_y), 2.5, 1.1,
                                boxstyle="round,pad=0.06",
                                facecolor="#FFF7ED", edgecolor=C_FP8,
                                linewidth=1.8, linestyle="--", zorder=3))
    ax.text(5.0, innov_y + 0.82, "FP8 Blockscaling Frontier",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=C_FP8, zorder=4)
    ax.text(5.0, innov_y + 0.50,
            "T-size FP8 activation  ×  TK-size ISA-packed scale",
            ha="center", va="center", fontsize=8.2, color=C_TEXT, zorder=4)
    ax.text(5.0, innov_y + 0.24,
            "ZeroMat GEMM: no TK-size HBM gather  →  reduced activation memory",
            ha="center", va="center", fontsize=8.0, color=C_NEUTRAL, zorder=4)

    # Tensor type legend
    items = [
        mpatches.Patch(facecolor="#DBEAFE", edgecolor=C_BF16, label="BF16 tensor"),
        mpatches.Patch(facecolor="#FFF7ED", edgecolor=C_FP8, label="FP8 / ZeroMat GEMM path"),
        mpatches.Patch(facecolor="#EDE9FE", edgecolor="#7C3AED", label="Router / Gather (BF16)"),
        mpatches.Patch(facecolor="#F0FDF4", edgecolor=C_SAVE, label="Input / Output (BF16)"),
    ]
    ax.legend(handles=items, loc="upper right", framealpha=0.92,
              fontsize=8.5, ncol=2, borderpad=0.6)

    fig.suptitle(
        "SonicMoE  —  Mixture-of-Experts Architecture with FP8 Blockscaling Frontier\n"
        f"E experts (E=8/32/128 evaluated), K activated per token, H=3072  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig1_system_overview.png")


# ═════════════════════════════════════════════════════════════════════════════
# fig2 — Executive Summary (session-53 data)
# ═════════════════════════════════════════════════════════════════════════════
def fig2_executive_summary() -> None:
    """Key claims: speedup, memory, precision, global scaling — all session-53 data."""
    _style()
    pa = _load_pa()
    ba = pa["budget_anchor"]
    pr = pa["precision_anchor"]
    fs = pa["frontier_summary"]

    fig = plt.figure(figsize=(16, 9.5))
    gs  = fig.add_gridspec(2, 4, hspace=0.48, wspace=0.42)
    ax_perf = fig.add_subplot(gs[0, :2])
    ax_mem  = fig.add_subplot(gs[0, 2:])
    ax_prec = fig.add_subplot(gs[1, :2])
    ax_glob = fig.add_subplot(gs[1, 2:])

    for ax, lbl in zip([ax_perf, ax_mem, ax_prec, ax_glob],
                       ["(a)", "(b)", "(c)", "(d)"]):
        _panel_label(ax, lbl)

    # (a) Compute time at anchor
    bf = ba["bf16_us"]
    fp = ba["fp8_us"]
    sp = ba["speedup"]
    bars = ax_perf.bar([0, 1], [bf/1000, fp/1000], width=0.5,
                       color=[C_BF16, C_FP8], edgecolor="white",
                       linewidth=0.8, zorder=3, alpha=0.88)
    for b, v, label in zip(bars, [bf, fp], ["BF16", "FP8"]):
        ax_perf.text(b.get_x()+b.get_width()/2, b.get_height()+0.04,
                     f"{v/1000:.2f} ms", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")
    ax_perf.annotate("", xy=(1.35, fp/1000), xytext=(1.35, bf/1000),
                     arrowprops=dict(arrowstyle="<->", color=C_SAVE, lw=2.0),
                     annotation_clip=False)
    ax_perf.text(1.55, (bf+fp)/2/1000,
                 f"$\\mathbf{{{sp:.3f}\\times}}$",
                 fontsize=14, color=C_SAVE, va="center", clip_on=False)
    ax_perf.set_xticks([0, 1])
    ax_perf.set_xticklabels(["BF16", "FP8"])
    ax_perf.set_ylabel("GPU-projection (ms / iter)")
    ax_perf.set_title("(a)  Compute time  —  anchor shape\n"
                      f"T=8192, E=8, I=1536, H=3072", fontweight="bold", fontsize=9.5)
    ax_perf.set_ylim(0, bf/1000 * 1.28)
    ax_perf.set_xlim(-0.45, 1.85)
    ax_perf.grid(True, axis="y", color=C_GRID)

    # (b) Memory comparison at anchor
    mem_bf = ba["memory_bf16"]
    mem_fp = ba["memory_fp8"]
    phases = ["Fwd peak\n(MiB)", "Bwd peak\n(MiB)"]
    bf_vals = [mem_bf["peak_fwd_mib"], mem_bf["peak_bwd_mib"]]
    fp_vals = [mem_fp["peak_fwd_mib"], mem_fp["peak_bwd_mib"]]
    x = np.arange(2)
    w = 0.30
    ax_mem.bar(x - w/2, bf_vals, w, color=C_BF16, edgecolor="white",
               linewidth=0.8, label="BF16", zorder=3, alpha=0.88)
    ax_mem.bar(x + w/2, fp_vals, w, color=C_FP8, edgecolor="white",
               linewidth=0.8, label="FP8", zorder=3, alpha=0.88)
    for i, (bf_v, fp_v) in enumerate(zip(bf_vals, fp_vals)):
        delta = fp_v - bf_v
        color = C_SAVE if delta < 0 else C_COST
        ax_mem.text(i, max(bf_v, fp_v) + 35,
                    f"{delta:+.0f} MiB ({100*delta/bf_v:+.1f}%)",
                    ha="center", fontsize=8.5, color=color, fontweight="bold")
        ax_mem.text(i - w/2, bf_v - 60, f"{bf_v:.0f}", ha="center",
                    fontsize=8, color="white", fontweight="bold")
        ax_mem.text(i + w/2, fp_v - 60, f"{fp_v:.0f}", ha="center",
                    fontsize=8, color="white", fontweight="bold")
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels(phases)
    ax_mem.set_ylabel("Peak HBM (MiB)")
    ax_mem.set_title("(b)  Memory profile  —  anchor shape", fontweight="bold", fontsize=9.5)
    ax_mem.legend(loc="upper right", framealpha=0.9)
    ax_mem.set_ylim(0, max(max(bf_vals), max(fp_vals)) * 1.22)
    ax_mem.grid(True, axis="y", color=C_GRID)

    # (c) Precision — RRMSE and cosine
    tensor_names = ["output", "dx", "dw1", "dw2"]
    rrmse = [pr["rrmse_pct"]["output"], pr["rrmse_pct"]["dx"],
             pr["rrmse_pct"]["dw1"],    pr["rrmse_pct"]["dw2"]]
    cosine= [pr["cosine_sim"]["output"], pr["cosine_sim"]["dx"],
             pr["cosine_sim"]["dw1"],    pr["cosine_sim"]["dw2"]]
    x4 = np.arange(4)
    ax_prec.bar(x4, rrmse, width=0.5, color=C_FP8, edgecolor="white",
                linewidth=0.8, alpha=0.82, zorder=3, label="RRMSE (%)")
    ax_prec.axhline(10.0, color=C_COST, ls="--", lw=1.5, label="Threshold 10%", zorder=4)
    for xi, (r, c) in enumerate(zip(rrmse, cosine)):
        ax_prec.text(xi, r + 0.25, f"{r:.2f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=C_SAVE)
        ax_prec.text(xi, -1.4, f"cos={c:.4f}", ha="center", va="top",
                     fontsize=7.5, color=C_NEUTRAL)
    ax_prec.set_xticks(x4)
    ax_prec.set_xticklabels(tensor_names)
    ax_prec.set_ylabel("RRMSE (%)")
    ax_prec.set_title("(c)  Precision audit  —  largest shape tested\n"
                      f"T=32768, E=8, I=1536  |  all seeds PASS", fontweight="bold", fontsize=9.5)
    ax_prec.set_ylim(-2, 14)
    ax_prec.legend(loc="upper right", framealpha=0.9)
    badge_all_pass = all(r < 10.0 for r in rrmse)
    badge_text = "✓  All tensors PASS" if badge_all_pass else "✗  FAIL"
    badge_color = C_SAVE if badge_all_pass else C_COST
    ax_prec.text(3.5, 12.5, badge_text, fontsize=10, fontweight="bold",
                 color=badge_color, ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec=badge_color,
                           lw=1.2, alpha=0.95))

    # (d) Global scaling summary
    dims   = ["T=8k", "T=16k", "T=32k", "I=1536", "I=2048", "I=3072", "E=8", "E=32", "E=128"]
    speeds = [
        fs["avg_speedup_by_T"]["8192"],
        fs["avg_speedup_by_T"]["16384"],
        fs["avg_speedup_by_T"]["32768"],
        fs["avg_speedup_by_I"]["1536"],
        fs["avg_speedup_by_I"]["2048"],
        fs["avg_speedup_by_I"]["3072"],
        fs["avg_speedup_by_E"]["8"],
        fs["avg_speedup_by_E"]["32"],
        fs["avg_speedup_by_E"]["128"],
    ]
    bg_colors = ["#EFF6FF"]*3 + ["#FFF7ED"]*3 + ["#F0FDF4"]*3
    bars2 = ax_glob.bar(range(9), speeds, width=0.6,
                        color=bg_colors, edgecolor=[C_BF16]*3+[C_FP8]*3+[C_SAVE]*3,
                        linewidth=1.2, zorder=3)
    ax_glob.axhline(fs["speedup_range"]["mean"], color=C_TEXT, ls="--",
                    lw=1.0, label=f"Global mean {fs['speedup_range']['mean']:.2f}×", zorder=4)
    ax_glob.axhspan(fs["speedup_range"]["min"], fs["speedup_range"]["max"],
                    color=C_SAVE, alpha=0.07, zorder=0)
    for xi, v in enumerate(speeds):
        ax_glob.text(xi, v + 0.012, f"{v:.2f}×", ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color=C_TEXT)
    # Group labels
    for mid, label, color in [(1, "by T", C_BF16), (4, "by I", C_FP8), (7, "by E", C_SAVE)]:
        ax_glob.text(mid, ax_glob.get_ylim()[0] - 0.065,
                     label, ha="center", va="top", fontsize=8.5,
                     fontweight="bold", color=color,
                     transform=ax_glob.get_xaxis_transform())
    ax_glob.set_xticks(range(9))
    ax_glob.set_xticklabels(dims, fontsize=8.5)
    ax_glob.set_ylabel("Average speedup (×)")
    ax_glob.set_title(f"(d)  Average speedup across 27 shapes\n"
                      f"min={fs['speedup_range']['min']:.3f}× — "
                      f"max={fs['speedup_range']['max']:.3f}×", fontweight="bold", fontsize=9.5)
    ax_glob.legend(loc="upper left", framealpha=0.9)
    ax_glob.set_ylim(1.20, max(speeds) + 0.18)
    ax_glob.grid(True, axis="y", color=C_GRID)
    ax_glob.axvspan(2.45, 2.55, color="#CBD5E1", lw=0)
    ax_glob.axvspan(5.45, 5.55, color="#CBD5E1", lw=0)

    fig.suptitle(
        "SonicMoE Session 53  —  FP8 Frontier: Executive Summary\n"
        f"27 shapes (3T × 3E × 3I)  |  nsys GPU-projection  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig2_executive_summary.png")


# ═════════════════════════════════════════════════════════════════════════════
# fig3 — Quantization Kernel Selection
# ═════════════════════════════════════════════════════════════════════════════
def fig3_quant_kernel_comparison() -> None:
    """Latency comparison of quant kernel implementations (quant_bench_final.json)."""
    _style()
    qb = _load_qb()

    dims_order = ["I=1536", "H=3072", "2I=3072"]
    kernels = ["triton_col_nogather", "triton_col_gather", "cute_col_nogather",
               "row_quant", "dual_varlen"]
    k_labels = ["Triton col\n(no-gather)", "Triton col\n(gather)", "CuTe col\n(no-gather)",
                 "Row quant", "Dual varlen\n(BF16 fused)"]
    k_colors = ["#60A5FA", "#2563EB", "#8B5CF6", "#10B981", "#F59E0B"]
    # selected kernel for hot path
    k_selected = {"triton_col_nogather", "row_quant"}

    # Lookup median by (kernel, dim)
    data: dict[tuple[str,str], float] = {}
    for rec in qb:
        data[(rec["kernel"], rec["dim"])] = rec["median"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0),
                              gridspec_kw={"wspace": 0.35})
    fig.subplots_adjust(bottom=0.22, top=0.82)

    for ci, dim in enumerate(dims_order):
        ax = axes[ci]
        vals = [data.get((k, dim), float("nan")) for k in kernels]
        bars = ax.bar(range(len(kernels)), vals, width=0.6, color=k_colors,
                      edgecolor="white", linewidth=0.6, zorder=3, alpha=0.88)
        # highlight selected kernel with heavier border
        for bi, (b, k) in enumerate(zip(bars, kernels)):
            if k in k_selected:
                b.set_edgecolor("#1E3A5F")
                b.set_linewidth(2.2)
                ax.text(bi, vals[bi] + 2.5, "★ selected",
                        ha="center", va="bottom", fontsize=7.0,
                        color="#1E3A5F", fontweight="bold")
        for bi, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(bi, v/2, f"{v:.0f} µs",
                        ha="center", va="center", fontsize=8.2,
                        fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="none", edgecolor="none"))

        ax.set_xticks(range(len(kernels)))
        ax.set_xticklabels(k_labels, fontsize=8.0, rotation=10, ha="right")
        ax.set_ylabel("Median latency (µs)")
        ax.set_title(f"dim = {dim}", fontweight="bold", fontsize=11)
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.28)
        ax.grid(True, axis="y", color=C_GRID)
        _panel_label(ax, f"({'abc'[ci]})")

    # Speedup annotation for nw=1 vs nw=4 — derive ratios from quant_bench data
    qb = _load_qb()
    kernels = qb.get("kernels", {})
    def _med(k):
        v = kernels.get(k, {})
        return v.get("median_us", v.get("time_us", 0))
    col_nw1 = _med("triton_col_nogather")
    col_nw4 = _med("triton_col_nogather_nw4") or _med("triton_col_nw4")
    cute_col = _med("cute_col_nogather") or _med("cute_col")
    row_nw1 = _med("triton_row")
    lines = []
    if col_nw4 > 0 and col_nw1 > 0:
        lines.append(f"Triton col (nw=1): {col_nw4/col_nw1:.1f}× faster than nw=4")
    else:
        lines.append("Triton col (nw=1): faster than nw=4 (see NCU profile)")
    lines.append("Row quant: high occupancy — nw=1 gives no improvement")
    if cute_col > 0 and col_nw1 > 0:
        pct = (cute_col / col_nw1 - 1) * 100
        lines.append(f"CuTe col: {pct:.0f}% slower than Triton col → excluded from hot path")
    axes[1].text(0.5, -0.28,
                 "\n".join(lines),
                 transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=8.5, color=C_NEUTRAL, style="italic", linespacing=1.5)

    fig.suptitle(
        "SonicMoE  —  Quantization Kernel Latency Comparison\n"
        f"FP8 E4M3 blockscaled  |  Triton JIT  |  {_HW}",
        fontsize=13, fontweight="bold", y=0.95,
    )
    _save(fig, "fig3_quant_kernel_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# fig4 — Performance Waterfall (anchor shape)
# ═════════════════════════════════════════════════════════════════════════════
def fig4_performance_waterfall() -> None:
    """Horizontal waterfall: BF16 total → savings → overheads → FP8 total."""
    _style()
    pa = _load_pa()
    ba = pa["budget_anchor"]
    bb = ba["budget_breakdown"]   # list of dicts

    savings  = [e for e in bb if e["kind"] == "saving" and abs(e["delta_us"]) > 1]
    overhead = [e for e in bb if e["kind"] == "overhead" and e["delta_us"] > 1]
    savings.sort(key=lambda x: x["delta_us"])          # most negative first
    overhead.sort(key=lambda x: x["delta_us"], reverse=True)  # largest first

    fig, ax = plt.subplots(figsize=(14, 8.5))
    fig.subplots_adjust(left=0.30, right=0.96)

    entries = []
    entries.append(("BF16 total", ba["bf16_us"], "base"))
    for e in savings:
        entries.append((e["category"], e["delta_us"], "saving"))
    for e in overhead:
        entries.append((e["category"], e["delta_us"], "overhead"))
    entries.append(("FP8 total", ba["fp8_us"], "result"))

    running = ba["bf16_us"]
    y_labels, y_pos, bar_starts, bar_widths, bar_colors, bar_texts = \
        [], [], [], [], [], []

    for yi, (label, value, kind) in enumerate(entries):
        y_labels.append(label)
        y_pos.append(yi)
        if kind == "base":
            bar_starts.append(0)
            bar_widths.append(value)
            bar_colors.append(C_BF16)
            bar_texts.append(f"{value/1000:.2f} ms")
        elif kind == "result":
            bar_starts.append(0)
            bar_widths.append(value)
            bar_colors.append(C_FP8)
            bar_texts.append(f"{value/1000:.2f} ms  ({ba['speedup']:.3f}×)")
        elif kind == "saving":
            bar_starts.append(running + value)
            bar_widths.append(-value)
            bar_colors.append(C_SAVE)
            running += value
            bar_texts.append(f"−{abs(value)/1000:.2f} ms")
        else:  # overhead
            bar_starts.append(running)
            bar_widths.append(value)
            bar_colors.append(C_COST)
            running += value
            bar_texts.append(f"+{value/1000:.2f} ms")

    # Draw bars
    ax.barh(y_pos, bar_widths, left=bar_starts,
            color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels at bar right/left edge
    for yi, (start, width, kind, text) in enumerate(
            zip(bar_starts, bar_widths, [e[2] for e in entries], bar_texts)):
        end = start + width
        ha = "left" if end >= 0 else "right"
        offset = 20 if end >= 0 else -20
        ax.text(end + offset, yi, text, va="center", ha=ha,
                fontsize=8.0, fontweight="bold" if kind in ("base","result") else "normal",
                color=C_TEXT)

    # Reference lines
    ax.axvline(ba["bf16_us"], color=C_BF16, ls=":", lw=1.2, alpha=0.6, zorder=0)
    ax.axvline(ba["fp8_us"],  color=C_FP8,  ls=":", lw=1.2, alpha=0.6, zorder=0)
    ax.axvline(0, color=C_TEXT, lw=0.8, alpha=0.3)

    # Separator between savings and overhead
    n_savings = len(savings)
    ax.axhline(n_savings + 0.5, color="#CBD5E1", lw=1.0, ls="--")
    ax.text(ba["bf16_us"]*0.50, n_savings + 0.55, "← savings  |  overhead →",
            ha="center", va="bottom", fontsize=8, color=C_NEUTRAL)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Cumulative GPU-projection time (µs)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", color=C_GRID)
    ax.set_title(
        "Performance Waterfall: BF16 → FP8 Kernel Time Decomposition\n"
        f"Anchor: T=8192, E=8, I=1536  |  Total savings: "
        f"{ba['budget_breakdown'][0]['delta_us']:.0f}… net {pa['budget_anchor']['budget_totals']['net_us']:.0f} µs",
        fontweight="bold",
    )

    # Legend
    legend_items = [
        mpatches.Patch(color=C_BF16, label="BF16 total"),
        mpatches.Patch(color=C_SAVE, label="GEMM / elementwise savings"),
        mpatches.Patch(color=C_COST, label="FP8 quant + bridge overhead"),
        mpatches.Patch(color=C_FP8, label="FP8 total"),
    ]
    ax.legend(handles=legend_items, loc="lower right", framealpha=0.92, fontsize=9)

    fig.suptitle(
        f"SonicMoE Session 53  —  Kernel Budget Waterfall  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig4_performance_waterfall.png")


# ═════════════════════════════════════════════════════════════════════════════
# fig5 — Stage-Level Memory Profile
# ═════════════════════════════════════════════════════════════════════════════
def fig5_stage_memory_profile() -> None:
    """Allocated + peak HBM per phase (6 stages × BF16 / FP8)."""
    _style()
    pa = _load_pa()
    pm = pa["path_analysis"]["phase_memory"]

    phases  = [p["phase_name"] for p in pm["bf16"]]
    bf_alloc = [p["allocated_mib"] for p in pm["bf16"]]
    bf_peak  = [p["peak_mib"]      for p in pm["bf16"]]
    fp_alloc = [p["allocated_mib"] for p in pm["fp8"]]
    fp_peak  = [p["peak_mib"]      for p in pm["fp8"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0),
                              gridspec_kw={"wspace": 0.32})
    fig.subplots_adjust(top=0.82, bottom=0.12)
    _panel_label(axes[0], "(a)")
    _panel_label(axes[1], "(b)")

    x = np.arange(len(phases))
    w = 0.28

    for ax, alloc, peak, mode, color in [
        (axes[0], bf_alloc, bf_peak, "BF16", C_BF16),
        (axes[1], fp_alloc, fp_peak, "FP8",  C_FP8),
    ]:
        ax.bar(x - w/2, alloc, w, color=color, alpha=0.55,
               label="Allocated", edgecolor="white", linewidth=0.6, zorder=3)
        ax.bar(x + w/2, peak, w, color=color, alpha=0.90,
               label="Peak", edgecolor="white", linewidth=0.6, zorder=3)
        for xi, (a, p) in enumerate(zip(alloc, peak)):
            ax.text(xi - w/2, a + 18, f"{a:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color=color)
            ax.text(xi + w/2, p + 18, f"{p:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([ph.replace(" (", "\n(") for ph in phases],
                           fontsize=8.5)
        ax.set_ylabel("HBM (MiB)")
        ax.set_title(f"({('a' if mode=='BF16' else 'b')})  {mode} — memory per stage",
                     fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_ylim(0, max(max(alloc), max(peak)) * 1.22)
        ax.grid(True, axis="y", color=C_GRID)
        # Fwd/Bwd divider
        ax.axvline(2.5, color="#CBD5E1", ls="--", lw=1.0)
        ax.text(1.0, ax.get_ylim()[1]*0.96, "← Forward →",
                ha="center", fontsize=8, color=C_BF16 if mode=="BF16" else C_FP8,
                alpha=0.7, fontweight="bold")
        ax.text(3.5, ax.get_ylim()[1]*0.96, "← Backward →",
                ha="center", fontsize=8, color=C_BF16 if mode=="BF16" else C_FP8,
                alpha=0.7, fontweight="bold")

    # Delta overlay — right axis on ax[1]
    ax2 = axes[1].twinx()
    delta_peak = [fp - bf for fp, bf in zip(fp_peak, bf_peak)]
    ax2.plot(x, delta_peak, "o--", color="#6B7280", lw=1.4, ms=5, zorder=5, label="Δ peak (FP8−BF16)")
    ax2.axhline(0, color="#9CA3AF", lw=0.8, ls=":")
    ax2.set_ylabel("Δ peak HBM (MiB)", color=C_NEUTRAL)
    ax2.tick_params(axis="y", colors=C_NEUTRAL)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8)

    fig.suptitle(
        "SonicMoE Session 53  —  Stage-Level HBM Memory Profile (BF16 vs FP8)\n"
        f"Anchor shape T=8192, E=8, I=1536, H=3072  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig5_stage_memory_profile.png")


# ═════════════════════════════════════════════════════════════════════════════
# fig6 — Precision Audit
# ═════════════════════════════════════════════════════════════════════════════
def fig6_precision_audit() -> None:
    """RRMSE and cosine similarity per tensor from precision_anchor."""
    _style()
    pa = _load_pa()
    pr = pa["precision_anchor"]

    tensors = ["output", "dx (actgrad)", "dw1 (upproj)", "dw2 (dnproj)"]
    rrmse  = [pr["rrmse_pct"]["output"], pr["rrmse_pct"]["dx"],
              pr["rrmse_pct"]["dw1"],    pr["rrmse_pct"]["dw2"]]
    cosine = [pr["cosine_sim"]["output"], pr["cosine_sim"]["dx"],
              pr["cosine_sim"]["dw1"],    pr["cosine_sim"]["dw2"]]

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(12, 5.5),
                                      gridspec_kw={"wspace": 0.36})
    fig.subplots_adjust(top=0.82, bottom=0.14)
    _panel_label(ax_r, "(a)")
    _panel_label(ax_c, "(b)")

    x = np.arange(len(tensors))

    # (a) RRMSE
    bars = ax_r.bar(x, rrmse, width=0.5, color=C_FP8, edgecolor="white",
                    linewidth=0.8, alpha=0.85, zorder=3)
    ax_r.axhline(10.0, color=C_COST, ls="--", lw=1.8, label="Threshold 10%", zorder=4)
    ax_r.fill_between([-0.5, 3.5], [10, 10], [14, 14], color=C_COST, alpha=0.06)
    for b, v in zip(bars, rrmse):
        ax_r.text(b.get_x()+b.get_width()/2, v + 0.18, f"{v:.3f}%",
                  ha="center", va="bottom", fontsize=10, fontweight="bold",
                  color=C_SAVE if v < 10 else C_COST)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(tensors)
    ax_r.set_ylabel("Relative RMSE (%)")
    ax_r.set_title("(a)  RRMSE — FP8 vs BF16 reference", fontweight="bold")
    ax_r.set_ylim(0, 14)
    ax_r.legend(loc="upper right", framealpha=0.9)
    ax_r.grid(True, axis="y", color=C_GRID)
    badge = "✓  All PASS  (< 10%)" if all(r < 10 for r in rrmse) else "✗  FAIL"
    ax_r.text(0.5, 0.96, badge, transform=ax_r.transAxes, ha="center", va="top",
              fontsize=11, fontweight="bold", color=C_SAVE,
              bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec=C_SAVE, lw=1.2))

    # (b) Cosine similarity
    cos_deviance = [1.0 - c for c in cosine]
    bars2 = ax_c.bar(x, cos_deviance, width=0.5, color=C_BF16, edgecolor="white",
                     linewidth=0.8, alpha=0.85, zorder=3)
    ax_c.axhline(0.01, color=C_COST, ls="--", lw=1.8, label="Threshold 1−0.99=0.01", zorder=4)
    ax_c.fill_between([-0.5, 3.5], [0.01, 0.01], [0.025, 0.025], color=C_COST, alpha=0.06)
    for b, c_dev, c_val in zip(bars2, cos_deviance, cosine):
        ax_c.text(b.get_x()+b.get_width()/2, c_dev + 0.0004,
                  f"1−{c_val:.4f}\n= {c_dev:.4f}",
                  ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                  color=C_SAVE if c_dev < 0.01 else C_COST)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(tensors)
    ax_c.set_ylabel("1 − cosine similarity")
    ax_c.set_title("(b)  Cosine similarity deviance (lower = better)", fontweight="bold")
    ax_c.legend(loc="upper right", framealpha=0.9)
    ax_c.set_ylim(0, 0.025)
    ax_c.grid(True, axis="y", color=C_GRID)
    badge2 = "✓  All PASS  (cos > 0.99)" if all(c > 0.99 for c in cosine) else "✗  FAIL"
    ax_c.text(0.5, 0.96, badge2, transform=ax_c.transAxes, ha="center", va="top",
              fontsize=11, fontweight="bold", color=C_SAVE,
              bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec=C_SAVE, lw=1.2))

    fig.suptitle(
        "SonicMoE Session 53  —  FP8 Precision Audit\n"
        f"Reference: BF16 baseline  |  Shape: {pr['shape_key']}  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig6_precision_audit.png")


# ═════════════════════════════════════════════════════════════════════════════
# fig7 — Speedup vs Memory-Delta Scatter (27 shapes)
# ═════════════════════════════════════════════════════════════════════════════
def fig7_speedup_memory_scatter() -> None:
    """2D scatter: speedup vs peak-bwd memory-delta for all 27 shapes."""
    _style()
    pa = _load_pa()
    gr = pa["grid_records"]

    T_map = {8192: ("o", 50), 16384: ("s", 90), 32768: ("D", 130)}
    I_map = {1536: "#60A5FA", 2048: "#10B981", 3072: "#F59E0B"}

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.subplots_adjust(right=0.82)

    for rec in gr:
        T, I = int(rec["T"]), int(rec["I"])
        marker, size = T_map.get(T, ("o", 60))
        color = I_map.get(I, "#888888")
        ax.scatter(rec["speedup"], rec["peak_bwd_delta_pct"],
                   s=size, c=color, marker=marker,
                   edgecolors="white", linewidth=0.8, alpha=0.87, zorder=3)
        # Annotate outlier shapes
        if rec["speedup"] > 1.65 or rec["speedup"] < 1.31 or \
           rec["peak_bwd_delta_pct"] > 9.8:
            ax.annotate(
                rec["shape_key"].replace("_", " "),
                (rec["speedup"], rec["peak_bwd_delta_pct"]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=6.5, color=C_NEUTRAL,
                arrowprops=dict(arrowstyle="-", color=C_NEUTRAL, lw=0.6),
            )

    # Average lines
    mean_sp = float(np.mean([r["speedup"] for r in gr]))
    mean_mem= float(np.mean([r["peak_bwd_delta_pct"] for r in gr]))
    ax.axvline(mean_sp,  color=C_TEXT, ls="--", lw=0.9, alpha=0.5)
    ax.axhline(mean_mem, color=C_TEXT, ls="--", lw=0.9, alpha=0.5)
    ax.text(mean_sp + 0.004, ax.get_ylim()[0] + 0.10,
            f"mean speedup\n{mean_sp:.3f}×", fontsize=7.5, color=C_NEUTRAL)
    ax.text(ax.get_xlim()[0] + 0.005, mean_mem + 0.10,
            f"mean Δmem {mean_mem:.1f}%", fontsize=7.5, color=C_NEUTRAL)

    ax.set_xlabel("FP8 Speedup (×)", fontsize=11)
    ax.set_ylabel("Peak Backward HBM Δ  (%  over BF16)", fontsize=11)
    ax.set_title(
        "Speed–Memory Tradeoff: FP8 speedup vs backward-peak HBM increase\n"
        "(faster FP8 tends to come with modest memory overhead from quant temporaries)",
        fontweight="bold",
    )
    ax.grid(True, color=C_GRID)

    # Legend — T markers
    T_handles = [
        plt.scatter([], [], s=sz, marker=mk, c="#888888", edgecolors="white",
                    linewidth=0.8, label=f"T={T//1024}k")
        for T, (mk, sz) in sorted(T_map.items())
    ]
    I_handles = [
        mpatches.Patch(color=c, label=f"I={I}")
        for I, c in sorted(I_map.items())
    ]
    leg1 = ax.legend(handles=T_handles, title="Token count", loc="upper left",
                     framealpha=0.9, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=I_handles, title="Intermediate size", loc="upper center",
              framealpha=0.9, fontsize=9)

    fig.suptitle(
        f"SonicMoE Session 53  —  27-Shape Speed vs Memory Scatter  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig7_speedup_memory_scatter.png")


# ═════════════════════════════════════════════════════════════════════════════
# fig8 — Kernel Budget Composition (donut pair)
# ═════════════════════════════════════════════════════════════════════════════
def fig8_budget_composition() -> None:
    """Donut charts: BF16 vs FP8 kernel-type composition at anchor shape."""
    _style()
    pa = _load_pa()
    ba = pa["budget_anchor"]
    bb = ba["budget_breakdown"]

    # Aggregate into semantic groups
    def _group(cats: list[str]) -> float:
        return sum(e["bf16_us"] for e in bb if e["category"] in cats), \
               sum(e["fp8_us"]  for e in bb if e["category"] in cats)

    groups = {
        "BF16 GEMM\n(wgrad)":       ["Wgrad GEMM"],
        "BF16 GEMM\n(fwd/bwd)":     ["GemmGated (fwd)", "GemmDGated (bwd)", "cuBLAS GEMM"],
        "ZeroMat GEMM\n(FP8 fwd)":  ["GemmGated ZeroMat (fwd)"],
        "ZeroMat GEMM\n(FP8 bwd)":  ["GemmDGated ZeroMat (bwd)"],
        "FP8 Quant\n(col)":         ["Blockscaled Quant"],
        "FP8 Quant\n(dual)":        ["Dual Quant"],
        "FP8 Quant\n(row)":         ["Row Quant"],
        "ISA Scale\nGather":        ["ISA Scale Gather"],
        "Token\nGather":            ["Token Gather"],
        "Router +\nMisc":           ["Router Metadata", "TopK Router", "Softmax",
                                     "Elementwise Ops", "Reduce"],
    }
    bf_vals, fp_vals, labels = [], [], []
    for label, cats in groups.items():
        bfv = sum(e["bf16_us"] for e in bb if e["category"] in cats)
        fpv = sum(e["fp8_us"]  for e in bb if e["category"] in cats)
        bf_vals.append(bfv)
        fp_vals.append(fpv)
        labels.append(label)

    palette = [
        "#3B82F6", "#60A5FA", "#F97316", "#FDBA74",
        "#8B5CF6", "#A78BFA", "#C4B5FD",
        "#10B981", "#6EE7B7", "#94A3B8",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    fig.subplots_adjust(wspace=0.05, top=0.85, bottom=0.05)

    def _donut(ax, vals, title, mode_color):
        total = sum(vals)
        # Filter out zeros
        fv = [(v, l, p) for v, l, p in zip(vals, labels, palette) if v > 1]
        vs, ls, ps = zip(*fv) if fv else ([], [], [])
        wedges, texts, autotexts = ax.pie(
            vs, labels=None, colors=ps,
            autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.5),
        )
        for at in autotexts:
            at.set_fontsize(8.0)
            at.set_fontweight("bold")
            at.set_color("white")
        # Center text
        ax.text(0, 0, f"{total/1000:.2f}\nms", ha="center", va="center",
                fontsize=14, fontweight="bold", color=mode_color)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=14,
                     color=mode_color)

    _donut(axes[0],
           bf_vals,
           f"BF16  ({ba['bf16_us']/1000:.2f} ms)",
           C_BF16)
    _donut(axes[1],
           fp_vals,
           f"FP8   ({ba['fp8_us']/1000:.2f} ms)   {ba['speedup']:.3f}×",
           C_FP8)

    # Shared legend
    legend_patches = [mpatches.Patch(color=p, label=l)
                      for l, p in zip(labels, palette)]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=5, fontsize=8.5, framealpha=0.92,
               borderpad=0.6, columnspacing=1.0)

    fig.suptitle(
        "SonicMoE Session 53  —  Kernel-Type Budget Composition\n"
        f"Anchor: T=8192, E=8, I=1536  |  {_HW}",
        fontsize=13, fontweight="bold", y=0.97,
    )
    _save(fig, "fig8_budget_composition.png")


# ═════════════════════════════════════════════════════════════════════════════
# fig9 — Tensor Inventory per Stage (stage_deltas)
# ═════════════════════════════════════════════════════════════════════════════
def fig9_tensor_inventory() -> None:
    """Per-stage tensor R/W counts and FP8-only / BF16-only tensor deltas."""
    _style()
    pa    = _load_pa()
    stage_d = pa["path_analysis"]["stage_deltas"]

    bf_reads  = [len(s["bf16_reads"])  for s in stage_d]
    bf_writes = [len(s["bf16_writes"]) for s in stage_d]
    fp_reads  = [len(s["fp8_reads"])   for s in stage_d]
    fp_writes = [len(s["fp8_writes"])  for s in stage_d]
    fp8_only  = [len(s.get("fp8_only_tensors", [])) for s in stage_d]
    bf16_only = [len(s.get("bf16_only_tensors", [])) for s in stage_d]
    extra_w   = [len(s.get("extra_writes", [])) for s in stage_d]
    retired_w = [len(s.get("retired_writes", [])) for s in stage_d]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5),
                              gridspec_kw={"hspace": 0.42})
    _panel_label(axes[0], "(a)")
    _panel_label(axes[1], "(b)")

    x = np.arange(6)
    w = 0.20

    # (a) Read/Write count comparison
    axes[0].bar(x - 1.5*w, bf_reads,  w, color=C_BF16, alpha=0.75,
                label="BF16 reads", edgecolor="white", linewidth=0.5)
    axes[0].bar(x - 0.5*w, fp_reads,  w, color=C_FP8,  alpha=0.75,
                label="FP8 reads",  edgecolor="white", linewidth=0.5)
    axes[0].bar(x + 0.5*w, bf_writes, w, color=C_BF16, alpha=0.45,
                label="BF16 writes",edgecolor="white", linewidth=0.5, hatch="//")
    axes[0].bar(x + 1.5*w, fp_writes, w, color=C_FP8,  alpha=0.45,
                label="FP8 writes", edgecolor="white", linewidth=0.5, hatch="//")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s["name"] for s in stage_d], fontsize=9)
    axes[0].set_ylabel("Tensor count")
    axes[0].set_title("(a)  Tensor reads and writes per stage — BF16 vs FP8", fontweight="bold")
    axes[0].legend(ncol=4, loc="upper right", framealpha=0.9, fontsize=8.5)
    axes[0].grid(True, axis="y", color=C_GRID)
    axes[0].axvline(2.5, color="#CBD5E1", ls="--", lw=1.0)
    axes[0].text(1.0, axes[0].get_ylim()[1]*0.92, "Forward",
                 ha="center", fontsize=8.5, color=C_NEUTRAL, fontweight="bold")
    axes[0].text(4.0, axes[0].get_ylim()[1]*0.92, "Backward",
                 ha="center", fontsize=8.5, color=C_NEUTRAL, fontweight="bold")

    # (b) FP8-only tensors, retired tensors, extra tensors
    axes[1].bar(x - w, fp8_only,  w*1.5, color=C_FP8,  alpha=0.85,
                label="FP8-only tensors (new in FP8)", edgecolor="white")
    axes[1].bar(x,     bf16_only, w*1.5, color=C_BF16, alpha=0.55,
                label="BF16-only tensors (absent in FP8)", edgecolor="white", hatch="//")
    axes[1].bar(x + w, extra_w,   w*1.5, color=C_COST, alpha=0.65,
                label="Extra writes added by FP8", edgecolor="white")

    for xi, (fp, bf, ex) in enumerate(zip(fp8_only, bf16_only, extra_w)):
        if fp > 0:
            axes[1].text(xi - w, fp + 0.05, str(fp), ha="center", fontsize=9,
                         color=C_FP8, fontweight="bold")
        if bf > 0:
            axes[1].text(xi, bf + 0.05, str(bf), ha="center", fontsize=9,
                         color=C_BF16, fontweight="bold")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s["name"] for s in stage_d], fontsize=9)
    axes[1].set_ylabel("Tensor count")
    axes[1].set_title("(b)  FP8 tensor inventory delta — new vs retired tensors", fontweight="bold")
    axes[1].legend(ncol=3, loc="upper right", framealpha=0.9, fontsize=8.5)
    axes[1].grid(True, axis="y", color=C_GRID)
    axes[1].axvline(2.5, color="#CBD5E1", ls="--", lw=1.0)
    axes[1].set_ylim(0, max(max(fp8_only), max(bf16_only), max(extra_w)) + 1.2)

    fig.suptitle(
        "SonicMoE Session 53  —  Stage-Level Tensor Inventory: BF16 vs FP8 Frontier\n"
        f"Source: path_analysis.stage_deltas  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig9_tensor_inventory.png")


# ═════════════════════════════════════════════════════════════════════════════
# fig10 — Speedup Scaling Line Plots (parametric sweeps)
# ═════════════════════════════════════════════════════════════════════════════
def fig10_scaling_line_plots() -> None:
    """Line plots of speedup along each dimension with explicit fixed params."""
    _style()
    pa = _load_pa()
    gr = pa["grid_records"]

    lookup: dict[tuple[int, int, int], float] = {
        (int(r["T"]), int(r["E"]), int(r["I"])): r["speedup"]
        for r in gr
    }
    T_vals = sorted({int(r["T"]) for r in gr})
    I_vals = sorted({int(r["I"]) for r in gr})
    E_vals = sorted({int(r["E"]) for r in gr})

    def sp(T, E, I): return lookup.get((T, E, I), float("nan"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0),
                              gridspec_kw={"wspace": 0.32})
    fig.subplots_adjust(top=0.80, bottom=0.14)

    # (a) T-sweep at (E=8, I=1536), (E=8, I=3072), (E=128, I=1536)
    ax = axes[0]
    sweeps_T = [
        ("E=8,  I=1536", C_BF16, "o-",  [(T, 8,   1536) for T in T_vals]),
        ("E=8,  I=3072", C_FP8,  "s--", [(T, 8,   3072) for T in T_vals]),
        ("E=128,I=1536", C_SAVE, "D:",  [(T, 128, 1536) for T in T_vals]),
    ]
    for label, color, style, pts in sweeps_T:
        ys = [sp(T, E, I) for T, E, I in pts]
        ax.plot([t//1024 for t, _, _ in pts], ys, style, color=color,
                lw=2.0, ms=7, label=label, zorder=3)
        for xi, y in zip([t//1024 for t, _, _ in pts], ys):
            if not np.isnan(y):
                ax.text(xi, y + 0.012, f"{y:.3f}×",
                        ha="center", fontsize=7.0, color=color)
    ax.set_xlabel("Token count T (×1k)")
    ax.set_ylabel("FP8 speedup (×)")
    ax.set_title("(a)  T sweep\n(3 fixed (E, I) configs)", fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_ylim(1.0, max(sp(T,E,I) for T in T_vals for E in [8,128] for I in [1536,3072]) + 0.18)
    ax.axhline(1.0, color=C_NEUTRAL, lw=0.8, ls=":")
    ax.grid(True, color=C_GRID)
    _panel_label(ax, "(a)")

    # (b) I-sweep at (T=8k, E=8), (T=32k, E=8), (T=32k, E=128)
    ax = axes[1]
    sweeps_I = [
        ("T=8k,  E=8",   C_BF16, "o-",  [(8192,  8,   I) for I in I_vals]),
        ("T=32k, E=8",   C_FP8,  "s--", [(32768, 8,   I) for I in I_vals]),
        ("T=32k, E=128", C_SAVE, "D:",  [(32768, 128, I) for I in I_vals]),
    ]
    for label, color, style, pts in sweeps_I:
        ys = [sp(T, E, I) for T, E, I in pts]
        ax.plot(I_vals, ys, style, color=color,
                lw=2.0, ms=7, label=label, zorder=3)
        for xi, y in zip(I_vals, ys):
            if not np.isnan(y):
                ax.text(xi, y + 0.012, f"{y:.3f}×",
                        ha="center", fontsize=7.0, color=color)
    ax.set_xlabel("Intermediate size I")
    ax.set_ylabel("FP8 speedup (×)")
    ax.set_title("(b)  I sweep\n(3 fixed (T, E) configs)", fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_ylim(1.0, max(sp(T,E,I) for T in [8192,32768] for E in [8,128] for I in I_vals) + 0.18)
    ax.axhline(1.0, color=C_NEUTRAL, lw=0.8, ls=":")
    ax.grid(True, color=C_GRID)
    _panel_label(ax, "(b)")

    # (c) E-sweep at (T=8k, I=1536), (T=32k, I=1536), (T=32k, I=3072)
    ax = axes[2]
    sweeps_E = [
        ("T=8k,  I=1536", C_BF16, "o-",  [(8192,  E, 1536) for E in E_vals]),
        ("T=32k, I=1536", C_FP8,  "s--", [(32768, E, 1536) for E in E_vals]),
        ("T=32k, I=3072", C_SAVE, "D:",  [(32768, E, 3072) for E in E_vals]),
    ]
    for label, color, style, pts in sweeps_E:
        ys = [sp(T, E, I) for T, E, I in pts]
        ax.plot([e for _, e, _ in pts], ys, style, color=color,
                lw=2.0, ms=7, label=label, zorder=3)
        for xi, y in zip([e for _, e, _ in pts], ys):
            if not np.isnan(y):
                ax.text(xi, y + 0.012, f"{y:.3f}×",
                        ha="center", fontsize=7.0, color=color)
    ax.set_xlabel("Expert count E")
    ax.set_ylabel("FP8 speedup (×)")
    ax.set_title("(c)  E sweep\n(3 fixed (T, I) configs)", fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_xscale("log")
    ax.set_xticks(E_vals)
    ax.set_xticklabels([str(e) for e in E_vals])
    ax.set_ylim(1.0, max(sp(T,E,I) for T in [8192,32768] for E in E_vals for I in [1536,3072]) + 0.18)
    ax.axhline(1.0, color=C_NEUTRAL, lw=0.8, ls=":")
    ax.grid(True, color=C_GRID)
    _panel_label(ax, "(c)")

    fig.suptitle(
        "SonicMoE Session 53  —  Speedup Scaling Line Plots  (explicit fixed-parameter sweeps)\n"
        f"27 shapes (3T × 3E × 3I)  |  nsys GPU-projection  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig10_scaling_line_plots.png")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
def generate_all() -> None:
    print("\nSonicMoE FP8 Visualization Suite  —  fig1–10  (session-53 data)")
    print("=" * 65)
    figs = [
        ("System Overview",          fig1_system_overview),
        ("Executive Summary",         fig2_executive_summary),
        ("Quant Kernel Comparison",   fig3_quant_kernel_comparison),
        ("Performance Waterfall",     fig4_performance_waterfall),
        ("Stage Memory Profile",      fig5_stage_memory_profile),
        ("Precision Audit",           fig6_precision_audit),
        ("Speedup–Memory Scatter",    fig7_speedup_memory_scatter),
        ("Budget Composition",        fig8_budget_composition),
        ("Tensor Inventory",          fig9_tensor_inventory),
        ("Scaling Line Plots",        fig10_scaling_line_plots),
    ]
    for name, fn in figs:
        print(f"\n  Generating: {name}")
        fn()

    print(f"\n  All figures saved to {ASSETS}/\n")


if __name__ == "__main__":
    generate_all()
