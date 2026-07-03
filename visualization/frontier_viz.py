"""Session 53 Frontier Visualization Suite — SonicMoE FP8 Blockscaled.

Generates four publication-quality figures from the 27-shape nsys grid
(3T × 3E × 3I, nsys GPU-projection + memory):

  fig11  Kernel Runtime Budget Breakdown  (6-panel waterfall)
  fig12  Peak Memory Scaling              (grouped bars + heatmap)
  fig13  Computation Data Flow            (BF16 vs FP8 side-by-side diagram)
  fig14  Multi-Dimensional Speedup Scaling (T/E/I interaction heatmaps)

Data source:
  reports/grid_session53/session53_grid_full.json
  (merged from per-GPU nsys outputs; no GPU required for visualization)

Usage:
  python -m visualization          # generates all figures
  python visualization/frontier_viz.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import matplotlib.patches as mpatches                        # noqa: E402
import matplotlib.patheffects as pe                          # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
import numpy as np                                           # noqa: E402

try:
    import seaborn as sns                                    # noqa: E402
except ImportError:
    sns = None

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
ASSETS  = ROOT / "assets"
GRID_JSON = ROOT / "reports" / "grid_session53" / "session53_grid_full.json"

# ── Palette ───────────────────────────────────────────────────────────────────
C_BF16      = "#2563EB"
C_FP8       = "#EA580C"
C_STASH     = "#7C3AED"
C_SAVE      = "#16A34A"
C_COST      = "#DC2626"
C_NEUTRAL   = "#6B7280"
C_FWD       = "#3B82F6"
C_BWD       = "#F59E0B"
C_QUANT     = "#DC2626"
C_GEMM      = "#2563EB"

# Heatmap diverging palette
CMAP_SPEEDUP = "RdYlGn"
CMAP_MEMORY  = "RdYlBu_r"

_HW = "SM100 Target GPU  ·  PyTorch 2.11  ·  QuACK 0.3.7"

# ── Canonical axis orderings ──────────────────────────────────────────────────
T_VALS = [8192, 16384, 32768]
E_VALS = [8, 32, 128]
I_VALS = [1536, 2048, 3072]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_style() -> None:
    """Publication-quality matplotlib defaults."""
    if sns is not None:
        sns.set_style("whitegrid", {
            "grid.linestyle": "--",
            "grid.alpha": 0.25,
            "axes.edgecolor": "#D1D5DB",
        })
    plt.rcParams.update({
        "font.family":          "serif",
        "font.serif":           ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset":     "stix",
        "font.size":            10,
        "axes.titlesize":       12,
        "axes.labelsize":       10.5,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      8.5,
        "figure.dpi":           200,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
        "savefig.pad_inches":   0.15,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
    })


def _save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / name
    fig.savefig(str(out))
    plt.close(fig)
    print(f"    -> {out}")


def _shape_key(T: int, I: int, E: int, K: int = 8) -> str:
    return f"T{T}_I{I}_E{E}K{K}"


def _short_label(T: int, E: int, I: int) -> str:
    return f"T={T//1024}k\nE={E}\nI={I}"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

class GridData:
    """Loads and indexes the 27-shape nsys grid."""

    def __init__(self, path: Path = GRID_JSON):
        if not path.exists():
            raise FileNotFoundError(f"Grid JSON not found: {path}")
        raw = json.loads(path.read_text())
        self.metadata = raw.get("metadata", {})
        self.shapes: dict[str, dict] = raw.get("shapes", {})

        # Build lookup indices
        self._by_tei: dict[tuple[int, int, int], dict] = {}
        for key, val in self.shapes.items():
            shape = val.get("shape", {})
            t, e, i = shape.get("T", 0), shape.get("E", 0), shape.get("I", 0)
            self._by_tei[(t, e, i)] = val

    def get(self, T: int, E: int, I: int) -> dict | None:
        return self._by_tei.get((T, E, I))

    def speedup(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("speedup", 1.0) if d else 1.0

    def bf16_us(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("bf16", {}).get("per_iter_us", 0) if d else 0

    def fp8_us(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("fp8", {}).get("per_iter_us", 0) if d else 0

    def mem_bf16_bwd(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("memory_bf16", {}).get("peak_bwd_mib", 0) if d else 0

    def mem_fp8_bwd(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("memory_fp8", {}).get("peak_bwd_mib", 0) if d else 0

    def mem_bf16_fwd(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("memory_bf16", {}).get("peak_fwd_mib", 0) if d else 0

    def mem_fp8_fwd(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("memory_fp8", {}).get("peak_fwd_mib", 0) if d else 0

    def budget(self, T: int, E: int, I: int) -> dict[str, dict]:
        d = self.get(T, E, I)
        return d.get("budget_breakdown", {}) if d else {}

    def budget_savings(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return abs(d.get("budget_savings_us", 0)) if d else 0

    def budget_overhead(self, T: int, E: int, I: int) -> float:
        d = self.get(T, E, I)
        return d.get("budget_overhead_us", 0) if d else 0

    def n_shapes(self) -> int:
        return len(self.shapes)

    def available_combos(self) -> list[tuple[int, int, int]]:
        return sorted(self._by_tei.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 11 — Kernel Runtime Budget Breakdown (waterfall + decomposition)
# ═══════════════════════════════════════════════════════════════════════════════

# Unified category ordering for consistent visualization
_SAVE_CATS = ["Wgrad GEMM", "GemmGated (fwd)", "GemmDGated (bwd)"]
_COST_CATS = [
    "GemmGated ZeroMat (fwd)", "GemmDGated ZeroMat (bwd)",
    "Blockscaled Quant", "Dual Quant", "Row Quant", "ISA Scale Gather",
]


def fig11_kernel_runtime_breakdown(data: GridData) -> None:
    """6-panel grid: waterfall budget for 6 representative shapes.

    Shows 2 rows (T=8k, T=32k) × 3 columns (I=1536, 2048, 3072),
    each with a horizontal budget bar decomposing GEMM savings vs quant overhead.
    All panels use E=8 (baseline expert count) to isolate T×I scaling.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.28})

    T_rows = [8192, 32768]
    I_cols = [1536, 2048, 3072]
    E_fixed = 8

    for ri, T in enumerate(T_rows):
        for ci, I in enumerate(I_cols):
            ax = axes[ri, ci]
            bb = data.budget(T, E_fixed, I)
            sp = data.speedup(T, E_fixed, I)
            bf16 = data.bf16_us(T, E_fixed, I)
            fp8 = data.fp8_us(T, E_fixed, I)

            if not bb:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            # Separate SAVE vs COST categories
            saves = []
            costs = []
            for cat, vals in sorted(bb.items(), key=lambda x: x[1]["delta_us"]):
                d = vals["delta_us"]
                if d < -5:
                    saves.append((cat, abs(d)))
                elif d > 5:
                    costs.append((cat, d))

            saves.sort(key=lambda x: -x[1])
            costs.sort(key=lambda x: -x[1])

            # Draw horizontal stacked bars: one for SAVE (green), one for COST (red)
            labels = []
            values = []
            colors = []
            for cat, v in saves:
                short = cat.replace(" (fwd)", "\n(fwd)").replace(" (bwd)", "\n(bwd)")
                labels.append(short)
                values.append(-v)  # negative = saving
                colors.append(C_SAVE)
            for cat, v in costs:
                short = cat.replace(" (fwd)", "\n(fwd)").replace(" (bwd)", "\n(bwd)")
                labels.append(short)
                values.append(v)
                colors.append(C_COST)

            y_pos = np.arange(len(labels))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.8,
                           edgecolor="white", linewidth=0.5, height=0.7)

            # Value labels — anchored at Delta=0 axis to avoid y-axis label overlap
            for bar, v in zip(bars, values):
                bw = bar.get_width()
                ha = "right" if bw < 0 else "left"
                offset = -5 if bw < 0 else 5
                ax.annotate(f"{v:+.0f}",
                            xy=(0, bar.get_y() + bar.get_height() / 2),
                            xytext=(offset, 0), textcoords="offset points",
                            va="center", ha=ha, fontsize=6.5, fontweight="bold",
                            color="#374151")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            ax.axvline(0, color="#374151", lw=0.8)
            ax.set_xlabel("Delta (us)", fontsize=8)
            ax.invert_yaxis()

            # Net annotation
            net = sum(values)
            ax.set_title(
                f"T={T//1024}k, I={I}, E={E_fixed}\n"
                f"BF16={bf16:.0f}  FP8={fp8:.0f}  "
                f"Speedup={sp:.3f}x  Net={net:+.0f}",
                fontsize=9, fontweight="bold",
            )

    # Legend
    fig.legend(
        handles=[
            mpatches.Patch(color=C_SAVE, label="GEMM Savings (replaced by FP8)"),
            mpatches.Patch(color=C_COST, label="FP8 Overhead (quant + ZeroMat)"),
        ],
        loc="lower center", ncol=2, fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "SonicMoE Session 53 — Kernel Runtime Budget Breakdown\n"
        f"E={E_fixed} (fixed)  |  nsys GPU-projection  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig11_kernel_budget_breakdown.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 12 — Peak Memory Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def fig12_memory_breakdown(data: GridData) -> None:
    """3-panel memory analysis:
    (a) Peak backward memory grouped bars (9 shapes at fixed E=8)
    (b) Memory overhead heatmap (FP8 peak_bwd / BF16 peak_bwd) across T×E at I=1536
    (c) Absolute peak scaling — all 27 shapes as scatter, colored by I
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 6),
                             gridspec_kw={"wspace": 0.30})

    # ── (a) Peak Bwd Memory: BF16 vs FP8 (9 shapes, E=8 fixed) ──────────
    ax = axes[0]
    E_fixed = 8
    combos = [(T, I) for T in T_VALS for I in I_VALS]
    labels = [f"T={T//1024}k\nI={I}" for T, I in combos]
    bf16_peaks = [data.mem_bf16_bwd(T, E_fixed, I) for T, I in combos]
    fp8_peaks  = [data.mem_fp8_bwd(T, E_fixed, I) for T, I in combos]

    x = np.arange(len(combos))
    w = 0.35
    b1 = ax.bar(x - w/2, bf16_peaks, w, label="BF16", color=C_BF16, alpha=0.85)
    b2 = ax.bar(x + w/2, fp8_peaks, w, label="FP8", color=C_FP8, alpha=0.85)

    # Delta % labels
    for i, (bf, fp) in enumerate(zip(bf16_peaks, fp8_peaks)):
        if bf > 0 and fp > 0:
            pct = (fp - bf) / bf * 100
            ax.text(x[i] + w/2, fp + 20, f"{pct:+.1f}%",
                    ha="center", fontsize=6, fontweight="bold", color=C_FP8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Peak Backward Memory (MiB)")
    ax.set_title(f"(a) Peak Bwd Memory (E={E_fixed})", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    # ── (b) Memory Overhead Heatmap: T × E at I=1536 ────────────────────
    ax = axes[1]
    I_fixed = 1536
    matrix = np.zeros((len(T_VALS), len(E_VALS)))
    for ti, T in enumerate(T_VALS):
        for ei, E in enumerate(E_VALS):
            bf = data.mem_bf16_bwd(T, E, I_fixed)
            fp = data.mem_fp8_bwd(T, E, I_fixed)
            matrix[ti, ei] = ((fp - bf) / bf * 100) if bf > 0 else 0

    im = ax.imshow(matrix, cmap=CMAP_MEMORY, aspect="auto",
                   vmin=-5, vmax=15)
    for ti in range(len(T_VALS)):
        for ei in range(len(E_VALS)):
            ax.text(ei, ti, f"{matrix[ti, ei]:+.1f}%",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if abs(matrix[ti, ei]) > 8 else "#374151")

    ax.set_xticks(range(len(E_VALS)))
    ax.set_xticklabels([f"E={e}" for e in E_VALS])
    ax.set_yticks(range(len(T_VALS)))
    ax.set_yticklabels([f"T={t//1024}k" for t in T_VALS])
    ax.set_title(f"(b) FP8 Memory Overhead % (I={I_fixed})", fontweight="bold")
    fig.colorbar(im, ax=ax, label="FP8 - BF16 (%)", shrink=0.8)

    # ── (c) Absolute Peak Scaling (all 27 shapes) ────────────────────────
    ax = axes[2]
    i_colors = {1536: "#3B82F6", 2048: "#10B981", 3072: "#F59E0B"}
    for I_val, color in i_colors.items():
        bf_vals = []
        fp_vals = []
        for T in T_VALS:
            for E in E_VALS:
                bf = data.mem_bf16_bwd(T, E, I_val)
                fp = data.mem_fp8_bwd(T, E, I_val)
                if bf > 0 and fp > 0:
                    bf_vals.append(bf)
                    fp_vals.append(fp)
        ax.scatter(bf_vals, fp_vals, c=color, s=40, alpha=0.8,
                   label=f"I={I_val}", edgecolors="white", linewidth=0.5, zorder=3)

    # y=x reference line
    lim_max = max(
        max(data.mem_bf16_bwd(T, E, I) for T in T_VALS for E in E_VALS for I in I_VALS if data.get(T, E, I)),
        max(data.mem_fp8_bwd(T, E, I) for T in T_VALS for E in E_VALS for I in I_VALS if data.get(T, E, I)),
    ) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel("BF16 Peak Bwd (MiB)")
    ax.set_ylabel("FP8 Peak Bwd (MiB)")
    ax.set_title("(c) BF16 vs FP8 Peak Memory (all 27 shapes)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        "SonicMoE Session 53 — Peak Memory Analysis\n"
        f"27 shapes (3T x 3E x 3I)  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    _save(fig, "fig12_memory_scaling.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 13 — Computation Data Flow (BF16 vs FP8 side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_box(ax, xy, w, h, text, color, fontsize=8, alpha=0.15,
              textcolor=None, bold=False, edgecolor=None):
    x0, y0 = xy
    ec = edgecolor or color
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, alpha=alpha,
        edgecolor=ec, linewidth=1.2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    tc = textcolor or "#1F2937"
    ax.text(x0 + w/2, y0 + h/2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=tc, zorder=10)


def _draw_arrow(ax, xy_from, xy_to, color="#374151", lw=1.2,
                style="-|>", connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, color=color, linewidth=lw,
        connectionstyle=connectionstyle, zorder=5, mutation_scale=12,
    )
    ax.add_patch(arrow)


def _draw_annotation(ax, xy, text, fontsize=6.5, color="#6B7280", **kw):
    ax.text(xy[0], xy[1], text, fontsize=fontsize, color=color,
            ha="center", va="center", style="italic", zorder=10, **kw)


def fig13_computation_dataflow(data: GridData) -> None:
    """Side-by-side BF16 vs FP8 zero-materialization data flow.

    Uses T=8192, E=8, I=1536 as the representative shape (reference-scale).
    Shows the key architectural difference: FP8 avoids TK-sized materialization
    via quantize_and_pack(T-sized) + ZeroMat gather-on-the-fly.
    """
    _apply_style()
    T, E, I, K = 8192, 8, 1536, 8
    TK = T * K
    H = 3072

    sp = data.speedup(T, E, I)
    bf16 = data.bf16_us(T, E, I)
    fp8_ = data.fp8_us(T, E, I)
    bf16_bwd = data.mem_bf16_bwd(T, E, I)
    fp8_bwd = data.mem_fp8_bwd(T, E, I)

    fig, axes = plt.subplots(1, 2, figsize=(16, 14))

    BOX_W = 3.2
    BOX_H = 0.55
    COL_X = 2.0
    Y_TOP = 12.5
    DY = -1.15

    # Pastel fill palette: (fill, edge, textcolor) per node type
    _P = {
        "input_bf16": ("#DBEAFE", "#2563EB", "#1E3A8A"),
        "input_fp8":  ("#FFF7ED", "#EA580C", "#7C2D12"),
        "router":     ("#FEF3C7", "#D97706", "#78350F"),
        "quant":      ("#FEE2E2", "#DC2626", "#7F1D1D"),
        "gemm_bf16":  ("#BFDBFE", "#1D4ED8", "#1E3A8A"),
        "gemm_fp8":   ("#BFDBFE", "#EA580C", "#1E3A8A"),
        "tensor_fp8": ("#FFF7ED", "#EA580C", "#7C2D12"),
        "tensor_bf16":("#DBEAFE", "#2563EB", "#1E3A8A"),
        "gather":     ("#FEE2E2", "#B91C1C", "#7F1D1D"),
        "scatter":    ("#F1F5F9", "#475569", "#1E293B"),
        "swiglu":     ("#EDE9FE", "#7C3AED", "#4C1D95"),
        "bwd_bf16":   ("#E0F2FE", "#0369A1", "#0C4A6E"),
        "bwd_fp8":    ("#ECFDF5", "#059669", "#064E3B"),
        "save_note":  ("#DCFCE7", "#16A34A", "#14532D"),
    }

    def _box(ax, xy, w, h, text, key, fontsize=8, bold=False):
        fill, edge, tc = _P[key]
        patch = FancyBboxPatch(
            (xy[0], xy[1]), w, h,
            boxstyle="round,pad=0.025",
            facecolor=fill, alpha=1.0,
            edgecolor=edge, linewidth=1.4, zorder=2,
        )
        ax.add_patch(patch)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal",
                color=tc, zorder=10, multialignment="center")

    def _note(ax, xy, text, key, fontsize=7):
        fill, edge, tc = _P[key]
        ax.text(xy[0], xy[1], text,
                fontsize=fontsize, color=tc, ha="center", va="center",
                style="italic", zorder=11, multialignment="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=fill,
                          edgecolor=edge, linewidth=0.8, alpha=0.95))

    for panel_idx, (ax, mode, main_color) in enumerate([
        (axes[0], "bf16", C_BF16),
        (axes[1], "fp8",  C_FP8),
    ]):
        ax.set_xlim(-0.3, 7.5)
        ax.set_ylim(-1.5, 14)
        ax.set_axis_off()
        ax.set_aspect("equal")

        is_fp8 = (mode == "fp8")
        input_key = "input_fp8" if is_fp8 else "input_bf16"

        # Section background bands (approximate fwd/bwd split)
        bwd_band = 3.6 if is_fp8 else 2.6
        ax.axhspan(-1.5, bwd_band, color="#F0FFF4", alpha=0.35, zorder=0)
        ax.axhspan(bwd_band, 14.0, color="#EFF6FF" if not is_fp8 else "#FFF7ED",
                   alpha=0.20, zorder=0)

        title = "FP8 Frontier (Zero-Materialization)" if is_fp8 else "BF16 Baseline"
        ax.text(COL_X + BOX_W / 2, Y_TOP + 1.2, title,
                fontsize=14, fontweight="bold", ha="center", color=main_color)
        ax.text(COL_X + BOX_W / 2, Y_TOP + 0.72,
                f"T={T}, H={H}, I={I}, E={E}, K={K}",
                fontsize=8, ha="center", color="#6B7280", style="italic")

        # Section labels on left margin
        ax.text(-0.16, 9.5, "FWD", fontsize=10, fontweight="bold",
                color=main_color, rotation=90, va="center", ha="center", alpha=0.70)
        ax.text(-0.16, 1.0, "BWD", fontsize=10, fontweight="bold",
                color=main_color, rotation=90, va="center", ha="center", alpha=0.70)

        # ── FORWARD PASS ──────────────────────────────────────────────────
        y = Y_TOP

        # Input
        _box(ax, (COL_X, y), BOX_W, BOX_H,
             f"x  [T \u00d7 H]  BF16\n{T} \u00d7 {H} = {T*H*2/1048576:.0f} MiB",
             input_key, fontsize=8, bold=True)

        # Routing
        y += DY
        _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H + 0.02),
                    (COL_X + BOX_W/2, y + BOX_H + DY + BOX_H - 0.02),
                    color=main_color)
        _box(ax, (COL_X, y), BOX_W, BOX_H,
             f"TopK Router  \u2192  A_idx [T\u2192TK]\n({T} \u2192 {TK})",
             "router", fontsize=7.5)

        if is_fp8:
            # Quantize (T-sized only)
            y += DY
            _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H + 0.02),
                        (COL_X + BOX_W/2, y + BOX_H + DY + BOX_H - 0.02),
                        color=main_color)
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 "quantize_and_pack(x)\nx_fp8 [T \u00d7 H]  FP8 + scales",
                 "quant", fontsize=7.5, bold=True)
            _note(ax, (COL_X + BOX_W + 0.85, y + BOX_H / 2),
                  "T-sized!\nNot TK-sized", "quant", fontsize=7)

            # GemmGated ZeroMat
            y += DY
            _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H + 0.02),
                        (COL_X + BOX_W/2, y + BOX_H + DY + BOX_H * 1.2 - 0.02),
                        color=main_color)
            _box(ax, (COL_X, y), BOX_W, BOX_H * 1.2,
                 "GemmGated ZeroMat\nx_fp8[T\u00d7H] + A_idx \u2192 z_fp8[T\u00d72I]\n"
                 "Gathers rows on-the-fly via A_idx",
                 "gemm_fp8", fontsize=7.5, bold=True)
            _note(ax, (COL_X + BOX_W + 0.95, y + BOX_H * 0.65),
                  "No TK\u00d7H FP8\nmaterialized!", "save_note", fontsize=7.5)

            # Epilogue FP8 output tensor
            y += DY * 1.1
            _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H * 1.2 + 0.02),
                        (COL_X + BOX_W/2, y + BOX_H * 1.2 + DY + BOX_H - 0.02),
                        color=main_color)
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 "z_fp8  [T \u00d7 2I]  FP8\nCUTLASS epilogue quant \u2014 zero alloc",
                 "tensor_fp8", fontsize=7.5, bold=True)
            _note(ax, (COL_X + BOX_W + 0.85, y + BOX_H / 2),
                  "CUTLASS epilogue\nwrites FP8 directly", "save_note", fontsize=6.5)
        else:
            # BF16: explicit Gather
            y += DY
            _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H + 0.02),
                        (COL_X + BOX_W/2, y + BOX_H + DY + BOX_H - 0.02),
                        color=main_color)
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 f"Gather x \u2192 x_TK [TK\u00d7H]  BF16\n{TK}\u00d7{H} = {TK*H*2/1048576:.0f} MiB",
                 "gather", fontsize=7.5)

            y += DY
            _draw_arrow(ax, (COL_X + BOX_W/2, y + BOX_H + 0.02),
                        (COL_X + BOX_W/2, y + BOX_H + DY + BOX_H - 0.02),
                        color=main_color)
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 "GemmGated (CUTLASS)\nx_TK[TK\u00d7H] \u00d7 W1[H\u00d72I] \u2192 z[TK\u00d72I]",
                 "gemm_bf16", fontsize=7.5)

            y += DY
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 f"z  [TK\u00d72I]  BF16\n{TK}\u00d7{2*I} = {TK*2*I*2/1048576:.0f} MiB",
                 "tensor_bf16", fontsize=7.5)

        # SwiGLU
        _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                    (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                    color=main_color)
        y += DY
        _box(ax, (COL_X, y), BOX_W, BOX_H,
             "SwiGLU Activation\ny1 = swish(z[:,:I]) \u2299 z[:,I:]",
             "swiglu", fontsize=7.5)

        # DownProj
        _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                    (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                    color=main_color)
        y += DY
        if is_fp8:
            _box(ax, (COL_X, y), BOX_W, BOX_H * 1.1,
                 "GemmDGated ZeroMat\ny_fp8[T\u00d7I] + A_idx \u2192 o[T\u00d7H]\n"
                 "Scatter-reduce in kernel",
                 "gemm_fp8", fontsize=7.5, bold=True)
        else:
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 "GemmDGated (CUTLASS)\ny1[TK\u00d7I] \u00d7 W2[I\u00d7H] \u2192 y2[TK\u00d7H]",
                 "gemm_bf16", fontsize=7.5)

        # Scatter + Output
        _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                    (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                    color=main_color)
        y += DY
        if not is_fp8:
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 "Scatter-Reduce  y2[TK\u00d7H] \u2192 o[T\u00d7H]",
                 "scatter", fontsize=7.5)
            _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                        (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                        color=main_color)
            y += DY

        _box(ax, (COL_X, y), BOX_W, BOX_H,
             f"output  [T\u00d7H]  BF16\n{T}\u00d7{H} = {T*H*2/1048576:.0f} MiB",
             input_key, fontsize=8, bold=True)

        # ── BACKWARD divider ──────────────────────────────────────────────
        bwd_y = y + DY * 0.8
        ax.plot([0.2, 6.8], [bwd_y + 0.25, bwd_y + 0.25],
                ls="--", lw=1.0, color="#94A3B8", zorder=3)
        ax.add_patch(FancyBboxPatch(
            (COL_X + BOX_W / 2 - 0.95, bwd_y + 0.08), 1.90, 0.34,
            boxstyle="round,pad=0.04", facecolor="white",
            edgecolor="#94A3B8", linewidth=0.8, zorder=4,
        ))
        ax.text(COL_X + BOX_W / 2, bwd_y + 0.25,
                "\u2193  BACKWARD",
                fontsize=8.5, fontweight="bold", ha="center", va="center",
                color=main_color, alpha=0.90, zorder=5)

        # Backward GEMMs
        y = bwd_y - 0.2
        bwd_key = "bwd_fp8" if is_fp8 else "bwd_bf16"
        if is_fp8:
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 "dGemmDGated ZeroMat \u2192 dy_fp8\n"
                 "Wgrad auto-tune: OFF at I=1536",
                 bwd_key, fontsize=7)
        else:
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 "dGemmDGated \u2192 dy [TK\u00d7I]  BF16\n"
                 "dW2 via Wgrad GEMM",
                 bwd_key, fontsize=7)

        _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                    (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                    color=main_color)
        y += DY
        if is_fp8:
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 "dGemmGated ZeroMat \u2192 dx_fp8\ndW1 via Wgrad GEMM",
                 bwd_key, fontsize=7)
        else:
            _box(ax, (COL_X, y), BOX_W, BOX_H,
                 "dGemmGated \u2192 dx [TK\u00d7H]  BF16\ndW1 via Wgrad GEMM",
                 bwd_key, fontsize=7)

        _draw_arrow(ax, (COL_X + BOX_W/2, y + 0.02),
                    (COL_X + BOX_W/2, y + DY + BOX_H - 0.02),
                    color=main_color)
        y += DY
        _box(ax, (COL_X, y), BOX_W, BOX_H,
             f"dx  [T\u00d7H]  BF16\n{T}\u00d7{H} = {T*H*2/1048576:.0f} MiB",
             input_key, fontsize=8, bold=True)

        # Memory / timing badge
        if is_fp8:
            mem_text = f"Peak Bwd: {fp8_bwd:.0f} MiB   |   {fp8_:.0f} \u03bcs/iter"
        else:
            mem_text = f"Peak Bwd: {bf16_bwd:.0f} MiB   |   {bf16:.0f} \u03bcs/iter"
        ax.text(COL_X + BOX_W / 2, -1.1, mem_text,
                fontsize=9, fontweight="bold", ha="center", color=main_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=main_color,
                          alpha=0.10, edgecolor=main_color, linewidth=1.0))

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#DBEAFE", edgecolor="#2563EB", label="BF16 tensor / BF16 GEMM"),
        mpatches.Patch(facecolor="#FFF7ED", edgecolor="#EA580C", label="FP8 tensor / FP8 ZeroMat"),
        mpatches.Patch(facecolor="#FEE2E2", edgecolor="#DC2626", label="Quantization kernel"),
        mpatches.Patch(facecolor="#FEF3C7", edgecolor="#D97706", label="Top-k Router"),
        mpatches.Patch(facecolor="#EDE9FE", edgecolor="#7C3AED", label="SwiGLU Activation"),
        mpatches.Patch(facecolor="#F1F5F9", edgecolor="#475569", label="Scatter / Reduce"),
        mpatches.Patch(facecolor="#DCFCE7", edgecolor="#16A34A", label="FP8 savings annotation"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, fontsize=8.5,
               framealpha=0.94, bbox_to_anchor=(0.5, -0.005),
               edgecolor="#CBD5E1")

    fig.suptitle(
        "SonicMoE Session 53 \u2014 Computation Data Flow\n"
        f"BF16 Baseline vs FP8 Zero-Materialization  |  Speedup: {sp:.3f}\u00d7",
        fontsize=14, fontweight="bold", y=0.99,
    )
    _save(fig, "fig13_computation_dataflow.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 14 — Multi-Dimensional Speedup Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def fig14_speedup_scaling(data: GridData) -> None:
    """Redesigned 2-row layout:
    Row 0 — Three speedup heatmaps: T×I (E=8 fixed), T×E (I=1536 fixed), E×I (avg over T).
    Row 1 — Grouped budget bars: BF16 vs FP8 absolute kernel time per category
             at the representative anchor shape (T=8k, E=8, I=1536).
    """
    _apply_style()
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[2.0, 1.7],
        hspace=0.42, wspace=0.32,
    )

    def _heat(ax, matrix, row_labels, col_labels, title, xlabel, ylabel, subtitle=""):
        im = ax.imshow(matrix, cmap=CMAP_SPEEDUP, aspect="auto", vmin=1.0, vmax=1.72)
        nrows, ncols = matrix.shape
        for ri in range(nrows):
            for ci in range(ncols):
                v = matrix[ri, ci]
                ax.text(ci, ri, f"{v:.3f}\u00d7",
                        ha="center", va="center", fontsize=10.5, fontweight="bold",
                        color="white" if v > 1.42 else "#374151")
        ax.set_xticks(range(ncols))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(nrows))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=8)
        if subtitle:
            ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7.5, color="#6B7280",
                    style="italic")
        fig.colorbar(im, ax=ax, label="Speedup (\u00d7)", shrink=0.82)

    # ── (a) T × I at E=8 ─────────────────────────────────────────────────
    E_fixed = 8
    mat_a = np.zeros((len(T_VALS), len(I_VALS)))
    for ti, T in enumerate(T_VALS):
        for ii, I in enumerate(I_VALS):
            mat_a[ti, ii] = data.speedup(T, E_fixed, I)
    _heat(fig.add_subplot(gs[0, 0]), mat_a,
          [f"T={t//1024}k" for t in T_VALS], [f"I={i}" for i in I_VALS],
          f"(a)  T \u00d7 I  speedup", "Intermediate size I", "Tokens T",
          subtitle=f"E={E_fixed} fixed")

    # ── (b) T × E at I=1536 ──────────────────────────────────────────────
    I_fixed = 1536
    mat_b = np.zeros((len(T_VALS), len(E_VALS)))
    for ti, T in enumerate(T_VALS):
        for ei, E in enumerate(E_VALS):
            mat_b[ti, ei] = data.speedup(T, E, I_fixed)
    _heat(fig.add_subplot(gs[0, 1]), mat_b,
          [f"T={t//1024}k" for t in T_VALS], [f"E={e}" for e in E_VALS],
          f"(b)  T \u00d7 E  speedup", "Expert count E", "Tokens T",
          subtitle=f"I={I_fixed} fixed")

    # ── (c) E × I averaged over T ─────────────────────────────────────────
    mat_c = np.zeros((len(E_VALS), len(I_VALS)))
    for ei, E in enumerate(E_VALS):
        for ii, I in enumerate(I_VALS):
            vals = [data.speedup(T, E, I) for T in T_VALS if data.get(T, E, I)]
            mat_c[ei, ii] = float(np.mean(vals)) if vals else 0.0
    _heat(fig.add_subplot(gs[0, 2]), mat_c,
          [f"E={e}" for e in E_VALS], [f"I={i}" for i in I_VALS],
          "(c)  E \u00d7 I  speedup", "Intermediate size I", "Expert count E",
          subtitle="averaged over T in {8k, 16k, 32k}")

    # ── (d) Grouped budget bars — BF16 vs FP8 per category ───────────────
    ax_d = fig.add_subplot(gs[1, :])
    T_rep, E_rep, I_rep = 8192, 8, 1536
    bb = data.budget(T_rep, E_rep, I_rep)
    bf16_total = data.bf16_us(T_rep, E_rep, I_rep)
    fp8_total = data.fp8_us(T_rep, E_rep, I_rep)
    sp_rep = data.speedup(T_rep, E_rep, I_rep)

    # Keep categories significant in either mode (>80 µs)
    sig = [(cat, v) for cat, v in bb.items()
           if v["bf16_us"] > 80 or v["fp8_us"] > 80]
    sig.sort(key=lambda x: x[1]["bf16_us"], reverse=True)

    n = len(sig)
    x = np.arange(n)
    w = 0.36
    bf16_ms = np.array([v["bf16_us"] for _, v in sig]) / 1000.0
    fp8_ms  = np.array([v["fp8_us"]  for _, v in sig]) / 1000.0

    ax_d.bar(x - w/2, bf16_ms, w, color=C_BF16, alpha=0.82,
             label=f"BF16  ({bf16_total/1000:.1f} ms total)",
             edgecolor="white", linewidth=0.5)
    ax_d.bar(x + w/2, fp8_ms, w, color=C_FP8, alpha=0.82,
             label=f"FP8   ({fp8_total/1000:.1f} ms total, {sp_rep:.3f}\u00d7 speedup)",
             edgecolor="white", linewidth=0.5)

    # Delta annotation above each group
    for xi, (bv, fv) in enumerate(zip(bf16_ms, fp8_ms)):
        d_pct = (fv - bv) / bv * 100 if bv > 0 else 0
        color = C_SAVE if d_pct < 0 else (C_COST if d_pct > 5 else C_NEUTRAL)
        ax_d.text(xi, max(bv, fv) + 0.04, f"{d_pct:+.0f}%",
                  ha="center", fontsize=7.2, fontweight="bold", color=color)

    cat_labels = [c.replace(" (fwd)", "\n(fwd)").replace(" (bwd)", "\n(bwd)")
                  for c, _ in sig]
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(cat_labels, fontsize=8.5)
    ax_d.set_ylabel("Time (ms / iter)")
    ax_d.set_title(
        f"(d)  Per-category kernel time: BF16 vs FP8"
        f"  \u2014  T={T_rep//1024}k, E={E_rep}, I={I_rep}",
        fontweight="bold",
    )
    ax_d.legend(loc="upper right", framealpha=0.92)
    ax_d.axhline(0, color="#9CA3AF", lw=0.5)

    fig.suptitle(
        "SonicMoE Session 53 \u2014 Multi-Dimensional FP8 Speedup Scaling\n"
        f"27 shapes (3T \u00d7 3E \u00d7 3I)  |  nsys GPU-projection  |  {_HW}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig14_speedup_scaling.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_frontier() -> None:
    """Generate all Session 53 frontier figures."""
    print("\n  === SonicMoE Session 53 Frontier Visualization ===")
    data = GridData()
    n = data.n_shapes()
    combos = data.available_combos()
    print(f"  Loaded {n} shapes from {GRID_JSON.name}")

    print(f"\n  Generating: fig11 -- Kernel Runtime Budget Breakdown ({n} shapes)")
    fig11_kernel_runtime_breakdown(data)

    print(f"  Generating: fig12 -- Peak Memory Scaling ({n} shapes)")
    fig12_memory_breakdown(data)

    print(f"  Generating: fig13 -- Computation Data Flow (representative shape)")
    fig13_computation_dataflow(data)

    print(f"  Generating: fig14 -- Multi-Dimensional Speedup Scaling ({n} shapes)")
    fig14_speedup_scaling(data)

    print(f"\n  All frontier figures saved to {ASSETS}/\n")


if __name__ == "__main__":
    generate_frontier()
