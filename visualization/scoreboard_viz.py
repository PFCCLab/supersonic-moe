#!/usr/bin/env python3
"""
SonicMoE Unified Buffer Lifecycle × Scoreboard
================================================

Single composite figure fusing:
  • Twin Gantt timeline (BF16 left | FP8 right) — R/W/L segment colouring
  • Cumulative memory envelope overlaid on each Gantt (twin y-axis)
  • Critical-path DAG flow arrows between major buffers
  • Operator-flow summary table (bottom strip)

Usage::

    python visualization/scoreboard_viz.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ─── paths ────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SB_PATH = ROOT / "scoreboard.json"

# ─── palette ──────────────────────────────────────────────────────────
_S = {"W": "#D32F2F", "R": "#1565C0", "L": "#CFD8DC"}  # state colours
_C = {  # buffer-class colours (for y-label tinting)
    "data": "#1565C0", "weight": "#2E7D32", "grad": "#C62828",
    "quant": "#E65100", "meta": "#6A1B9A", "aux": "#546E7A",
}
_CO = {"data": 0, "weight": 1, "grad": 2, "quant": 3, "meta": 4, "aux": 5}

# ─── fonts ────────────────────────────────────────────────────────────
F = dict(sup=15, title=12.5, label=10.5, tick=9, ann=8, leg=8.5, tbl=8)


def _load() -> dict:
    if not SB_PATH.exists():
        sys.exit(f"scoreboard.json not found — run  python tools/scoreboard.py")
    return json.loads(SB_PATH.read_text())


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

def _buf_list(bf16: dict, fp8: dict, min_mib: float = 0.5) -> list[str]:
    pool: dict[str, tuple] = {}
    for m in (bf16, fp8):
        for bn, br in m["bufs"].items():
            if br["mib"] < min_mib:
                continue
            k = (_CO.get(br["class"], 9), -br["mib"], bn)
            if bn not in pool or br["mib"] > -pool[bn][1]:
                pool[bn] = k
    return [n for n, _ in sorted(pool.items(), key=lambda x: x[1])]


def _phase_mem(mode: dict) -> np.ndarray:
    return np.array([ph["alloc_mib"] for ph in mode["board"]])


# ─── Critical dataflow chain definitions ─────────────────────────────
# Waypoints: (buffer_name, phase_index).
# Same-buffer consecutive = carry-over (horizontal).
# Different-buffer = operator transformation (orthogonal right-angle routing).

_CHAINS: dict[str, list[dict]] = {
    "bf16": [
        {"name": "Fwd Act",   "c": "#1565C0", "d": False,
         "wps": [("x",1), ("y1",1), ("y1",2), ("output",2)]},
        {"name": "Gate→Grad", "c": "#E65100", "d": False,
         "wps": [("z",1), ("z",3), ("dz",3), ("dz",5), ("dx",5)]},
        {"name": "Wgrad",     "c": "#2E7D32", "d": True,
         "wps": [("dz",4), ("dw1",4)]},
    ],
    "fp8": [
        {"name": "Fwd FP8",    "c": "#1565C0", "d": False,
         "wps": [("x",1), ("fwd_fp8",1), ("fwd_fp8",2), ("output",2)]},
        {"name": "FP8 Bridge",  "c": "#E65100", "d": False,
         "wps": [("z_fp8",1), ("z_fp8",3), ("bwd_fp8",3), ("bwd_fp8",5), ("dx",5)]},
        {"name": "Prequant",    "c": "#6A1B9A", "d": True,
         "wps": [("z_fp8",3), ("prequant_bwd_0",3), ("prequant_bwd_0",4), ("dw1",4)]},
    ],
}


def _draw_chains(ax: plt.Axes, mode_key: str,
                 buf_idx: dict[str, int]) -> list[dict]:
    """Draw critical-path dataflow chains as orthogonal polylines.

    Returns list of chain defs that were actually rendered (for legend).
    """
    chains = _CHAINS.get(mode_key, [])
    drawn: list[dict] = []
    n = len(chains)

    for ci, ch in enumerate(chains):
        wps = [(b, p) for b, p in ch["wps"] if b in buf_idx]
        if len(wps) < 2:
            continue
        drawn.append(ch)

        color = ch["c"]
        ls: object = (0, (5, 3)) if ch["d"] else "-"
        v_off = (ci - (n - 1) / 2) * 0.14

        # ── Build orthogonal path through the Gantt ──
        path: list[tuple[float, float]] = []
        for wi, (buf, phase) in enumerate(wps):
            x, y = phase + 0.5, buf_idx[buf] + v_off
            if wi == 0:
                path.append((x, y))
                continue
            prev_buf, prev_ph = wps[wi - 1]
            yp = buf_idx[prev_buf] + v_off
            if prev_buf == buf:                          # carry-over
                path.append((x, y))
            elif prev_ph == phase:                       # same-phase op
                x_jog = phase + 0.88
                path.extend([(x_jog, yp), (x_jog, y), (x, y)])
            else:                                        # cross-phase + buf
                x_mid = max(prev_ph, phase) + 0.02
                path.extend([(x_mid, yp), (x_mid, y), (x, y)])

        if len(path) < 2:
            continue

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        # Polyline shaft (with optional dashes)
        ax.plot(xs, ys, color=color, lw=2.0, ls=ls, alpha=0.6, zorder=7,
                solid_capstyle="round", solid_joinstyle="round")

        # Arrowhead at terminus (invisible-shaft FancyArrowPatch)
        arrow = mpatches.FancyArrowPatch(
            (xs[-2], ys[-2]), (xs[-1], ys[-1]),
            arrowstyle="-|>", color=color, lw=0.01,
            alpha=0.7, mutation_scale=14, zorder=8,
        )
        ax.add_patch(arrow)

        # Dots at actual waypoints (not routing-jog intermediates)
        for buf, phase in wps:
            if buf in buf_idx:
                ax.plot(phase + 0.5, buf_idx[buf] + v_off, "o",
                        color=color, ms=3.5, alpha=0.55, zorder=9)

    return drawn


# ═════════════════════════════════════════════════════════════════════════
# Helpers – compact formatting
# ═════════════════════════════════════════════════════════════════════════

def _kdim(v: int) -> str:
    """Compact dimension: 65536→64K, 3072→3K, 8→8."""
    if v >= 1024:
        k = v / 1024
        return f"{k:g}K"
    return str(v)


# ═════════════════════════════════════════════════════════════════════════
# Gantt + Scoreboard + Memory Envelope (unified panel)
# ═════════════════════════════════════════════════════════════════════════

def _draw_panel(ax: plt.Axes, mode: dict, buf_list: list[str],
                title: str, mode_key: str = "",
                show_ylabels: bool = True) -> list[dict]:
    bufs = mode["bufs"]
    board = mode["board"]
    n_ph = len(board)
    n_buf = len(buf_list)

    BAR_H = 0.68
    # R|W sub-column geometry within each phase unit [j, j+1]
    R_L, R_R = 0.03, 0.47
    W_L, W_R = 0.53, 0.97

    # ── Phase background tinting by memory pressure ──
    mem = _phase_mem(mode)
    peak = mem.max()
    for j in range(n_ph):
        intensity = 0.012 + (mem[j] / peak) * 0.04 if peak > 0 else 0.012
        ax.axvspan(j, j + 1, color="#90A4AE", alpha=intensity, zorder=0)

    # ── R|W sub-column dividers (thin dotted line at phase center) ──
    for j in range(n_ph):
        ax.axvline(j + 0.5, color="#CFD8DC", lw=0.5, ls=":", alpha=0.6, zorder=0)

    # ── Sub-column headers: R (left) / W (right) ──
    for j in range(n_ph):
        ax.text(j + 0.25, -1.5, "R", ha="center", va="bottom",
                fontsize=6, color=_S["R"], fontweight="bold", alpha=0.5)
        ax.text(j + 0.75, -1.5, "W", ha="center", va="bottom",
                fontsize=6, color=_S["W"], fontweight="bold", alpha=0.5)

    # ── Operator labels at phase column top ──
    for j, ph in enumerate(board):
        ax.text(j + 0.5, -1.1, ph["name"], ha="center", va="bottom",
                fontsize=F["ann"] - 0.5, color="#555",
                fontweight="bold", fontstyle="italic")

    # ── Gantt bars: R → left sub-col, W → right sub-col, L → full ──
    for i, bn in enumerate(buf_list):
        if bn not in bufs:
            continue
        br = bufs[bn]
        alive = br["alive"]
        for j in range(n_ph):
            if j < alive[0] or j > alive[1]:
                continue
            state = board[j]["state"].get(bn, "L")
            if state == "R":
                ax.barh(i, R_R - R_L, left=j + R_L, height=BAR_H,
                        color=_S["R"], alpha=1.0, edgecolor="white",
                        linewidth=0.4, zorder=3)
                ax.text(j + 0.25, i, "R", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold", zorder=4)
            elif state == "W":
                ax.barh(i, W_R - W_L, left=j + W_L, height=BAR_H,
                        color=_S["W"], alpha=1.0, edgecolor="white",
                        linewidth=0.4, zorder=3)
                ax.text(j + 0.75, i, "W", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold", zorder=4)
            else:  # L — live/idle
                ax.barh(i, W_R - R_L, left=j + R_L, height=BAR_H,
                        color=_S["L"], alpha=0.28, edgecolor="white",
                        linewidth=0.4, zorder=2)

    # ── Memory envelope (twinx) ──
    ax2 = ax.twinx()
    xs = np.arange(n_ph) + 0.5
    ax2.step(xs, mem, where="mid", color="#E65100", lw=2.0, alpha=0.5, zorder=5)
    ax2.fill_between(xs, mem, alpha=0.06, color="#E65100", step="mid")
    pk_idx = int(np.argmax(mem))
    ax2.plot(xs[pk_idx], mem[pk_idx], "v", color="#E65100", ms=7, zorder=6)
    ax2.annotate(f"peak {mem[pk_idx]:.0f}M",
                 xy=(xs[pk_idx], mem[pk_idx]),
                 xytext=(xs[pk_idx], mem[pk_idx] + peak * 0.12),
                 fontsize=F["ann"], fontweight="bold", color="#BF360C",
                 arrowprops=dict(arrowstyle="-", color="#BF360C",
                                 lw=0.6, alpha=0.4),
                 ha="center", va="bottom")
    ax2.set_ylim(0, peak * 1.35)
    ax2.set_ylabel("MiB", fontsize=7, color="#BF360C", alpha=0.6,
                    rotation=0, labelpad=10)
    ax2.tick_params(axis="y", labelsize=6, colors="#BF360C")

    # FWD / BWD divider
    ax.axvline(3, color="#C62828", lw=1.0, ls="--", alpha=0.25, zorder=1)
    ax.text(1.5, -0.6, "FORWARD", ha="center", fontsize=F["ann"],
            color="#1565C0", fontweight="bold", alpha=0.4)
    ax.text(4.5, -0.6, "BACKWARD", ha="center", fontsize=F["ann"],
            color="#C62828", fontweight="bold", alpha=0.4)

    # ── Compact left y-axis labels: name · dtype · K-shape · MiB ──
    if show_ylabels:
        max_name = max((len(bn) for bn in buf_list), default=1)
        labels, label_colors = [], []
        for bn in buf_list:
            if bn not in bufs:
                labels.append(bn)
                label_colors.append("#BDBDBD")
                continue
            br = bufs[bn]
            cls_c = _C.get(br.get("class", "aux"), "#333")
            dt = br.get("dtype", "?")
            sh = "\u00d7".join(_kdim(s) for s in br.get("shape", []))
            m = br["mib"]
            ms = f"{m:.0f}M" if m >= 1 else f"{m:.1f}M"
            labels.append(f"{bn:<{max_name}}  {dt} {sh} {ms}")
            label_colors.append(cls_c)
        ax.set_yticks(range(n_buf))
        ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
        for tick, clr in zip(ax.get_yticklabels(), label_colors):
            tick.set_color(clr)
    else:
        ax.set_yticks(range(n_buf))
        ax.set_yticklabels([])

    ax.set_xticks(np.arange(n_ph) + 0.5)
    ax.set_xticklabels([f"P{j}" for j in range(n_ph)],
                       fontsize=F["tick"], fontweight="bold")
    ax.set_xlim(0, n_ph)
    ax.set_ylim(n_buf - 0.5, -1.8)

    # Phase grid
    for j in range(n_ph + 1):
        ax.axvline(j, color="#E0E0E0", lw=0.4, zorder=0)

    # Class separator lines
    prev_cls = None
    for i, bn in enumerate(buf_list):
        cls = bufs.get(bn, {}).get("class", "?")
        if prev_cls and cls != prev_cls:
            ax.axhline(i - 0.5, color="#BDBDBD", lw=0.6, ls=":", zorder=1)
        prev_cls = cls

    ax.set_facecolor("#FAFAFA")
    ax.set_title(title, fontsize=F["title"], fontweight="bold", pad=18)
    ax.tick_params(length=0)
    return _CHAINS.get(mode_key, [])


# ═════════════════════════════════════════════════════════════════════════
# Operator-Flow Summary Table (bottom strip)
# ═════════════════════════════════════════════════════════════════════════

def _draw_table(ax: plt.Axes, bf16: dict, fp8: dict) -> None:
    """Render phase-by-phase comparison table as a matplotlib table widget."""
    ax.axis("off")
    n_ph = len(bf16["board"])

    headers = ["Phase", "Operator", "BF16 Reads", "BF16 Writes",
               "FP8 Reads", "FP8 Writes", "BF16\nMiB", "FP8\nMiB", "Δ MiB"]

    def _abbrev(names: list[str], bufs: dict, min_mib: float = 0.5) -> str:
        major = [n for n in names if n in bufs and bufs[n]["mib"] >= min_mib]
        minor = len(names) - len(major)
        s = ", ".join(major[:4])
        if len(major) > 4:
            s += f" +{len(major)-4}"
        if minor > 0:
            s += f" (+{minor})"
        return s or "—"

    rows = []
    for j in range(n_ph):
        bph = bf16["board"][j]
        fph = fp8["board"][j]
        bop = bf16["ops"][j]
        fop = fp8["ops"][j]
        b_mib = bph["alloc_mib"]
        f_mib = fph["alloc_mib"]
        delta = f_mib - b_mib
        d_str = f"{delta:+.0f}" if abs(delta) > 0.5 else "0"

        rows.append([
            f"P{j}", bph["name"],
            _abbrev(bop["R"], bf16["bufs"]),
            _abbrev(bop["W"], bf16["bufs"]),
            _abbrev(fop["R"], fp8["bufs"]),
            _abbrev(fop["W"], fp8["bufs"]),
            f"{b_mib:.0f}", f"{f_mib:.0f}", d_str,
        ])

    tbl = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        colColours=["#E3F2FD"] * len(headers),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(F["tbl"] + 0.5)
    tbl.scale(1.0, 1.8)

    # Style header
    for j in range(len(headers)):
        cell = tbl[0, j]
        cell.set_text_props(fontweight="bold", fontsize=F["tbl"])
        cell.set_facecolor("#BBDEFB")
        cell.set_edgecolor("#90CAF9")

    # Colour delta column
    for i in range(n_ph):
        cell = tbl[i + 1, 8]
        val = float(rows[i][8].replace("+", "")) if rows[i][8] != "0" else 0
        if val > 10:
            cell.set_facecolor("#FFCDD2")
        elif val < -10:
            cell.set_facecolor("#C8E6C9")

    ax.set_title("Operator-Flow Summary  ·  Phase-by-Phase Read/Write Sets & Memory",
                 fontsize=F["title"], fontweight="bold", pad=6)


# ═════════════════════════════════════════════════════════════════════════
# Composite Figure
# ═════════════════════════════════════════════════════════════════════════

def render() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sb = _load()
    bf16, fp8 = sb["bf16"], sb["fp8"]
    blist = _buf_list(bf16, fp8, min_mib=0.5)

    fig = plt.figure(figsize=(26, 20), dpi=150)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3.5, 1.2],
        width_ratios=[1, 1],
        hspace=0.15, wspace=0.06,
        left=0.14, right=0.99, top=0.93, bottom=0.03,
    )

    # ── Row 1: Twin Gantt panels ──
    ax_b = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1])

    pk_b = bf16["peak"]["alloc_mib"]
    pk_f = fp8["peak"]["alloc_mib"]
    chains_b = _draw_panel(ax_b, bf16, blist,
                f"BF16 Baseline  ·  {len(bf16['bufs'])} bufs  ·  "
                f"peak {pk_b:.0f} MiB @ P{bf16['peak']['phase']}",
                mode_key="bf16", show_ylabels=True)
    chains_f = _draw_panel(ax_f, fp8, blist,
                f"FP8 Frontier  ·  {len(fp8['bufs'])} bufs  ·  "
                f"peak {pk_f:.0f} MiB @ P{fp8['peak']['phase']}",
                mode_key="fp8", show_ylabels=False)

    # Shared legend (R/W/L states + buffer class colours)
    leg = [
        mpatches.Patch(fc=_S["R"], label="Read (R)"),
        mpatches.Patch(fc=_S["W"], label="Write (W)"),
        mpatches.Patch(fc=_S["L"], alpha=0.28, label="Live (idle)"),
    ]
    for cls in ["data", "weight", "grad", "quant", "meta"]:
        leg.append(mpatches.Patch(fc="white", ec=_C[cls], lw=2.5,
                                  label=cls.title()))
    ax_b.legend(handles=leg, fontsize=F["leg"], loc="lower left",
                ncol=2, framealpha=0.92, handlelength=1.4,
                handletextpad=0.4, columnspacing=0.8,
                bbox_to_anchor=(0.0, 0.0))

    # ── Row 2: Operator-flow table (full width) ──
    ax_tbl = fig.add_subplot(gs[1, :])
    _draw_table(ax_tbl, bf16, fp8)

    # ── Suptitle ──
    delta = pk_f - pk_b
    fig.suptitle(
        f"SonicMoE Unified Buffer Lifecycle × Scoreboard\n"
        f"reference-shape (T=8192, H=3072, I=1536, E=K=8)  ·  "
        f"BF16 peak {pk_b:.0f} → FP8 peak {pk_f:.0f} MiB "
        f"({delta:+.0f} MiB, {delta/pk_b*100:+.1f}%)",
        fontsize=F["sup"], fontweight="bold", y=0.98,
    )

    out = ASSETS / "scoreboard_unified.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")

    # ── Print agent-readable summary table to stdout ──
    _print_table(bf16, fp8)


def _print_table(bf16: dict, fp8: dict) -> None:
    """Print compact comparison table for agent/human consumption."""
    hdr = (f"{'Ph':<4} {'Operator':<18} {'BF16':>6} {'FP8':>6} {'Δ':>6}  "
           f"BF16 R→W  /  FP8 R→W")
    sep = "─" * 105
    print(f"\n{sep}\n{hdr}\n{sep}")
    for j, (bph, fph) in enumerate(zip(bf16["board"], fp8["board"])):
        bop, fop = bf16["ops"][j], fp8["ops"][j]
        bm, fm = bph["alloc_mib"], fph["alloc_mib"]
        d = fm - bm
        def _sig(names, bufs):
            return ",".join(n for n in names if bufs.get(n, {}).get("mib", 0) >= 0.5)
        br = _sig(bop["R"], bf16["bufs"])
        bw = _sig(bop["W"], bf16["bufs"])
        fr = _sig(fop["R"], fp8["bufs"])
        fw = _sig(fop["W"], fp8["bufs"])
        print(f"P{j:<3} {bph['name']:<18} {bm:>5.0f}  {fm:>5.0f}  "
              f"{d:>+5.0f}  {br}→{bw}  /  {fr}→{fw}")
    print(f"{sep}\n")


if __name__ == "__main__":
    render()
