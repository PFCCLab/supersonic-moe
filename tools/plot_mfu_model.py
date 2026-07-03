#!/usr/bin/env python3
"""MFU performance model visualization tool.

Generates publication-quality figures from nsys GPU-projection sweep data.
Model: gpu_proj_us = α·TK·H·I + β·TK·max(H,2I)·ln(TK)/1e9 + γ·E + δ

Usage: python tools/plot_mfu_model.py
"""
import os
import sys
import csv
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(_REPO, "reports", "perf_sweep_nsys_8gpu.csv")
OUT_DIR = os.path.join(_REPO, "reports")

# ═══ MODEL PARAMETERS (global DE optimization, 94 points, MAPE=3.95%) ═══
# gpu_proj_us = α·TK·H·I + β·TK·max(H,2I)·ln(1+TK/TK₀) + γ·E
#
# Physical basis:
#   α: per-FLOP GPU time (TC compute + L2-warm quant), η_tc = 53.4%
#   β: L2 cache miss penalty per byte of working set when TK > TK₀
#   TK₀: L2 transition scale (≈ L2_per_expert / tile_bytes × tile_M)
#   γ: per-expert fixed cost (weight TMA descriptor + metadata kernel)
ALPHA = 7.491431e-9    # µs per TK·H·I (η_tc = 53.4%)
BETA = 9.519563e-7     # L2 miss penalty coefficient
TK0 = 39747.0          # L2 transition scale
GAMMA = 21.72          # per-expert fixed (µs)
PEAK_TFLOPS = 4500     # FP8 peak (TFLOPS)


def gpu_proj_us(TK, H, I, E):
    """Predict GPU-projection time (µs) for fwd+bwd."""
    maxHI = max(H, 2*I)
    return ALPHA * TK * H * I + BETA * TK * maxHI * np.log(1.0 + TK / TK0) + GAMMA * E


def mfu_pct(TK, H, I, E):
    """Predict MFU (%)."""
    proj = gpu_proj_us(TK, H, I, E)
    if proj <= 0:
        return 0.0
    return 18.0 * TK * H * I / (PEAK_TFLOPS * 1e6 * proj) * 100.0


def mfu_inf(H, I):
    """Approximate steady-state MFU at TK=4M."""
    return mfu_pct(4194304, H, I, 8)


def load_data():
    """Load sweep data from CSV."""
    data = []
    with open(DATA_CSV) as f:
        for row in csv.DictReader(f):
            if row['status'] == 'OK' and float(row['gpu_projection_us']) > 0:
                d = {
                    'E': int(row['E']), 'TK': int(row['TK']),
                    'H': int(row['H']), 'I': int(row['I']),
                    'proj_us': float(row['gpu_projection_us']),
                }
                d['mfu'] = 18*d['TK']*d['H']*d['I'] / (d['proj_us']*1e-6) / (PEAK_TFLOPS*1e12) * 100
                data.append(d)
    # Deduplicate
    seen = set()
    unique = []
    for d in data:
        k = (d['E'], d['TK'], d['H'], d['I'])
        if k not in seen:
            seen.add(k)
            unique.append(d)
    return unique


def plot_mfu_vs_tk(data, ax):
    """Left panel: MFU vs TK for multiple (E, H, I) configs with model curves."""
    TK_line = np.logspace(3.9, 6.9, 300)

    configs = [
        (8, 3072, 1536, '#1f77b4', 'o', 'E=8 H=3072 I=1536 (reference)'),
        (8, 4096, 2048, '#2ca02c', 's', 'E=8 H=4096 I=2048 (Qwen3)'),
        (8, 4096, 4096, '#ff7f0e', '^', 'E=8 H=4096 I=4096 (peak)'),
        (8, 6144, 3072, '#d62728', 'D', 'E=8 H=6144 I=3072 (70B)'),
        (32, 3072, 1536, '#9467bd', 'v', 'E=32 H=3072 I=1536'),
        (128, 3072, 1536, '#8c564b', 'P', 'E=128 H=3072 I=1536'),
    ]

    for E_val, H_val, I_val, color, marker, label in configs:
        # Model curve
        mfu_curve = np.array([mfu_pct(tk, H_val, I_val, E_val) for tk in TK_line])
        ax.semilogx(TK_line, mfu_curve, color=color, lw=2, zorder=3)

        # Peak marker
        idx = np.argmax(mfu_curve)
        ax.plot(TK_line[idx], mfu_curve[idx], '*', color=color, ms=14,
                zorder=7, markeredgecolor='k', markeredgewidth=0.5)

        # Data points
        pts = [d for d in data if d['E']==E_val and d['H']==H_val and d['I']==I_val]
        if pts:
            ax.scatter([d['TK'] for d in pts], [d['mfu'] for d in pts],
                      marker=marker, c=color, s=55, zorder=5, edgecolors='k',
                      lw=0.5, label=label)

    # Annotations
    ax.axvspan(40000, 200000, alpha=0.06, color='#2ecc71', zorder=1)
    ax.text(85000, 2, 'reference optimal', fontsize=8, color='#27ae60', ha='center', style='italic')
    ax.text(5e6, 48, 'L2 thrashing →', fontsize=8, color='gray', style='italic')

    ax.set_xlabel('TK (token-expert pairs)', fontsize=11)
    ax.set_ylabel('MFU (%)', fontsize=11)
    ax.set_title('MFU vs TK: Model Curves + Measured (★=peak)', fontsize=11)
    ax.legend(loc='lower right', fontsize=8, ncol=1, framealpha=0.9)
    ax.set_ylim(0, 58)
    ax.set_xlim(6000, 8e6)
    ax.grid(True, alpha=0.2)
    ax.set_yticks(np.arange(0, 60, 5))


def plot_contour(data, ax):
    """Contour map of MFU(TK, E) with peak curve."""
    E_fine = np.linspace(8, 128, 60)
    TK_fine = np.logspace(4, 6.85, 60)
    MFU_grid = np.zeros((len(TK_fine), len(E_fine)))
    for i, tk in enumerate(TK_fine):
        for j, e in enumerate(E_fine):
            MFU_grid[i, j] = mfu_pct(tk, 3072, 1536, e)

    E_mesh, TK_mesh = np.meshgrid(E_fine, TK_fine)

    # Filled contour
    cs = ax.contourf(np.log10(TK_mesh), E_mesh, MFU_grid,
                     levels=np.arange(5, 47, 2), cmap='RdYlGn', extend='both')
    plt.colorbar(cs, ax=ax, label='MFU (%)')

    # Iso-lines
    cs2 = ax.contour(np.log10(TK_mesh), E_mesh, MFU_grid,
                     levels=[15, 25, 35, 40, 43, 45], colors='white',
                     linewidths=1, linestyles='--')
    ax.clabel(cs2, fmt='%d%%', fontsize=8, colors='white')

    # Peak TK* curve
    peak_tks = []
    for e_val in E_fine:
        mfus = [mfu_pct(tk, 3072, 1536, e_val) for tk in TK_fine]
        peak_tks.append(TK_fine[np.argmax(mfus)])
    ax.plot(np.log10(peak_tks), E_fine, 'r-', lw=3, label='TK* (peak MFU)', zorder=6)

    # Data points
    pts = [d for d in data if d['H'] == 3072 and d['I'] == 1536]
    ax.scatter([np.log10(d['TK']) for d in pts], [d['E'] for d in pts],
              c=[d['mfu'] for d in pts], cmap='RdYlGn', s=45,
              edgecolors='white', lw=1, zorder=7, vmin=5, vmax=47)

    ax.set_xlabel('log₁₀(TK)', fontsize=11)
    ax.set_ylabel('E (experts)', fontsize=11)
    ax.set_title('MFU Contour + TK* Curve (H=3072, I=1536)', fontsize=11)
    ax.set_yticks([8, 16, 32, 64, 128])
    ax.legend(loc='upper left', fontsize=10)


def main():
    import matplotlib
    matplotlib.use('Agg')
    global plt
    import matplotlib.pyplot as plt

    data = load_data()
    print(f"Loaded {len(data)} data points")

    # Verify model
    proj_reference = gpu_proj_us(65536, 3072, 1536, 8)
    mfu_reference = mfu_pct(65536, 3072, 1536, 8)
    print(f"Model check (reference): proj={proj_reference:.0f}µs, MFU={mfu_reference:.1f}% (measured: 2715µs, 45.6%)")

    # ═══ Figure: 3×2 grid ═══
    # Top row: reference shape (H=3072, I=1536) — curve + contour
    # Middle/Bottom rows: other HI configs
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    HI_configs = [
        (3072, 1536, r"$H{=}3072,\; I{=}1536$ (reference)"),
        (4096, 2048, r"$H{=}4096,\; I{=}2048$"),
        (4096, 4096, r"$H{=}4096,\; I{=}4096$"),
    ]

    for row, (H_val, I_val, title_prefix) in enumerate(HI_configs):
        # ── Left: MFU vs TK curves for different E ──
        ax = axes[row, 0]
        TK_line = np.logspace(3.9, 6.9, 300)
        e_colors = {8: '#1f77b4', 32: '#ff7f0e', 64: '#d62728', 128: '#9467bd'}
        e_markers = {8: 'o', 32: '^', 64: 'D', 128: 'v'}

        # Compute all curves first to determine ylim
        all_mfus = []
        for e_val in [8, 32, 64, 128]:
            mfu_curve = np.array([mfu_pct(tk, H_val, I_val, e_val) for tk in TK_line])
            all_mfus.append(mfu_curve)
            ax.semilogx(TK_line, mfu_curve, color=e_colors[e_val], lw=2.5,
                       label=f'E={e_val}', zorder=3)
            # Peak star
            idx = np.argmax(mfu_curve)
            ax.plot(TK_line[idx], mfu_curve[idx], '*', color=e_colors[e_val],
                   ms=14, zorder=7, markeredgecolor='k', markeredgewidth=0.5)
            # Data points
            pts = [d for d in data if d['E'] == e_val and d['H'] == H_val and d['I'] == I_val]
            if pts:
                ax.scatter([d['TK'] for d in pts], [d['mfu'] for d in pts],
                          marker=e_markers[e_val], c=e_colors[e_val], s=55,
                          zorder=5, edgecolors='k', lw=0.5)

        # Dynamic ylim based on actual data range
        max_mfu_curve = max(np.max(m) for m in all_mfus)
        ax.set_ylim(0, min(max_mfu_curve + 8, 75))

        # MFU∞ asymptote line + measured peak annotation
        e8_pts = [d for d in data if d['E']==8 and d['H']==H_val and d['I']==I_val]
        if e8_pts:
            real_peak = max(e8_pts, key=lambda d: d['mfu'])
            mfu_steady = mfu_pct(4000000, H_val, I_val, 8)
            ax.axhline(y=mfu_steady, color='gray', linestyle=':', lw=1.5, alpha=0.7, zorder=2)
            ax.text(4e6, mfu_steady + 1.5, f'steady≈{mfu_steady:.0f}%', fontsize=8,
                   color='gray', style='italic', ha='right')
            # Peak info in lower-left (axes coords, no overlap with curves)
            ax.text(0.03, 0.05,
                    f'Measured peak: {real_peak["mfu"]:.1f}% @ TK={real_peak["TK"]/1000:.0f}K\n'
                    f'Steady state: ≈{mfu_steady:.0f}% (TK>2M)',
                    fontsize=9, color='#2c3e50',
                    transform=ax.transAxes, ha='left', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.9))

        ax.set_xlabel('TK (token-expert pairs)', fontsize=11)
        ax.set_ylabel('MFU (%)', fontsize=11)
        ax.set_title(f'{title_prefix}\nMFU vs TK (★=peak)', fontsize=11)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(6000, 8e6)
        ax.grid(True, alpha=0.2)

        # ── Right: Contour for this (H, I) — dynamic levels ──
        ax = axes[row, 1]
        E_fine = np.linspace(8, 128, 60)
        TK_fine = np.logspace(4, 6.85, 60)
        MFU_grid = np.zeros((len(TK_fine), len(E_fine)))
        for i, tk in enumerate(TK_fine):
            for j, e in enumerate(E_fine):
                MFU_grid[i, j] = mfu_pct(tk, H_val, I_val, e)

        E_mesh, TK_mesh = np.meshgrid(E_fine, TK_fine)

        # Dynamic contour levels based on this shape's MFU range
        mfu_min = MFU_grid.min()
        mfu_max = MFU_grid.max()
        levels = np.linspace(max(mfu_min, 5), min(mfu_max, 75), 20)

        cs = ax.contourf(np.log10(TK_mesh), E_mesh, MFU_grid,
                         levels=levels, cmap='RdYlGn', extend='both')
        plt.colorbar(cs, ax=ax, label='MFU (%)')

        # Iso-lines at meaningful percentages
        iso_levels = [l for l in [15, 25, 35, 40, 45, 50, 55, 60, 65]
                      if mfu_min < l < mfu_max]
        if iso_levels:
            cs2 = ax.contour(np.log10(TK_mesh), E_mesh, MFU_grid,
                             levels=iso_levels, colors='white', linewidths=1, linestyles='--')
            ax.clabel(cs2, fmt='%d%%', fontsize=8, colors='white')

        # Peak TK* curve
        peak_tks = []
        for e_val in E_fine:
            mfus = [mfu_pct(tk, H_val, I_val, e_val) for tk in TK_fine]
            peak_tks.append(TK_fine[np.argmax(mfus)])
        ax.plot(np.log10(peak_tks), E_fine, 'r-', lw=3, label='TK* (peak)', zorder=6)

        # Data
        pts = [d for d in data if d['H'] == H_val and d['I'] == I_val]
        if pts:
            ax.scatter([np.log10(d['TK']) for d in pts], [d['E'] for d in pts],
                      c=[d['mfu'] for d in pts], cmap='RdYlGn', s=45,
                      edgecolors='white', lw=1, zorder=7,
                      vmin=levels[0], vmax=levels[-1])

        ax.set_xlabel('log₁₀(TK)', fontsize=11)
        ax.set_ylabel('E (experts)', fontsize=11)
        ax.set_title(f'{title_prefix}\nMFU Contour + TK* Curve', fontsize=11)
        ax.set_yticks([8, 16, 32, 64, 128])
        ax.legend(loc='upper left', fontsize=8)

    # Layout: ample spacing to prevent any title/formula overlap
    plt.subplots_adjust(top=0.94, bottom=0.10, hspace=0.32, wspace=0.28)

    # Short descriptive title at top (no overlap risk with 3 rows)
    fig.suptitle('GPU-Projection Performance Model: MFU vs TK and Expert Count',
                 fontsize=13, y=0.98, va='top', weight='bold')

    # Full formula as footer caption (no competition with subplot titles)
    fig.text(0.50, 0.005,
             r'$T_{\rm proj} = \alpha\cdot TK\cdot H\cdot I'
             r' + \beta\cdot TK\cdot \max(H,2I)\cdot \ln\!\left(1+\frac{TK}{TK_0}\right)'
             r' + \gamma\cdot E$'
             r'$\qquad|\qquad$'
             r'$\alpha=$' + f'{ALPHA:.3e}' + r'$, \beta=$' + f'{BETA:.3e}'
             + r'$, TK_0=$' + f'{TK0:.0f}' + r'$, \gamma=$' + f'{GAMMA:.1f}' + r'$\mu s$'
             r'$\qquad|\qquad$'
             r'MAPE = 3.79%, N = 104',
             fontsize=8.5, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f6fa', edgecolor='#bdc3c7', alpha=0.9))

    path1 = os.path.join(OUT_DIR, "mfu_model_nsys.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {path1}")

    # Also save standalone contour (reference only)
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 7))
    E_fine = np.linspace(8, 128, 60)
    TK_fine = np.logspace(4, 6.85, 60)
    MFU_grid = np.zeros((len(TK_fine), len(E_fine)))
    for i, tk in enumerate(TK_fine):
        for j, e in enumerate(E_fine):
            MFU_grid[i, j] = mfu_pct(tk, 3072, 1536, e)
    E_mesh, TK_mesh = np.meshgrid(E_fine, TK_fine)
    cs = ax2.contourf(np.log10(TK_mesh), E_mesh, MFU_grid,
                      levels=np.arange(5, 47, 2), cmap='RdYlGn', extend='both')
    plt.colorbar(cs, ax=ax2, label='MFU (%)')
    cs2 = ax2.contour(np.log10(TK_mesh), E_mesh, MFU_grid,
                      levels=[15, 25, 35, 40, 43, 45], colors='white', linewidths=1, linestyles='--')
    ax2.clabel(cs2, fmt='%d%%', fontsize=9, colors='white')
    peak_tks = []
    for e_val in E_fine:
        mfus = [mfu_pct(tk, 3072, 1536, e_val) for tk in TK_fine]
        peak_tks.append(TK_fine[np.argmax(mfus)])
    ax2.plot(np.log10(peak_tks), E_fine, 'r-', lw=3, label='TK* (peak MFU)', zorder=6)
    pts = [d for d in data if d['H'] == 3072 and d['I'] == 1536]
    ax2.scatter([np.log10(d['TK']) for d in pts], [d['E'] for d in pts],
               c=[d['mfu'] for d in pts], cmap='RdYlGn', s=50,
               edgecolors='white', lw=1, zorder=7, vmin=5, vmax=47)
    ax2.set_xlabel('log₁₀(TK)', fontsize=12)
    ax2.set_ylabel('E (experts)', fontsize=12)
    ax2.set_title('MFU Contour + TK* (Reference: H=3072, I=1536)', fontsize=13)
    ax2.set_yticks([8, 16, 32, 64, 128])
    ax2.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, "mfu_contour.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
