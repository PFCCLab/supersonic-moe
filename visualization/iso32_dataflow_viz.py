"""ISO32 vs Baseline FP8 dataflow comparison visualization.

Generates a single figure with two sub-panels showing:
- Left: Baseline pair-kernel path (4 FP8 buffers, contiguous)
- Right: ISO32 single-buffer path (2 FP8 buffers, stride views)

Key features:
- Forward→backward connection via dashed save_for_backward arrows
- Weight buffer allocation highlighted (shared vs separate)
- Operator names match actual CUTLASS kernel dispatch
- Memory accounting in bottom badge

Usage:
    source .runenv.sh
    python visualization/iso32_dataflow_viz.py
    # Output: reports/fig_iso32_dataflow_comparison.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'figure.dpi': 150})

C_ACT  = '#2E7D32'
C_W    = '#1565C0'
C_WI   = '#0D47A1'
C_GRAD = '#B71C1C'
C_OP   = '#ECEFF1'
C_SAVE = '#7B1FA2'
C_CTX  = '#F3E5F5'
C_BDR  = '#37474F'


def draw_panel(ax, is_iso32):
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 24)
    ax.axis('off')
    CW = C_WI if is_iso32 else C_W
    title = "ISO32 (2 buffers, stride views)" if is_iso32 else "Baseline Pair-Kernel (4 buffers)"
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=6, color=CW)

    def box(x, y, w, h, text, fc, tc='k', fs=7.5, lw=0.7):
        r = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.12",
                          fc=fc, ec=C_BDR, lw=lw, alpha=0.93, zorder=2)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, text, ha='center', va='center',
               fontsize=fs, color=tc, linespacing=1.25, zorder=3)

    def arr(x1, y1, x2, y2, c='#555', lw=0.9):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), zorder=1,
                   arrowprops=dict(arrowstyle='-|>', color=c, lw=lw))

    def darr(x1, y1, x2, y2, c=C_SAVE, lw=1.0):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), zorder=1,
                   arrowprops=dict(arrowstyle='-|>', color=c, lw=lw, linestyle='dashed'))

    def note(x, y, text, fs=6.5, c='#555', ha='left'):
        ax.text(x, y, text, fontsize=fs, color=c, ha=ha, va='center', style='italic')

    # ══ FORWARD ══
    ax.text(0.3, 23.3, "FORWARD", fontsize=9, fontweight='bold', color=C_ACT)
    box(0.5, 21.5, 2.5, 1.2, "x\n(T, H) bf16", C_ACT, 'white')

    if is_iso32:
        box(4.0, 21.5, 5.5, 1.2,
            "iso32_dual_quantize_weight_3d\nw1→fp8(E,2I,H) + row_sc + col_sc", CW, 'white', fs=7)
        note(4.0, 20.9, "ONE buffer, dual scales", fs=6, c=CW)
    else:
        box(4.0, 21.5, 5.5, 1.2,
            "_quantize_pair_kernel(w1)\n→ w1_fused(E,2I,H) + w1T(E,H,2I)", CW, 'white', fs=7)
        note(4.0, 20.9, "TWO separate fp8 buffers", fs=6, c=CW)

    box(11.0, 21.5, 4.5, 1.0, "quant_and_pack(x)\n→ x_fp8, x_scales", C_OP, fs=7)
    arr(3.0, 22.0, 11.0, 22.0, C_ACT)

    box(0.5, 18.5, 6.5, 1.8,
        "GemmGatedSm100ZeroMat (fused SwiGLU)\nA = x_fp8 (TK,H)\n"
        "B = w1_fp8" + (".mT view" if is_iso32 else " (contig)") + " + scales\n"
        "→ y1(TK,I) fp8,  z(TK,2I) fp8", C_OP, fs=7)
    arr(1.5, 21.5, 1.5, 20.3, C_ACT)
    arr(6.5, 21.5, 5.0, 20.3, CW, lw=1.2)

    if is_iso32:
        box(4.0, 16.0, 5.5, 1.2,
            "iso32_dual_quantize_weight_3d\nw2→fp8(E,H,I) + row_sc + col_sc", CW, 'white', fs=7)
    else:
        box(4.0, 16.0, 5.5, 1.2,
            "_quantize_pair_kernel(w2)\n→ w2_varlen(E,H,I) + w2_dgated(E,I,H)", CW, 'white', fs=7)

    box(0.5, 13.5, 6.5, 1.5,
        "blockscaled_fp8_gemm_varlen\nA = y1_fp8,  B = w2_fp8 + scales\n→ y2 (TK, H) bf16", C_OP, fs=7)
    arr(3.5, 18.5, 3.5, 15.0, C_ACT)
    arr(6.5, 16.0, 5.5, 15.0, CW, lw=1.2)

    box(0.5, 11.5, 5.0, 1.2,
        "_router_forward(y2, scores, scatter)\n→ output (T, H) bf16", C_OP, fs=7)
    arr(3.5, 13.5, 3.5, 12.7, C_ACT)

    # ctx box
    ctx_x = 12.0
    box(ctx_x, 14.0, 6.5, 5.5, "", C_CTX, lw=0.5)
    ax.text(ctx_x+3.25, 19.2, "save_for_backward (ctx)", fontsize=7.5,
           ha='center', fontweight='bold', color=C_SAVE)
    ctx_items = ["z_fp8, z_raw_scales", "expert_freq_offset",
                "x_gather_idx, scatter_idx", "scores (router bwd)"]
    if is_iso32:
        ctx_items += ["w1T = fp8_w1.permute(0,2,1) [SHARED]",
                     "w2T = fp8_w2.permute(0,2,1) [SHARED]"]
    else:
        ctx_items += ["w1T_fp8 (E,H,2I) [72 MiB COPY]",
                     "w2_dgated_fp8 (E,I,H) [36 MiB COPY]"]
    for i, item in enumerate(ctx_items):
        c = C_SAVE if "SHARED" in item or "COPY" in item else '#333'
        ax.text(ctx_x+0.3, 18.7-i*0.65, f"• {item}", fontsize=6.5, va='center', color=c)

    darr(7.0, 19.0, ctx_x, 18.5)
    darr(7.0, 14.5, ctx_x, 16.0)

    # ══ BACKWARD ══
    ax.text(0.3, 10.5, "BACKWARD", fontsize=9, fontweight='bold', color=C_GRAD)
    box(0.5, 8.8, 2.8, 1.0, "dout (T,H)\nbf16", C_GRAD, 'white')
    box(4.0, 8.5, 7.0, 1.5,
        "GemmDGatedFP8CLoad (CUTLASS)\ndout × w2" +
        ("_T_view" if is_iso32 else "_dgated_fp8") + " → dz, y1s", C_OP, fs=7)
    arr(3.3, 9.3, 4.0, 9.3, C_GRAD)
    darr(ctx_x, 14.0, 10.0, 10.0, C_SAVE)
    note(11.5, 10.3, "w2T from ctx" + ("\n(stride view)" if is_iso32 else "\n(36 MiB copy)"), fs=6, c=CW)

    box(0.5, 6.0, 5.5, 1.5,
        "fused_dual_colwise_quantize(dz)\n→ dz_row_fp8 + dz_col_fp8\n   (iso32: ONE buffer aliased)", C_OP, fs=6.5)
    arr(7.0, 8.5, 3.0, 7.5, C_GRAD)

    box(0.5, 3.5, 5.5, 1.5,
        "wgrad: blockscaled_gemm_varlen_k\ndz_col × x_col → dw1\ndout_col × y1s_col → dw2", C_OP, fs=6.5)
    box(7.5, 3.5, 6.5, 1.5,
        "actgrad: _run_cutlass_blockscaled_gemm\ndz_fp8 × w1T" +
        ("_view" if is_iso32 else "_fp8") + " → dx (T, H)\n"
        "leading_dim=" + ("auto" if is_iso32 else "1"), C_OP, fs=6.5)
    arr(3.0, 6.0, 3.0, 5.0, C_GRAD)
    arr(5.0, 7.0, 9.0, 5.0, C_GRAD)
    darr(ctx_x+3, 14.0, 10.0, 5.0, C_SAVE)
    note(14.5, 6.2, "w1T from ctx" + ("\n(stride view)" if is_iso32 else "\n(72 MiB copy)"), fs=6, c=CW)

    box(7.5, 1.5, 2.5, 1.0, "dx (T,H)\nbf16", C_GRAD, 'white')
    box(0.5, 1.5, 3.0, 1.0, "dw1, dw2\n(main_grad)", C_GRAD, 'white', fs=7)
    arr(10.5, 3.5, 8.75, 2.5, C_GRAD)
    arr(3.0, 3.5, 2.0, 2.5, C_GRAD)

    if is_iso32:
        box(15.5, 1.5, 3.0, 1.5, "Weight cache:\n108 MiB\n(−48.5%)", '#E8F5E9', CW, fs=7, lw=1.5)
    else:
        box(15.5, 1.5, 3.0, 1.5, "Weight cache:\n216 MiB\npersistent", '#FFEBEE', CW, fs=7, lw=1.5)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 11))
    fig.suptitle("SonicMoE FP8 Frontier: Forward–Backward Dataflow Comparison",
                fontsize=12, fontweight='bold', y=0.995)
    draw_panel(ax1, is_iso32=False)
    draw_panel(ax2, is_iso32=True)
    legend_elements = [
        mpatches.Patch(fc=C_ACT, ec=C_BDR, label='Activations'),
        mpatches.Patch(fc=C_W, ec=C_BDR, label='Weight FP8 (baseline)'),
        mpatches.Patch(fc=C_WI, ec=C_BDR, label='Weight FP8 (iso32)'),
        mpatches.Patch(fc=C_GRAD, ec=C_BDR, label='Gradients'),
        mpatches.Patch(fc=C_CTX, ec=C_SAVE, label='save_for_backward ctx'),
        mpatches.Patch(fc=C_OP, ec=C_BDR, label='CUTLASS/Triton Kernels'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=8,
              frameon=True, edgecolor='#ccc', fancybox=True, bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    out = Path(__file__).resolve().parent.parent / "reports" / "fig_iso32_dataflow_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
