"""ISO32 vs Baseline FP8 Weight Dataflow — Publication-quality v8.

Fixes from v7:
  - DeepEP pipeline: adds persistent side column (same style as weight figure)
  - prequantize clearly marked as "once/step, not per-forward"
  - Larger boxes in pipeline figure for readability
  - README update ready

Output:
  reports/fig_iso32_dataflow_comparison.png
  reports/fig_deepep_fp8_pipeline.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.5,
    'figure.dpi': 200,
    'axes.linewidth': 0,
})

C = {
    'act': '#1B5E20', 'act_f': '#E8F5E9',
    'wt': '#0D47A1', 'wt_f': '#E3F2FD',
    'wi': '#1A237E', 'wi_f': '#E8EAF6',
    'grad': '#B71C1C', 'grad_f': '#FFEBEE',
    'op': '#37474F', 'op_f': '#FAFAFA', 'op_b': '#90A4AE',
    'ctx': '#6A1B9A', 'ctx_f': '#F3E5F5',
    'mg': '#E65100', 'mg_f': '#FFF3E0',
    'ds': '#00695C', 'ds_f': '#E0F2F1',
    'elim': '#9E9E9E', 'elim_f': '#FAFAFA',
    'save': '#2E7D32',
    'comm': '#4A148C', 'comm_f': '#EDE7F6',
    'bdr': '#546E7A', 'div': '#B0BEC5', 'note': '#616161',
}


def _rbox(ax, cx, cy, w, h, fc, ec, lw=0.8, zorder=3, alpha=0.95, ls='solid'):
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=lw, zorder=zorder, alpha=alpha, linestyle=ls
    )
    ax.add_patch(patch)


def _arr(ax, x1, y1, x2, y2, color='#555', lw=0.8, dashed=False):
    ls = (0, (3.5, 2.5)) if dashed else 'solid'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=2,
               arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, linestyle=ls))


def _curv(ax, x1, y1, x2, y2, color, lw=0.7, dashed=False, rad=0.25):
    ls = (0, (3.5, 2.5)) if dashed else 'solid'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=2,
               arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                              connectionstyle=f'arc3,rad={rad}', linestyle=ls))


# ═══════════════════════════════════════════════════════════════════════════════
# Shared: Side column drawing (persistent cache + ctx)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_side_column(ax, sc_x, sc_w, sc_top, sc_bot, sep_y, is_iso32):
    wc = C['wi'] if is_iso32 else C['wt']
    wf = C['wi_f'] if is_iso32 else C['wt_f']
    flow_center = (sc_top + sc_bot) / 2

    _rbox(ax, sc_x, flow_center, sc_w, sc_top - sc_bot, wf, wc, lw=1.2, zorder=1, alpha=0.4)
    ax.plot([sc_x - sc_w/2 + 0.4, sc_x + sc_w/2 - 0.4], [sep_y, sep_y],
           color=wc, lw=0.6, ls=':', alpha=0.7, zorder=4)

    # Upper: Persistent cache
    ax.text(sc_x, sc_top - 0.5, "Persistent Weight FP8 Cache", ha='center', va='top',
           fontsize=7.5, color=wc, fontweight='bold', zorder=4)
    ax.text(sc_x, sc_top - 1.2, "(once per step, not per forward)", ha='center', va='top',
           fontsize=5.5, color=C['note'], fontstyle='italic', zorder=4)

    buf_w = sc_w - 0.8
    buf_h = 0.7

    if is_iso32:
        bufs = [
            ("w1_fp8 (E,2I,H) + dual_sc", True, "E·2IH B"),
            ("w2_fp8 (E,H,I) + dual_sc", True, "E·HI B"),
            ("w1T  ELIMINATED (→ .permute view)", False, "saves E·2IH"),
            ("w2_dgated  ELIMINATED (→ .permute view)", False, "saves E·IH"),
        ]
    else:
        bufs = [
            ("w1_fused_fp8 (E,2I,H) + sc", True, "E·2IH B"),
            ("w1T_fp8 (E,H,2I) + sc", True, "E·2IH B"),
            ("w2_fp8 (E,H,I) + sc", True, "E·HI B"),
            ("w2_dgated_fp8 (E,I,H) + sc", True, "E·IH B"),
        ]

    y_cur = sc_top - 2.2
    for (label, is_solid, mem) in bufs:
        if is_solid:
            _rbox(ax, sc_x, y_cur, buf_w, buf_h, 'white', wc, lw=0.7, zorder=3)
            ax.text(sc_x - buf_w/2 + 0.2, y_cur, label, fontsize=5.8, color=wc,
                   fontweight='bold', va='center', zorder=4)
            ax.text(sc_x + buf_w/2 - 0.2, y_cur, mem, fontsize=5.2, color=C['note'],
                   ha='right', va='center', fontstyle='italic', zorder=4)
        else:
            _rbox(ax, sc_x, y_cur, buf_w, buf_h, C['elim_f'], C['elim'],
                  lw=0.8, zorder=3, ls=(0, (3, 2)))
            ax.text(sc_x - buf_w/2 + 0.2, y_cur, label, fontsize=5.5, color=C['elim'],
                   va='center', zorder=4)
            ax.text(sc_x + buf_w/2 - 0.2, y_cur, mem, fontsize=5.5, color=C['save'],
                   ha='right', va='center', fontweight='bold', zorder=4)
            ax.plot([sc_x - buf_w/2 + 0.15, sc_x + buf_w/2 - 0.15], [y_cur, y_cur],
                   color=C['elim'], lw=0.7, zorder=5, alpha=0.4)
        y_cur -= 0.95

    # Total
    if is_iso32:
        ax.text(sc_x, sep_y + 1.4, "Net: E·(2IH+HI)+sc ≈ 108 MiB", ha='center',
               va='center', fontsize=6.5, color=wc, fontweight='bold', zorder=4)
        ax.text(sc_x, sep_y + 0.6, "▼ Saves 48.5% (108 MiB)", ha='center',
               va='center', fontsize=6, color=C['save'], fontweight='bold', zorder=4)
    else:
        ax.text(sc_x, sep_y + 0.8, "Total: E·6HI+sc ≈ 216 MiB", ha='center',
               va='center', fontsize=6.5, color=wc, fontweight='bold', zorder=4)

    # Lower: ctx
    ax.text(sc_x, sep_y - 0.7, "ctx (save_for_backward)", ha='center', va='top',
           fontsize=6.8, color=C['ctx'], fontweight='bold', zorder=4)

    ctx_items = [
        ("z_fp8 + z_raw_scales", "TK·2I B"),
        ("scores (topk_scores)", "4·TK B"),
        ("expert_freq_offset, indices", ""),
    ]
    if is_iso32:
        ctx_items += [
            ("w1T = w1_fp8.permute [view]", "0 B ✓"),
            ("w2T = w2_fp8.permute [view]", "0 B ✓"),
        ]
    else:
        ctx_items += [
            ("w1T_fp8 [ref → cache]", ""),
            ("w2_dgated_fp8 [ref → cache]", ""),
        ]

    y_cur = sep_y - 1.5
    for (name, mem) in ctx_items:
        c = C['ctx'] if 'view' in name or 'ref' in name else '#444'
        ax.text(sc_x - sc_w/2 + 0.4, y_cur, name, fontsize=5.8, color=c, va='center', zorder=4)
        if mem:
            ax.text(sc_x + sc_w/2 - 0.3, y_cur, mem, fontsize=5.2,
                   color=C['save'] if '✓' in mem else C['note'],
                   ha='right', va='center', zorder=4)
        y_cur -= 0.7


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Weight-level dataflow
# ═══════════════════════════════════════════════════════════════════════════════

class WeightPanel:
    def __init__(self, ax, is_iso32):
        self.ax = ax
        self.iso = is_iso32
        self.W, self.H = 28, 40
        ax.set_xlim(-0.3, self.W + 0.3)
        ax.set_ylim(-1.5, self.H + 0.5)
        ax.axis('off')
        ax.set_aspect('equal')
        self.wc = C['wi'] if is_iso32 else C['wt']

    def tensor(self, cx, cy, name, formula, color, fill, w=5.0, h=1.0):
        _rbox(self.ax, cx, cy, w, h, fill, color, lw=0.9)
        self.ax.text(cx, cy + 0.18, name, ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold', zorder=4)
        self.ax.text(cx, cy - 0.22, formula, ha='center', va='center',
                    fontsize=5.8, color=color, fontstyle='italic', alpha=0.85, zorder=4)

    def op(self, cx, cy, w, h, lines, fs=6.5):
        _rbox(self.ax, cx, cy, w, h, C['op_f'], C['op_b'])
        n = len(lines)
        for i, ln in enumerate(lines):
            yy = cy + (n-1)*0.30/2 - i*0.30 if n > 1 else cy
            fw = 'bold' if i == 0 else 'normal'
            self.ax.text(cx, yy, ln, ha='center', va='center',
                        fontsize=fs, color=C['op'], fontweight=fw, zorder=4)

    def note(self, x, y, text, color=None, fs=5.8, ha='left'):
        self.ax.text(x, y, text, fontsize=fs, color=color or C['note'],
                    ha=ha, va='center', fontstyle='italic', zorder=5)

    def section(self, x, y, text, color):
        self.ax.text(x, y, text, fontsize=9, fontweight='bold', color=color, va='center')

    def draw(self):
        ax = self.ax
        iso = self.iso
        wc = self.wc

        title = "ISO32 — Single Buffer + Stride Views" if iso else "Baseline — Pair-Kernel (4 Buffers)"
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10, color=wc)

        # Side column
        sc_x, sc_w = 22.5, 8.5
        sc_top, sc_bot = 31.0, 13.0
        sep_y = 22.0
        draw_side_column(ax, sc_x, sc_w, sc_top, sc_bot, sep_y, iso)

        # Forward
        self.section(0.2, 39.0, "FORWARD", C['act'])
        mx = 7.0

        self.tensor(mx, 37.8, "x (input)", "T×H  bf16", C['act'], C['act_f'])
        self.op(mx, 36.5, 6.0, 0.7, ["TopK Router → dispatch"], fs=6.5)
        _arr(ax, mx, 37.8-0.5, mx, 36.5+0.35, C['act'])

        self.tensor(4.5, 35.2, "x_gathered", "TK×H  bf16", C['act'], C['act_f'], w=4.3)
        self.tensor(10.5, 35.2, "scores (s)", "TK  fp32", C['ds'], C['ds_f'], w=3.8)
        _arr(ax, mx-1.0, 36.5-0.35, 4.5, 35.2+0.5, C['act'])
        _arr(ax, mx+2.0, 36.5-0.35, 10.5, 35.2+0.5, C['ds'])

        self.op(4.5, 33.9, 5.8, 0.65, ["quant → x_fp8 + scales"], fs=6.2)
        _arr(ax, 4.5, 35.2-0.5, 4.5, 33.9+0.33, C['act'])

        g1y = 32.4
        g1t = "x_fp8 × w1.mT [view]" if iso else "x_fp8 × w1_fused"
        self.op(mx, g1y, 10.5, 0.9, ["GemmGatedSm100 (fused SwiGLU)", g1t], fs=6.2)
        _arr(ax, 4.5, 33.9-0.33, mx-2.5, g1y+0.45, C['act'])
        _arr(ax, sc_x-sc_w/2, sc_top-2.5, mx+5.2, g1y+0.1, wc, lw=0.9)

        self.tensor(3.5, 30.7, "z (pre-act)", "TK×2I  fp8", C['act'], C['act_f'], w=4.3)
        self.tensor(10.5, 30.7, "y1 (gated)", "TK×I  fp8", C['act'], C['act_f'], w=4.0)
        _arr(ax, mx-2.5, g1y-0.45, 3.5, 30.7+0.5, C['act'])
        _arr(ax, mx+2.5, g1y-0.45, 10.5, 30.7+0.5, C['act'])

        g2y = 29.0
        g2t = "y1_fp8 × w2 [direct]" if iso else "y1_fp8 × w2_fp8"
        self.op(mx, g2y, 10.5, 0.9, ["blockscaled_fp8_gemm_varlen", g2t], fs=6.2)
        _arr(ax, 10.5, 30.7-0.5, mx+2.0, g2y+0.45, C['act'])
        _arr(ax, sc_x-sc_w/2, sc_top-4.2, mx+5.2, g2y+0.1, wc, lw=0.9)

        self.tensor(mx, 27.3, "y2", "TK×H  bf16", C['act'], C['act_f'], w=3.8)
        _arr(ax, mx, g2y-0.45, mx, 27.3+0.5, C['act'])

        self.op(mx, 26.1, 7.5, 0.7, ["router_fwd: y2·s → scatter → out"], fs=6.2)
        _arr(ax, mx, 27.3-0.5, mx, 26.1+0.35, C['act'])
        _curv(ax, 10.5, 35.2-0.5, mx+3.2, 26.1+0.35, C['ds'], lw=0.5, dashed=True, rad=0.04)

        self.tensor(mx, 24.9, "output", "T×H  bf16", C['act'], C['act_f'], w=3.8)
        _arr(ax, mx, 26.1-0.35, mx, 24.9+0.5, C['act'])
        _curv(ax, 3.5, 30.7-0.5, sc_x-sc_w/2, sep_y-2.5, C['ctx'], lw=0.5, dashed=True, rad=-0.05)

        # Divider
        ax.axhline(23.2, color=C['div'], lw=0.7, ls='--', zorder=1, xmin=0.01, xmax=0.58)

        # Backward
        self.section(0.2, 22.3, "BACKWARD", C['grad'])
        self.tensor(mx, 21.2, "grad_output", "T×H  bf16", C['grad'], C['grad_f'], w=5.0)
        self.op(mx, 19.9, 7.5, 0.65, ["gather + quant → dout_fp8 (TK,H)"], fs=6.2)
        _arr(ax, mx, 21.2-0.5, mx, 19.9+0.33, C['grad'])

        g3y = 18.3
        g3t = "dout × w2T [view] ⊗ z ⊗ s" if iso else "dout × w2_dgated [copy] ⊗ z ⊗ s"
        self.op(mx, g3y, 12.0, 0.9, ["GemmDGated (fused dgrad+act+score bwd)", g3t], fs=6)
        _arr(ax, mx, 19.9-0.33, mx, g3y+0.45, C['grad'])
        _arr(ax, sc_x-sc_w/2, sep_y-4.5, mx+6.0, g3y+0.1, C['ctx'], lw=0.6, dashed=True)

        out_y = 16.3
        self.tensor(3.0, out_y, "dz", "TK×2I", C['grad'], C['grad_f'], w=3.0)
        self.tensor(7.5, out_y, "y1s", "TK×I", C['grad'], C['grad_f'], w=2.8)
        self.tensor(12.5, out_y, "ds", "TK fp32", C['ds'], C['ds_f'], w=3.0)
        _arr(ax, mx-3.0, g3y-0.45, 3.0, out_y+0.5, C['grad'])
        _arr(ax, mx, g3y-0.45, 7.5, out_y+0.5, C['grad'])
        _arr(ax, mx+3.8, g3y-0.45, 12.5, out_y+0.5, C['ds'])
        self.note(12.5, out_y-0.65, "→ router autograd", color=C['ds'], fs=5.5, ha='center')

        qy = 14.7
        self.op(5.5, qy, 9.0, 0.75, ["fused_dual_colwise_quantize(dz, dout)", "→ dz_fp8 + dz_col + dout_col"], fs=5.8)
        _arr(ax, 3.0, out_y-0.5, 4.0, qy+0.38, C['grad'])

        fork_y = 13.5
        ax.plot([0.5, 14.5], [fork_y, fork_y], color=C['div'], lw=0.4, alpha=0.5, zorder=1)
        ax.text(3.2, fork_y+0.22, "dw path", fontsize=5.8, ha='center', color=C['mg'], fontweight='bold')
        ax.text(11.5, fork_y+0.22, "dx path", fontsize=5.8, ha='center', color=C['grad'], fontweight='bold')

        dw1y = 11.8
        self.op(3.2, dw1y, 6.2, 0.8, ["wgrad₁: dz_col × x_col", "→ += mg_w1 [E,2I,H] fp32"], fs=5.8)
        _arr(ax, 4.5, qy-0.38, 3.2, dw1y+0.4, C['grad'])

        dw2y = 10.2
        self.op(3.2, dw2y, 6.2, 0.8, ["wgrad₂: dout_col × y1s_col", "→ += mg_w2 [E,H,I] fp32"], fs=5.8)
        _arr(ax, 3.2, dw1y-0.4, 3.2, dw2y+0.4, C['grad'])
        _curv(ax, 7.5, out_y-0.5, 5.2, dw2y+0.4, C['grad'], lw=0.5, rad=0.08)

        self.tensor(2.2, 8.4, "mg_w1", "[E,H,2I] fp32", C['mg'], C['mg_f'], w=3.8, h=0.8)
        self.tensor(5.8, 8.4, "mg_w2", "[E,I,H] fp32", C['mg'], C['mg_f'], w=3.6, h=0.8)
        _arr(ax, 2.2, dw1y-0.4, 2.2, 8.4+0.4, C['mg'], lw=0.9)
        _arr(ax, 4.2, dw2y-0.4, 5.8, 8.4+0.4, C['mg'], lw=0.9)
        self.note(4.0, 7.5, "step()→flush→optim", color=C['mg'], fs=5.3, ha='center')

        dxy = 11.8
        dxt = "dz × w1T [view]" if iso else "dz × w1T [copy]"
        self.op(11.5, dxy, 6.0, 0.85, ["actgrad GEMM", dxt, "→ dx_exp(TK,H)"], fs=5.8)
        _arr(ax, 7.5, qy-0.38, 10.0, dxy+0.43, C['grad'])
        _curv(ax, sc_x-sc_w/2, sep_y-5.5, 11.5+3.0, dxy+0.2, C['ctx'], lw=0.6, dashed=True, rad=0.05)

        self.op(11.5, 10.0, 5.0, 0.6, ["reduce_scatter → dx"], fs=6)
        _arr(ax, 11.5, dxy-0.43, 11.5, 10.0+0.3, C['grad'])

        self.tensor(11.5, 8.8, "dx", "T×H  bf16", C['grad'], C['grad_f'], w=3.5)
        _arr(ax, 11.5, 10.0-0.3, 11.5, 8.8+0.5, C['grad'])
        self.note(11.5, 7.9, "→ autograd (A2A)", color=C['grad'], fs=5.3, ha='center')

        if iso:
            badge = "Weight FP8: 108 MiB (−48.5%)  |  E=8, H=3072, I=1536"
        else:
            badge = "Weight FP8: 216 MiB  |  E=8, H=3072, I=1536"
        ax.text(self.W/2, -1.0, badge, fontsize=6.5, color=wc, ha='center', va='center',
               fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=wc, lw=0.7))


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: DeepEP pipeline (dual panel, with side column)
# ═══════════════════════════════════════════════════════════════════════════════

class PipelinePanel:
    def __init__(self, ax, is_iso32):
        self.ax = ax
        self.iso = is_iso32
        self.W, self.H = 26, 40
        ax.set_xlim(-0.3, self.W + 0.3)
        ax.set_ylim(-1.5, self.H + 0.5)
        ax.axis('off')
        ax.set_aspect('equal')
        self.wc = C['wi'] if is_iso32 else C['wt']

    def tensor(self, cx, cy, name, formula, color, fill, w=6.0, h=1.1):
        _rbox(self.ax, cx, cy, w, h, fill, color, lw=0.9)
        self.ax.text(cx, cy + 0.18, name, ha='center', va='center',
                    fontsize=7.5, color=color, fontweight='bold', zorder=4)
        self.ax.text(cx, cy - 0.25, formula, ha='center', va='center',
                    fontsize=6, color=color, fontstyle='italic', alpha=0.85, zorder=4)

    def op(self, cx, cy, w, h, lines, fs=6.5):
        _rbox(self.ax, cx, cy, w, h, C['op_f'], C['op_b'])
        n = len(lines)
        for i, ln in enumerate(lines):
            yy = cy + (n-1)*0.33/2 - i*0.33 if n > 1 else cy
            fw = 'bold' if i == 0 else 'normal'
            self.ax.text(cx, yy, ln, ha='center', va='center',
                        fontsize=fs, color=C['op'], fontweight=fw, zorder=4)

    def comm(self, cx, cy, w, h, lines, fs=6.5):
        _rbox(self.ax, cx, cy, w, h, C['comm_f'], C['comm'], lw=1.0)
        n = len(lines)
        for i, ln in enumerate(lines):
            yy = cy + (n-1)*0.33/2 - i*0.33 if n > 1 else cy
            fw = 'bold' if i == 0 else 'normal'
            self.ax.text(cx, yy, ln, ha='center', va='center',
                        fontsize=fs, color=C['comm'], fontweight=fw, zorder=4)

    def note(self, x, y, text, color=C['note'], fs=5.8, ha='center'):
        self.ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va='center',
                   fontstyle='italic', zorder=5)

    def draw(self):
        ax = self.ax
        iso = self.iso
        wc = self.wc
        mx = 8.5  # main flow center x

        title = "ISO32 Path" if iso else "Baseline Path"
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10, color=wc)

        # Side column (right side, vertically centered on MLP area)
        sc_x, sc_w = 21.5, 7.5
        sc_top, sc_bot = 30.5, 14.5
        sep_y = 22.5
        draw_side_column(ax, sc_x, sc_w, sc_top, sc_bot, sep_y, iso)

        # ═══ FORWARD ═══
        ax.text(0.3, 39.5, "FORWARD", fontsize=9, fontweight='bold', color=C['act'])

        self.tensor(mx, 38.3, "hidden_states", "B×S×H  bf16", C['act'], C['act_f'], w=6.5)
        self.op(mx, 36.9, 9.0, 0.85, ["Router: F.linear(x, W_r) → softmax → topK"], fs=6.5)
        _arr(ax, mx, 38.3-0.55, mx, 36.9+0.43, C['act'])

        self.comm(mx, 35.3, 10.0, 1.0, ["DeepEP dispatch (All-to-All)", "hidden_states → dispatched (N_recv, H)"], fs=6.2)
        _arr(ax, mx, 36.9-0.43, mx, 35.3+0.5, C['act'])

        self.tensor(mx, 33.5, "dispatched_hidden", "N_recv×H  bf16", C['act'], C['act_f'], w=6.5)
        _arr(ax, mx, 35.3-0.5, mx, 33.5+0.55, C['act'])

        self.op(mx, 32.0, 10.5, 0.8, ["deepep_topk_to_sonic_metadata (zero CPU-GPU sync)", "→ expert_freq_offset, gather_idx, scatter_idx, scores"], fs=5.5)
        _arr(ax, mx, 33.5-0.55, mx, 32.0+0.4, C['act'])

        # ─── FP8 FFN ───
        g1y = 30.2
        g1t = "x_fp8 × w1.mT [view] → z + y1" if iso else "x_fp8 × w1_fused → z + y1"
        self.op(mx, g1y, 12.0, 0.85, ["GemmGatedSm100 (fused SwiGLU CUTLASS)", g1t], fs=6)
        _arr(ax, mx, 32.0-0.4, mx, g1y+0.43, C['act'])
        _arr(ax, sc_x-sc_w/2, sc_top-2.5, mx+6.0, g1y+0.1, wc, lw=0.9)

        g2y = 28.5
        g2t = "y1_fp8 × w2 [direct] → y2" if iso else "y1_fp8 × w2_fp8 → y2"
        self.op(mx, g2y, 12.0, 0.85, ["blockscaled_fp8_gemm_varlen (CUTLASS)", g2t], fs=6)
        _arr(ax, mx, g1y-0.43, mx, g2y+0.43, C['act'])
        _arr(ax, sc_x-sc_w/2, sc_top-4.0, mx+6.0, g2y+0.1, wc, lw=0.9)

        self.op(mx, 26.8, 10.5, 0.8, ["router_forward: y2 × scores → scatter → combine"], fs=6)
        _arr(ax, mx, g2y-0.43, mx, 26.8+0.4, C['act'])

        # save_for_backward
        self.note(mx, 25.7, "save_for_backward: z_fp8, scores, indices, w1T/w2T", color=C['ctx'], fs=5.3)
        _curv(ax, mx+4.0, 25.9, sc_x-sc_w/2, sep_y-2.0, C['ctx'], lw=0.5, dashed=True, rad=-0.08)

        self.tensor(mx, 24.8, "mlp_output", "N_recv×H  bf16", C['act'], C['act_f'])
        _arr(ax, mx, 26.8-0.4, mx, 24.8+0.55, C['act'])

        self.comm(mx, 23.0, 10.0, 1.0, ["DeepEP combine (reverse All-to-All)", "mlp_output → combined_output (B×S, H)"], fs=6.2)
        _arr(ax, mx, 24.8-0.55, mx, 23.0+0.5, C['act'])

        self.tensor(mx, 21.2, "output", "B×S×H  bf16", C['act'], C['act_f'])
        _arr(ax, mx, 23.0-0.5, mx, 21.2+0.55, C['act'])

        # ═══ DIVIDER ═══
        ax.axhline(19.8, color=C['div'], lw=0.8, ls='--', zorder=1, xmin=0.01, xmax=0.62)

        # ═══ BACKWARD ═══
        ax.text(0.3, 19.0, "BACKWARD", fontsize=9, fontweight='bold', color=C['grad'])

        self.tensor(mx, 17.8, "grad_output", "B×S×H  bf16", C['grad'], C['grad_f'])

        self.comm(mx, 16.2, 10.0, 1.0, ["DeepEP reverse All-to-All", "grad → dispatched_grad (N_recv, H)"], fs=6.2)
        _arr(ax, mx, 17.8-0.55, mx, 16.2+0.5, C['grad'])

        # GemmDGated
        g3y = 14.3
        g3t = "dout × w2T ⊗ z_fp8 → dz + y1s + ds" if iso else "dout × w2_dgated ⊗ z_fp8 → dz + y1s + ds"
        self.op(mx, g3y, 12.0, 0.9, ["GemmDGated (fused dgrad + act_bwd + score_bwd)", g3t], fs=5.8)
        _arr(ax, mx, 16.2-0.5, mx, g3y+0.45, C['grad'])
        _arr(ax, sc_x-sc_w/2, sep_y-4.0, mx+6.0, g3y+0.1, C['ctx'], lw=0.6, dashed=True)

        # Outputs
        out_y = 12.2
        self.tensor(3.5, out_y, "dz", "TK×2I  bf16", C['grad'], C['grad_f'], w=3.8, h=0.9)
        self.tensor(8.5, out_y, "y1s", "TK×I  bf16", C['grad'], C['grad_f'], w=3.5, h=0.9)
        self.tensor(14.0, out_y, "ds", "TK  fp32", C['ds'], C['ds_f'], w=3.5, h=0.9)
        _arr(ax, mx-3.5, g3y-0.45, 3.5, out_y+0.45, C['grad'])
        _arr(ax, mx, g3y-0.45, 8.5, out_y+0.45, C['grad'])
        _arr(ax, mx+4.0, g3y-0.45, 14.0, out_y+0.45, C['ds'])
        self.note(14.0, out_y-0.65, "→ router gate autograd", color=C['ds'], fs=5.3)

        # dw
        self.op(4.0, 10.0, 7.5, 0.9, ["wgrad: dz_col×x_col → += mg_w1", "dout_col×y1s_col → += mg_w2"], fs=5.8)
        _arr(ax, 3.5, out_y-0.45, 4.0, 10.0+0.45, C['grad'])
        _curv(ax, 8.5, out_y-0.45, 6.0, 10.0+0.45, C['grad'], lw=0.5, rad=0.08)

        self.tensor(4.0, 8.0, "main_grad w1, w2", "fp32 accum [TMA add]", C['mg'], C['mg_f'], w=7.0, h=0.95)
        _arr(ax, 4.0, 10.0-0.45, 4.0, 8.0+0.48, C['mg'], lw=0.9)
        self.note(4.0, 7.0, "step() → layout flush → optimizer", color=C['mg'], fs=5.3)

        # dx
        dxt = "dz_fp8 × w1T [view] → dx" if iso else "dz_fp8 × w1T_fp8 [copy] → dx"
        self.op(14.0, 10.0, 7.0, 0.9, ["actgrad GEMM + reduce_scatter", dxt], fs=5.8)
        _arr(ax, 5.0, out_y-0.45, 12.5, 10.0+0.45, C['grad'])

        self.tensor(14.0, 8.0, "dx", "N_recv×H  bf16", C['grad'], C['grad_f'], w=5.0)
        _arr(ax, 14.0, 10.0-0.45, 14.0, 8.0+0.48, C['grad'])

        # DeepEP reverse for dx
        self.comm(mx, 5.8, 10.0, 1.0, ["DeepEP reverse A2A (dx)", "→ grad_hidden_states (B×S, H)"], fs=6.2)
        _arr(ax, 14.0, 8.0-0.48, mx+3.0, 5.8+0.5, C['grad'])

        self.tensor(mx, 3.8, "grad_hidden_states", "B×S×H  bf16", C['grad'], C['grad_f'], w=6.5)
        _arr(ax, mx, 5.8-0.5, mx, 3.8+0.55, C['grad'])

        # Gradient path legend at bottom
        self.note(mx, 2.5, "dx → autograd → upstream layers", color=C['grad'], fs=5.8)
        self.note(mx, 1.8, "ds → autograd → router gate W_r", color=C['ds'], fs=5.8)
        self.note(mx, 1.1, "dw → main_grad accum → step() → optimizer", color=C['mg'], fs=5.8)

        # Badge
        if iso:
            badge = "Weight cache: 108 MiB (−48.5%)"
            bc = C['save']
        else:
            badge = "Weight cache: 216 MiB"
            bc = wc
        ax.text(mx, -1.0, badge, fontsize=7, color=bc, ha='center', va='center',
               fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=bc, lw=0.7))


def main():
    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 17))
    fig1.suptitle("SonicMoE FP8:  Forward–Backward Weight Dataflow",
                 fontsize=13, fontweight='bold', y=0.997)
    fig1.text(0.5, 0.985,
            "━━ allocated    ┄┄ eliminated (ISO32 saves)  │  E=8 H=3072 I=1536",
            fontsize=7.5, ha='center', color='#555', fontstyle='italic')
    WeightPanel(ax1, False).draw()
    WeightPanel(ax2, True).draw()
    lh = [
        mpatches.Patch(fc=C['act_f'], ec=C['act'], lw=1.2, label='Activations'),
        mpatches.Patch(fc=C['wt_f'], ec=C['wt'], lw=1.2, label='Weight FP8 (baseline)'),
        mpatches.Patch(fc=C['wi_f'], ec=C['wi'], lw=1.2, label='Weight FP8 (iso32)'),
        mpatches.Patch(fc=C['grad_f'], ec=C['grad'], lw=1.2, label='Gradients'),
        mpatches.Patch(fc=C['mg_f'], ec=C['mg'], lw=1.0, label='main_grad'),
        mpatches.Patch(fc=C['ds_f'], ec=C['ds'], lw=1.0, label='ds (score grad)'),
        mpatches.Patch(fc=C['elim_f'], ec=C['elim'], lw=0.8, label='Eliminated (ISO32)'),
        mpatches.Patch(fc=C['op_f'], ec=C['op_b'], lw=0.8, label='CUTLASS / Triton'),
    ]
    fig1.legend(handles=lh, loc='lower center', ncol=8, fontsize=7,
              frameon=True, edgecolor='#ccc', fancybox=True, bbox_to_anchor=(0.5, 0.002))
    plt.tight_layout(rect=[0, 0.03, 1, 0.975])
    fig1.savefig(out_dir / "fig_iso32_dataflow_comparison.png", dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_dir / 'fig_iso32_dataflow_comparison.png'}")
    plt.close(fig1)

    # Figure 2
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(22, 17))
    fig2.suptitle("SonicMoE FP8 Frontier:  Full DeepEP Pipeline Dataflow",
                 fontsize=13, fontweight='bold', y=0.997)
    fig2.text(0.5, 0.985,
            "Router → DeepEP A2A → FP8 FFN → Combine  │  3 grad paths: dx (autograd) · ds (router) · dw (main_grad)",
            fontsize=7.5, ha='center', color='#555', fontstyle='italic')
    PipelinePanel(ax3, False).draw()
    PipelinePanel(ax4, True).draw()
    lh2 = [
        mpatches.Patch(fc=C['act_f'], ec=C['act'], lw=1.2, label='Activations'),
        mpatches.Patch(fc=C['grad_f'], ec=C['grad'], lw=1.2, label='Gradients'),
        mpatches.Patch(fc=C['ds_f'], ec=C['ds'], lw=1.0, label='ds (score)'),
        mpatches.Patch(fc=C['mg_f'], ec=C['mg'], lw=1.0, label='main_grad'),
        mpatches.Patch(fc=C['comm_f'], ec=C['comm'], lw=1.0, label='All-to-All'),
        mpatches.Patch(fc=C['op_f'], ec=C['op_b'], lw=0.8, label='Compute'),
        mpatches.Patch(fc=C['elim_f'], ec=C['elim'], lw=0.8, label='Eliminated (ISO32)'),
    ]
    fig2.legend(handles=lh2, loc='lower center', ncol=7, fontsize=7,
              frameon=True, edgecolor='#ccc', fancybox=True, bbox_to_anchor=(0.5, 0.002))
    plt.tight_layout(rect=[0, 0.03, 1, 0.975])
    fig2.savefig(out_dir / "fig_deepep_fp8_pipeline.png", dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_dir / 'fig_deepep_fp8_pipeline.png'}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
