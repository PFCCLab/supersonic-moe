"""Iso32 vs 1×32 weight blockscaled-FP8 quant — rigorous numerics audit.

Pure PyTorch implementation of both quant flavors so we don't have to deal with
the ISA-tile-packed scale layout. Compares quant→dequant fidelity vs the BF16
ground truth on realistic reference expert-weight shapes.

Metrics per-shape:
- cosine similarity   (closer to 1 better)
- RRMSE = ||w - w_dq||_2 / ||w||_2
- max abs error
- per-(32-row, 32-col) tile dynamic-range loss in mantissa bits

E4M3 has a single max abs of 448 (S.1111.110). The quant scale is chosen as the
smallest e8m0 power-of-two such that block_amax / 2^e ≤ 448. Iso32 picks one
scale per 32×32 tile (broadcast to all 32 rows in the ISA scale layout); 1×32
picks one scale per (1-row, 32-col) tile. Iso32 is strictly less precise: any
row whose amax is much smaller than the tile amax pays log2(tile_amax/row_amax)
mantissa bits.
"""

from __future__ import annotations

import math
import torch


_E4M3_MAX = 448.0
_GROUP = 32  # K group
_ROW_TILE = 32  # iso row tile


def _quant_dequant_blockscaled(w: torch.Tensor, row_tile: int) -> torch.Tensor:
    """Pure-PyTorch blockscaled FP8 quant→dequant.

    row_tile=1  → 1×32 row-wise (production 1x32 path)
    row_tile=32 → 32×32 isotropic (current default weight-quant path)
    """
    M, K = w.shape
    assert M % row_tile == 0 and K % _GROUP == 0
    w_f = w.float()
    tiles = w_f.view(M // row_tile, row_tile, K // _GROUP, _GROUP)
    amax = tiles.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-30)
    # e8m0: pick exp e such that 2^e * 448 >= amax, i.e. e = ceil(log2(amax/448))
    e = torch.ceil(torch.log2(amax / _E4M3_MAX))
    scale = torch.pow(2.0, e)
    # quantize to e4m3 (clamp to ±448, round-to-nearest-even via fp8 cast)
    q = (tiles / scale).clamp(-_E4M3_MAX, _E4M3_MAX)
    q = q.to(torch.float8_e4m3fn).to(torch.float32)
    dq = (q * scale).view(M, K).to(w.dtype)
    return dq


def _metrics(w: torch.Tensor, w_dq: torch.Tensor) -> dict:
    w_f = w.float()
    d = w_f - w_dq.float()
    cos = torch.nn.functional.cosine_similarity(
        w_f.flatten().unsqueeze(0), w_dq.float().flatten().unsqueeze(0)
    ).item()
    rrmse = (d.pow(2).sum().sqrt() / w_f.pow(2).sum().sqrt().clamp_min(1e-30)).item()
    return {"cosine": cos, "rrmse": rrmse, "max_abs": d.abs().max().item()}


def _per_row_dyn_range_loss(w: torch.Tensor) -> dict:
    M, K = w.shape
    if M % _ROW_TILE or K % _GROUP:
        return {}
    w_abs = w.float().abs()
    tiles = w_abs.view(M // _ROW_TILE, _ROW_TILE, K // _GROUP, _GROUP)
    tile_amax = tiles.amax(dim=(1, 3), keepdim=True)  # iso32 effective amax
    row_amax = tiles.amax(dim=3, keepdim=True)        # 1x32 amax
    bits_lost = (tile_amax / row_amax.clamp_min(1e-30)).log2().clamp_min(0)
    flat = bits_lost.flatten()
    return {
        "mean": flat.mean().item(),
        "p50": flat.quantile(0.50).item(),
        "p95": flat.quantile(0.95).item(),
        "p99": flat.quantile(0.99).item(),
        "max": flat.max().item(),
        "frac>0.5b": (flat > 0.5).float().mean().item(),
        "frac>1b":   (flat > 1.0).float().mean().item(),
        "frac>2b":   (flat > 2.0).float().mean().item(),
    }


SHAPES = [
    ("w1   E8 (2I=3072, H=3072)",  8 * 3072, 3072),
    ("w2   E8 (H=3072,  I=1536)",  8 * 3072, 1536),
    ("w1   E32(2I=3072, H=3072)", 32 * 3072, 3072),
    ("w2   E32(H=3072,  I=1536)", 32 * 3072, 1536),
]


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    print(f"{'Shape':<30s}  {'method':<6s}  {'cosine':>11s}  {'RRMSE':>11s}  {'max_abs':>11s}")
    print("-" * 80)
    for label, M, K in SHAPES:
        w = (torch.randn(M, K, device=device, dtype=torch.bfloat16) * (1.0 / math.sqrt(K))).contiguous()
        for name, rt in (("1x32", 1), ("iso32", 32)):
            dq = _quant_dequant_blockscaled(w, rt)
            m = _metrics(w, dq)
            print(f"{label:<30s}  {name:<6s}  {m['cosine']:>11.8f}  {m['rrmse']:>11.4e}  {m['max_abs']:>11.4e}")
        dr = _per_row_dyn_range_loss(w)
        if dr:
            print(
                f"  iso32 vs 1x32 dyn-range loss: mean={dr['mean']:.3f}b "
                f"p95={dr['p95']:.3f}b p99={dr['p99']:.3f}b max={dr['max']:.3f}b "
                f"frac>0.5b={dr['frac>0.5b']*100:.2f}% frac>1b={dr['frac>1b']*100:.2f}% "
                f"frac>2b={dr['frac>2b']*100:.2f}%"
            )
        print("-" * 80)

    print("\n[Stress] heavy-tail rows (3% of rows scaled 100×, mimics outlier weights):")
    M, K = 8 * 3072, 3072
    w = (torch.randn(M, K, device=device, dtype=torch.bfloat16) * (1.0 / math.sqrt(K))).contiguous()
    idx = torch.randperm(M, device=device)[: M // 32]
    w[idx] *= 100.0
    for name, rt in (("1x32", 1), ("iso32", 32)):
        dq = _quant_dequant_blockscaled(w, rt)
        m = _metrics(w, dq)
        print(f"  {name:<6s}  cos={m['cosine']:.8f}  rrmse={m['rrmse']:.4e}  max_abs={m['max_abs']:.4e}")
    dr = _per_row_dyn_range_loss(w)
    print(
        f"  iso32 dyn-range loss: mean={dr['mean']:.3f}b p95={dr['p95']:.3f}b "
        f"p99={dr['p99']:.3f}b max={dr['max']:.3f}b frac>0.5b={dr['frac>0.5b']*100:.2f}% "
        f"frac>1b={dr['frac>1b']*100:.2f}% frac>2b={dr['frac>2b']*100:.2f}%"
    )

    print("\n[Stress] per-row variance (rows have wildly different scales):")
    M, K = 8 * 3072, 3072
    base = torch.randn(M, K, device=device, dtype=torch.float32)
    row_scale = torch.pow(2.0, torch.randint(-6, 7, (M, 1), device=device, dtype=torch.float32))
    w = (base * row_scale * (1.0 / math.sqrt(K))).to(torch.bfloat16).contiguous()
    for name, rt in (("1x32", 1), ("iso32", 32)):
        dq = _quant_dequant_blockscaled(w, rt)
        m = _metrics(w, dq)
        print(f"  {name:<6s}  cos={m['cosine']:.8f}  rrmse={m['rrmse']:.4e}  max_abs={m['max_abs']:.4e}")
    dr = _per_row_dyn_range_loss(w)
    print(
        f"  iso32 dyn-range loss: mean={dr['mean']:.3f}b p95={dr['p95']:.3f}b "
        f"p99={dr['p99']:.3f}b max={dr['max']:.3f}b frac>0.5b={dr['frac>0.5b']*100:.2f}% "
        f"frac>1b={dr['frac>1b']*100:.2f}% frac>2b={dr['frac>2b']*100:.2f}%"
    )


if __name__ == "__main__":
    main()
