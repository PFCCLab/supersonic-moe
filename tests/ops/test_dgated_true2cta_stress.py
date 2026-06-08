"""Stress + safety tests for the dgated true-2CTA (tile_m=256) default lever.

Validates the Session-38 change in gemm_interface.default_config (tile_m=256 for
the backward dgated GEMM at E in [2,64]) across the regimes that matter for
large-scale pretraining stability:

  * INT32 / UINT32 element-offset boundary: dz/z have TK*2I elements; at
    I=4096 TK=262144 that is exactly 2**31 (> INT32_MAX), and TK=524288 is
    2**32 (> UINT32_MAX).  The dgated GEMM (CuTe DSL) must use 64-bit
    addressing -> tile_m=256 must stay BIT-IDENTICAL to tile_m=128 and finite.
  * Expert-count boundaries: E=2 (min gated -> tile256), E=64 (max -> tile256),
    E=65 (just over -> tile128).  tile256 must equal the tile128 reference.
  * Adversarial routing: 80%-skew, single-dominant, empty experts.

Run directly (compare mode) OR under compute-sanitizer (single-config mode):
  python tests/ops/test_dgated_true2cta_stress.py                 # compare all
  python tests/ops/test_dgated_true2cta_stress.py --case int32 --tm 256  # 1 cfg
  compute-sanitizer --tool memcheck --error-exitcode 9 \
      python tests/ops/test_dgated_true2cta_stress.py --case int32 --tm 256
"""
import argparse, os, sys
import torch

# Allow running from a source checkout without install.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(_HERE, "..", "..")),
           os.path.abspath(os.path.join(_HERE, "..", "..", "third_party", "quack"))):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_activation_blockscaled_fast,
    precompute_weight_fp8_for_direct_fused_dgated,
)
from sonicmoe.functional import _gather_1x32_scales_to_isa
from sonicmoe.quack_utils.gemm_dgated import gemm_dgated as gemm_dgated_kernel
from sonicmoe.quack_utils import gemm_interface

DEV = "cuda"
_E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)


def _div_up(a, b):
    return (a + b - 1) // b


def _cu_uniform(TK, E):
    per = TK // E
    assert per * E == TK and per % 128 == 0, (TK, E, per)
    return torch.arange(0, TK + 1, per, device=DEV, dtype=torch.int32)


def _cu_skew(TK, E, frac=0.8):
    # one dominant expert gets ~frac of tokens (128-aligned), rest split evenly
    big = int(TK * frac) // 128 * 128
    rest = TK - big
    others = E - 1
    base = (rest // others) // 128 * 128
    segs = [big] + [base] * others
    segs[-1] += TK - sum(segs)  # absorb remainder into last
    assert sum(segs) == TK and all(s % 128 == 0 for s in segs), segs
    cu = [0]
    for s in segs:
        cu.append(cu[-1] + s)
    return torch.tensor(cu, device=DEV, dtype=torch.int32)


def _cu_single(TK, E):
    segs = [TK - 128, 128] + [0] * (E - 2)
    cu = [0]
    for s in segs:
        cu.append(cu[-1] + s)
    return torch.tensor(cu, device=DEV, dtype=torch.int32)


def build(TK, H, I, E, seed=0):
    torch.manual_seed(seed)
    dout = (torch.randn(TK, H, device=DEV, dtype=torch.bfloat16) * 0.02)
    dout_fp8, dout_raw = quantize_activation_blockscaled_fast(dout)
    gi = torch.arange(TK, device=DEV, dtype=torch.int32)
    dsc = _gather_1x32_scales_to_isa(dout_raw, gi, TK, H)
    w2 = (torch.randn(H, I, E, device=DEV, dtype=torch.bfloat16) * 0.02)
    w2f, w2s = precompute_weight_fp8_for_direct_fused_dgated(w2)
    n = w2f.shape[-2]
    z = (torch.randn(TK, 2 * I, device=DEV, dtype=torch.bfloat16) * 0.02)
    sf = (torch.rand(TK, device=DEV, dtype=torch.float32) + 0.5)
    return dict(dout_fp8=dout_fp8, w2f=w2f, w2s=w2s, z=z, n=n, sf=sf,
                gi=gi, dsc=dsc, TK=TK)


def run(d, cu, TM, clamp=0.0):
    TK, n = d["TK"], d["n"]
    dz = torch.empty((TK, n * 2), dtype=torch.bfloat16, device=DEV)
    cr = torch.empty((TK, _div_up(n, 128)), dtype=torch.float32, device=DEV)
    y = torch.empty((TK, n), dtype=torch.bfloat16, device=DEV)
    gemm_dgated_kernel(
        d["dout_fp8"], d["w2f"], dz, d["z"], y, None, "swiglu",
        TM, 128, 2, 1, False, persistent=True, max_swizzle_size=8,
        colvec_scale=d["sf"], colvec_reduce=cr,
        cu_seqlens_m=cu, A_idx=d["gi"], a_scales=d["dsc"], b_scales=d["w2s"],
        preact_fp8=None, preact_scales=None, y1s_col_fp8=None, y1s_col_scales=None,
        swiglu_clamp_value=clamp,
    )
    return dz, cr, y


CASES = {
    # name: (TK, H, I, E)  -- dz/z elements = TK*2I; max element index = elems-1.
    # TK must be E*8192*k so per-expert tokens stay a multiple of 128.
    "int32":  (270336, 2048, 4096, 64),   # 2,214,592,512 elems -> max idx > INT32_MAX
    "uint32": (786432, 2048, 4096, 64),   # 6,442,450,944 elems -> max idx > UINT32_MAX
    "prod":   (262144, 2048, 1024, 64),   # production Qwen3-30B-A3B
    "e2":     (262144, 2048, 1024, 2),    # min gated E
    "e64":    (262144, 2048, 1024, 64),
}


def _check_finite(tag, dz, cr, y):
    assert torch.isfinite(dz).all(), f"{tag}: dz non-finite"
    assert torch.isfinite(cr).all(), f"{tag}: colvec non-finite"
    assert torch.isfinite(y.float()).all(), f"{tag}: y1s non-finite"


def compare_case(name, routing="uniform"):
    TK, H, I, E = CASES[name]
    d = build(TK, H, I, E)
    if routing == "uniform":
        cu = _cu_uniform(TK, E)
    elif routing == "skew":
        cu = _cu_skew(TK, E)
    elif routing == "single":
        cu = _cu_single(TK, E)
    else:
        raise ValueError(routing)
    dz1, cr1, y1 = run(d, cu, 128)
    torch.cuda.synchronize()
    _check_finite(f"{name}/{routing}/tm128", dz1, cr1, y1)
    dz2, cr2, y2 = run(d, cu, 256)
    torch.cuda.synchronize()
    _check_finite(f"{name}/{routing}/tm256", dz2, cr2, y2)
    ddz = (dz1.float() - dz2.float()).abs().max().item()
    dcr = (cr1 - cr2).abs().max().item()
    dy = (y1.float() - y2.float()).abs().max().item()
    elems = TK * 2 * I
    ok = (ddz == 0.0 and dcr == 0.0 and dy == 0.0)
    if elems - 1 > 2**32 - 1:
        boundary = ">UINT32"
    elif elems - 1 > 2**31 - 1:
        boundary = ">INT32"
    else:
        boundary = "ok"
    print(f"[{name:7s}/{routing:7s}] TK={TK} 2I={2*I} elems={elems:,} maxidx={elems-1:,} "
          f"({boundary}) dz_d={ddz} cr_d={dcr} y_d={dy} -> {'PASS' if ok else 'FAIL'}")
    del d, dz1, dz2, cr1, cr2, y1, y2
    torch.cuda.empty_cache()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="single case for sanitizer")
    ap.add_argument("--tm", type=int, default=256)
    ap.add_argument("--routing", default="uniform")
    ap.add_argument("--clamp", type=float, default=0.0)
    args = ap.parse_args()

    if args.case:  # single-config mode (for compute-sanitizer)
        TK, H, I, E = CASES[args.case]
        d = build(TK, H, I, E)
        cu = {"uniform": _cu_uniform, "skew": _cu_skew, "single": _cu_single}[args.routing](TK, E)
        dz, cr, y = run(d, cu, args.tm, clamp=args.clamp)
        torch.cuda.synchronize()
        _check_finite(f"{args.case}/{args.routing}/tm{args.tm}/clamp{args.clamp}", dz, cr, y)
        print(
            f"[{args.case}/{args.routing}/tm{args.tm}/clamp{args.clamp}] "
            f"finite OK dz_norm={dz.float().norm().item():.4f}"
        )
        return

    allok = True
    # int32/uint32 boundary (uniform) -- the headline safety test
    for c in ("int32", "uint32"):
        allok &= compare_case(c, "uniform")
    # expert-count boundary + production, with adversarial routing
    for c in ("prod", "e2", "e64"):
        for r in ("uniform", "skew", "single"):
            allok &= compare_case(c, r)
    # E=65 gate check: default must pick tile128 (no tile256 at E>64)
    cfg65 = gemm_interface.default_config(torch.device(DEV), num_experts=65)
    gate_ok = (cfg65.tile_m == 128 and cfg65.cluster_m == 1)
    print(f"[gate E=65] tile_m={cfg65.tile_m} cluster_m={cfg65.cluster_m} -> {'PASS' if gate_ok else 'FAIL'}")
    allok &= gate_ok
    print("=" * 60)
    print("ALL PASS" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
