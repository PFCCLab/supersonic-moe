# Blockscaled FP8: 128-Row Alignment Constraint

## Summary

Blockscaled FP8 on SM100 requires each expert's M-dimension segment to be a multiple of **128**. This is a **hardware ISA constraint** from `tcgen05.mma` scale-factor atom layout, not a software choice.

**Token rounding routing** (SonicMoE paper Section 5, Algorithm 4) guarantees 128-aligned segments in production. The padding fallback in `blockscaled_fp8_gemm_varlen` is a safety net for vanilla top-K and tests.

## Why 128?

```
Scale factor atom: 32 rows x 4 columns (tcgen05 MMA spec)
Tile packing: 4 atoms stacked -> 128 rows x 128 columns per tile
Storage: _SF_TILE_STORAGE = 128 x (128/32) = 512 bytes per tile
```

| Aspect | BF16 | Blockscaled FP8 |
|--------|------|-----------------|
| Scale factors | None | Pre-packed 1x32 UE8M0 atoms |
| M-alignment | None required | Multiple of 128 |
| Root cause | Data is self-describing | Scale factors have fixed geometry |

## Code References

- Scale packing: `blockscaled_fp8_gemm.py:169` `_scale_pack_index()`
- Padding fallback: `blockscaled_fp8_gemm_varlen` line 862
- Token rounding: `benchmarks/moe-token-rounding.py` `forward_token_choice_rounding()`
