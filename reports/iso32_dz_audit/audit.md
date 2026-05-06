# iso32 dz precision audit (Phase 0.2)
Pure-PyTorch quant→dequant comparison on **real** dz tensors captured from the bwd dGated path during Ernie-shape inference.
**Pass thresholds**: direct cos ≥ 0.9995; direct RRMSE iso32/1×32 ≤ 3×; downstream dx & dw1 RRMSE iso32/1×32 ≤ 2×.  Dyn-range bits-lost is reported as advisory only — for e4m3 the 3-bit mantissa noise dominates, so high frac>1b does **not** translate to downstream error.

## dz_extreme_iter4.npy  shape=(65536, 3072)
- direct 1×32 :  cos=0.999646  RRMSE=2.6592e-02  max_abs=1.2500e-01
- direct 32×32:  cos=0.999646  RRMSE=2.6592e-02  max_abs=1.2500e-01
- dyn-range bits-lost: {'mean': 1.555686354637146, 'p50': 1.532607078552246, 'p95': 2.967085361480713, 'p99': 3.563588857650757, 'max': 6.101538181304932, 'frac>0.5b': 0.8926607966423035, 'frac>1b': 0.7351044416427612, 'frac>2b': 0.29165220260620117}
- dx  1×32  vs ref:  cos=0.999647  RRMSE=2.6590e-02  max_abs=2.1980e-02
- dx  32×32 vs ref:  cos=0.999646  RRMSE=2.6590e-02  max_abs=2.1980e-02
- dw1 1×32  vs ref:  cos=0.999647  RRMSE=2.6583e-02  max_abs=8.0897e-02
- dw1 32×32 vs ref:  cos=0.999647  RRMSE=2.6583e-02  max_abs=8.0899e-02
  - direct RRMSE ratio (iso32/1x32) = 1.0000×
  - dyn-range frac>1b=0.7351 (advisory)
  - dx downstream RRMSE ratio (iso32/1x32) = 1.0000×
  - dw1 downstream RRMSE ratio (iso32/1x32) = 1.0000×
- **GATE**: ✅ PASS

## dz_extreme_iter5.npy  shape=(65536, 3072)
- direct 1×32 :  cos=0.999646  RRMSE=2.6592e-02  max_abs=1.2500e-01
- direct 32×32:  cos=0.999646  RRMSE=2.6592e-02  max_abs=1.2500e-01
- dyn-range bits-lost: {'mean': 1.555686354637146, 'p50': 1.532607078552246, 'p95': 2.967085361480713, 'p99': 3.563588857650757, 'max': 6.101538181304932, 'frac>0.5b': 0.8926607966423035, 'frac>1b': 0.7351044416427612, 'frac>2b': 0.29165220260620117}
- dx  1×32  vs ref:  cos=0.999647  RRMSE=2.6590e-02  max_abs=2.1980e-02
- dx  32×32 vs ref:  cos=0.999646  RRMSE=2.6590e-02  max_abs=2.1980e-02
- dw1 1×32  vs ref:  cos=0.999647  RRMSE=2.6583e-02  max_abs=8.0897e-02
- dw1 32×32 vs ref:  cos=0.999647  RRMSE=2.6583e-02  max_abs=8.0899e-02
  - direct RRMSE ratio (iso32/1x32) = 1.0000×
  - dyn-range frac>1b=0.7351 (advisory)
  - dx downstream RRMSE ratio (iso32/1x32) = 1.0000×
  - dw1 downstream RRMSE ratio (iso32/1x32) = 1.0000×
- **GATE**: ✅ PASS

## dz_none_iter4.npy  shape=(65536, 3072)
- direct 1×32 :  cos=0.999646  RRMSE=2.6599e-02  max_abs=1.2500e-01
- direct 32×32:  cos=0.999646  RRMSE=2.6599e-02  max_abs=1.2500e-01
- dyn-range bits-lost: {'mean': 1.5551875829696655, 'p50': 1.5322210788726807, 'p95': 2.9646668434143066, 'p99': 3.560267925262451, 'max': 5.923184394836426, 'frac>0.5b': 0.8925444483757019, 'frac>1b': 0.7351382970809937, 'frac>2b': 0.2911829948425293}
- dx  1×32  vs ref:  cos=0.999646  RRMSE=2.6595e-02  max_abs=2.1211e-02
- dx  32×32 vs ref:  cos=0.999646  RRMSE=2.6595e-02  max_abs=2.1211e-02
- dw1 1×32  vs ref:  cos=0.999646  RRMSE=2.6593e-02  max_abs=8.2873e-02
- dw1 32×32 vs ref:  cos=0.999646  RRMSE=2.6593e-02  max_abs=8.2874e-02
  - direct RRMSE ratio (iso32/1x32) = 1.0000×
  - dyn-range frac>1b=0.7351 (advisory)
  - dx downstream RRMSE ratio (iso32/1x32) = 1.0000×
  - dw1 downstream RRMSE ratio (iso32/1x32) = 1.0000×
- **GATE**: ✅ PASS

## dz_none_iter5.npy  shape=(65536, 3072)
- direct 1×32 :  cos=0.999646  RRMSE=2.6599e-02  max_abs=1.2500e-01
- direct 32×32:  cos=0.999646  RRMSE=2.6599e-02  max_abs=1.2500e-01
- dyn-range bits-lost: {'mean': 1.5551875829696655, 'p50': 1.5322210788726807, 'p95': 2.9646668434143066, 'p99': 3.560267925262451, 'max': 5.923184394836426, 'frac>0.5b': 0.8925444483757019, 'frac>1b': 0.7351382970809937, 'frac>2b': 0.2911829948425293}
- dx  1×32  vs ref:  cos=0.999646  RRMSE=2.6595e-02  max_abs=2.1211e-02
- dx  32×32 vs ref:  cos=0.999646  RRMSE=2.6595e-02  max_abs=2.1211e-02
- dw1 1×32  vs ref:  cos=0.999646  RRMSE=2.6593e-02  max_abs=8.2873e-02
- dw1 32×32 vs ref:  cos=0.999646  RRMSE=2.6593e-02  max_abs=8.2874e-02
  - direct RRMSE ratio (iso32/1x32) = 1.0000×
  - dyn-range frac>1b=0.7351 (advisory)
  - dx downstream RRMSE ratio (iso32/1x32) = 1.0000×
  - dw1 downstream RRMSE ratio (iso32/1x32) = 1.0000×
- **GATE**: ✅ PASS

## dz_skew_iter4.npy  shape=(65536, 3072)
- direct 1×32 :  cos=0.999646  RRMSE=2.6596e-02  max_abs=1.2500e-01
- direct 32×32:  cos=0.999646  RRMSE=2.6596e-02  max_abs=1.2500e-01
- dyn-range bits-lost: {'mean': 1.5556211471557617, 'p50': 1.5326679944992065, 'p95': 2.966447353363037, 'p99': 3.561427116394043, 'max': 5.908576965332031, 'frac>0.5b': 0.8926331400871277, 'frac>1b': 0.7351779937744141, 'frac>2b': 0.2914822995662689}
- dx  1×32  vs ref:  cos=0.999646  RRMSE=2.6593e-02  max_abs=2.3619e-02
- dx  32×32 vs ref:  cos=0.999646  RRMSE=2.6593e-02  max_abs=2.3619e-02
- dw1 1×32  vs ref:  cos=0.999646  RRMSE=2.6603e-02  max_abs=7.8509e-02
- dw1 32×32 vs ref:  cos=0.999646  RRMSE=2.6603e-02  max_abs=7.8509e-02
  - direct RRMSE ratio (iso32/1x32) = 1.0000×
  - dyn-range frac>1b=0.7352 (advisory)
  - dx downstream RRMSE ratio (iso32/1x32) = 1.0000×
  - dw1 downstream RRMSE ratio (iso32/1x32) = 1.0000×
- **GATE**: ✅ PASS

## dz_skew_iter5.npy  shape=(65536, 3072)
- direct 1×32 :  cos=0.999646  RRMSE=2.6596e-02  max_abs=1.2500e-01
- direct 32×32:  cos=0.999646  RRMSE=2.6596e-02  max_abs=1.2500e-01
- dyn-range bits-lost: {'mean': 1.5556211471557617, 'p50': 1.5326679944992065, 'p95': 2.966447353363037, 'p99': 3.561427116394043, 'max': 5.908576965332031, 'frac>0.5b': 0.8926331400871277, 'frac>1b': 0.7351779937744141, 'frac>2b': 0.2914822995662689}
- dx  1×32  vs ref:  cos=0.999646  RRMSE=2.6593e-02  max_abs=2.3619e-02
- dx  32×32 vs ref:  cos=0.999646  RRMSE=2.6593e-02  max_abs=2.3619e-02
- dw1 1×32  vs ref:  cos=0.999646  RRMSE=2.6603e-02  max_abs=7.8509e-02
- dw1 32×32 vs ref:  cos=0.999646  RRMSE=2.6603e-02  max_abs=7.8509e-02
  - direct RRMSE ratio (iso32/1x32) = 1.0000×
  - dyn-range frac>1b=0.7352 (advisory)
  - dx downstream RRMSE ratio (iso32/1x32) = 1.0000×
  - dw1 downstream RRMSE ratio (iso32/1x32) = 1.0000×
- **GATE**: ✅ PASS

# Overall: ✅ PASS — proceed to Phase 1A
