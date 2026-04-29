#!/usr/bin/env python3
"""Strict JIT-mechanism baseline gate for sonic-moe CI.

Measures four orthogonal indicators and enforces budgets from
``tools/ci/baselines.json``:

  cold-warmup           rm -rf cache  → warmup_jit(force=True).
  warm-sentinel-skip    cache + sentinel present → warmup_jit returns False fast.
  cross-process-reload  sentinel cleared, disk caches kept → warmup_jit runs but
                        Triton/Quack hit disk; CuTe re-compiles in-process.
  in-process-reuse      Already-warmed process; second SonicMoEMlpNode forward
                        latency excluding the (warm) first call.

Each subprocess phase is launched cleanly so timings are not contaminated by
prior in-memory state. Exits non-zero on first budget breach (or, with
``--soft``, only when an indicator exceeds budget AND the bench itself
succeeded — used by ``--fast`` CI).

Usage:
    python tools/ci/jit_bench.py [--phase cold|warm|reload|reuse|all]
                                 [--baselines tools/ci/baselines.json]
                                 [--cache-dir /tmp/sonicmoe_jit_ci]
                                 [--soft]   # warn only, exit 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# ── helpers ───────────────────────────────────────────────────────────────────
def _color(s: str, c: str) -> str:
    return f"\033[{c}m{s}\033[0m" if sys.stdout.isatty() else s


def _green(s): return _color(s, "32")
def _red(s):   return _color(s, "31")
def _yel(s):   return _color(s, "33")


def _run_subprocess(body: str, cache_dir: Path, *, env_extra=None) -> tuple[int, str, float]:
    env = os.environ.copy()
    env["SONIC_MOE_CACHE_DIR"] = str(cache_dir)
    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))
    # quack lives outside the importing interpreter's site-packages on this
    # host; tests/ops/* benches inject it via sys.path.insert. Mirror that
    # for every JIT-bench subprocess so /usr/local/bin/python (no quack
    # installed) can still import sonicmoe.
    _QUACK = ("/root/paddlejob/share-storage/gpfs/system-public/"
              "zhangyichen/sonicmoe_for_ernie/quack")
    if os.path.isdir(_QUACK):
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            _QUACK + (os.pathsep + prev if prev else "")
        )
    if env_extra:
        env.update(env_extra)
    t0 = time.perf_counter()
    cp = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )
    dt = time.perf_counter() - t0
    return cp.returncode, (cp.stdout or "") + (cp.stderr or ""), dt


# ── phase bodies ──────────────────────────────────────────────────────────────
_BODY_WARMUP = """
import os, sys, json, time
# Match production import order: enable torch-proxy BEFORE importing sonicmoe.
# sonicmoe.jit binds ``torch.utils.cpp_extension.load`` at module-import time;
# without this the JIT compiles a torch-pybind .so that rejects paddle.Tensor
# with TypeError(incompatible function arguments) on first kernel call.
import paddle
paddle.compat.enable_torch_proxy(scope={'sonicmoe', 'quack', 'triton'}, silent=True)

from sonicmoe.cache_manager import setup_cache, clear_all_caches, clear_warmup_sentinel
from sonicmoe.jit_warmup import warmup_jit

cache = os.environ['SONIC_MOE_CACHE_DIR']
mode  = os.environ['JIT_BENCH_MODE']  # cold | warm | reload
E, H, I = int(os.environ['E']), int(os.environ['H']), int(os.environ['I'])
fp8 = os.environ.get('FP8', '1') == '1'

setup_cache(cache)
if mode == 'cold':
    clear_all_caches()
elif mode == 'reload':
    # Keep the disk caches; only clear the sentinel so warmup_jit re-runs
    # but reloads compiled artifacts from Triton/Quack disk caches.
    clear_warmup_sentinel()
elif mode == 'warm':
    pass  # cache + sentinel both preserved

t0 = time.perf_counter()
ran = warmup_jit(E=E, H=H, I=I, fp8=fp8)
dt  = time.perf_counter() - t0
print(json.dumps({'phase': mode, 'ran': bool(ran), 'elapsed_s': dt}))
"""

_BODY_REUSE = """
import os, sys, json, time
import paddle
paddle.compat.enable_torch_proxy(scope={'sonicmoe', 'quack', 'triton'}, silent=True)
import torch
from sonicmoe.cache_manager import setup_cache
setup_cache(os.environ['SONIC_MOE_CACHE_DIR'])

# Skip-if-warm to ensure compiled state without re-running compile work.
from sonicmoe.jit_warmup import warmup_jit, _warmup_single
E, H, I = int(os.environ['E']), int(os.environ['H']), int(os.environ['I'])
warmup_jit(E=E, H=H, I=I, fp8=True)

# In-process reuse signal: time N successive _warmup_single calls. The
# first one through any given shape may still trigger CuTe in-process
# compile (cubin reload from disk via Triton/Quack caches), but the
# remaining N-1 are pure dispatch — exactly what production sees inside a
# training loop after warmup.
device = torch.device('cuda', torch.cuda.current_device())
total_K = E * 128

# 1st call — warmup_single populates in-process CuTe cache.
torch.cuda.synchronize()
t1 = time.perf_counter()
_warmup_single(E=E, H=H, I=I, total_K=total_K, device=device, fp8=True)
torch.cuda.synchronize()
first_us = (time.perf_counter() - t1) * 1e6

N = 5
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    _warmup_single(E=E, H=H, I=I, total_K=total_K, device=device, fp8=True)
torch.cuda.synchronize()
us = (time.perf_counter() - t0) * 1e6 / N
print(json.dumps({'phase': 'reuse', 'iters': N, 'us_per_iter': us,
                  'first_call_us': first_us}))
"""


# ── runners ───────────────────────────────────────────────────────────────────
def _parse_json_line(out: str) -> dict | None:
    for ln in reversed(out.splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                continue
    return None


def _gate(label, value, budget, warn, unit, results):
    ok = value <= budget
    warn_only = ok and value > warn
    tag = _green("PASS") if (ok and not warn_only) else (
        _yel("WARN") if warn_only else _red("FAIL"))
    print(f"  {tag} {label:32s}  {value:8.1f} {unit}  "
          f"(warn≤{warn}, budget≤{budget})")
    results.append({
        "label": label, "value": value, "budget": budget,
        "warn": warn, "unit": unit,
        "status": "fail" if not ok else ("warn" if warn_only else "pass"),
    })
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="all",
                   choices=["all", "cold", "warm", "reload", "reuse",
                            "parallel-cold"])
    p.add_argument("--baselines", default=str(REPO / "tools/ci/baselines.json"))
    p.add_argument("--cache-dir", default=None,
                   help="Per-run cache root; default = tempdir.")
    p.add_argument("--soft", action="store_true",
                   help="Print results but exit 0 on budget breach.")
    p.add_argument("--json", default=None,
                   help="Write a machine-readable summary to this path.")
    args = p.parse_args()

    base = json.loads(Path(args.baselines).read_text())
    sh = base["jit"]["shape"]
    env_shape = {"E": str(sh["E"]), "H": str(sh["H"]), "I": str(sh["I"]),
                 "FP8": "1" if sh["fp8"] else "0"}

    cache = Path(args.cache_dir) if args.cache_dir else Path(
        os.environ.get("SONIC_MOE_CACHE_DIR")
        or tempfile.mkdtemp(prefix="sonicmoe_jit_ci_"))
    cache.mkdir(parents=True, exist_ok=True)
    print(f"[jit_bench] cache_dir = {cache}")
    print(f"[jit_bench] shape     = E={sh['E']} H={sh['H']} I={sh['I']} fp8={sh['fp8']}")

    results: list[dict] = []
    fail = False

    def run_phase(mode: str) -> dict | None:
        rc, out, _ = _run_subprocess(
            _BODY_WARMUP, cache,
            env_extra={**env_shape, "JIT_BENCH_MODE": mode},
        )
        rec = _parse_json_line(out)
        if rc != 0 or rec is None:
            print(_red(f"[{mode}] subprocess failed rc={rc}"))
            sys.stdout.write(out[-2000:] + "\n")
            return None
        return rec

    phases = ["cold", "warm", "reload", "reuse", "parallel-cold"] \
        if args.phase == "all" else [args.phase]

    if "cold" in phases:
        rec = run_phase("cold")
        if rec is None: fail = True
        else:
            b = base["jit"]["cold_warmup_s"]
            if not _gate("cold_warmup_s", rec["elapsed_s"],
                         b["budget"], b["warn"], "s", results): fail = True
            assert rec["ran"] is True, "cold warmup should have actually run"

    if "warm" in phases:
        rec = run_phase("warm")
        if rec is None: fail = True
        else:
            b = base["jit"]["warm_sentinel_skip_s"]
            if rec["ran"]:
                print(_red("  [warm] sentinel did NOT skip — warmup ran again"))
                fail = True
            if not _gate("warm_sentinel_skip_s", rec["elapsed_s"],
                         b["budget"], b["warn"], "s", results): fail = True

    if "reload" in phases:
        rec = run_phase("reload")
        if rec is None: fail = True
        else:
            b = base["jit"]["cross_process_reload_s"]
            assert rec["ran"], "reload must actually re-run warmup_jit"
            if not _gate("cross_process_reload_s", rec["elapsed_s"],
                         b["budget"], b["warn"], "s", results): fail = True

    if "reuse" in phases:
        rc, out, _ = _run_subprocess(
            _BODY_REUSE, cache, env_extra=env_shape,
        )
        rec = _parse_json_line(out)
        if rc != 0 or rec is None:
            print(_red(f"[reuse] subprocess failed rc={rc}"))
            sys.stdout.write(out[-2000:] + "\n")
            fail = True
        else:
            b = base["jit"]["in_process_reuse_us"]
            if not _gate("in_process_reuse_us", rec["us_per_iter"],
                         b["budget_us"], b["warn_us"], "µs", results):
                fail = True
            print(f"  [reuse] first-call cold-in-proc = {rec.get('first_call_us', 0):.0f} µs")

    if "parallel-cold" in phases:
        # Wipe cache so this is a true cold parallel.
        import shutil as _sh
        for sub in ("triton", "quack", "sonicmoe"):
            _sh.rmtree(cache / sub, ignore_errors=True)
        try:
            (cache / "warm.sentinel").unlink()
        except FileNotFoundError:
            pass

        body = """
import os, json, time
import paddle
paddle.compat.enable_torch_proxy(scope={'sonicmoe','quack','triton'}, silent=True)
from sonicmoe.cache_manager import setup_cache
setup_cache(os.environ['SONIC_MOE_CACHE_DIR'])
from sonicmoe.jit_warmup import warmup_jit_parallel
E, H, I = int(os.environ['E']), int(os.environ['H']), int(os.environ['I'])
ks = [E*128, E*256, E*512, E*1024]
W = int(os.environ.get('PARALLEL_WORKERS','2'))
t0 = time.perf_counter()
dt = warmup_jit_parallel(E=E, H=H, I=I, fp8=True,
                         total_K_list=ks, workers=W)
print(json.dumps({'phase':'parallel-cold','elapsed_s':dt,'workers':W,
                  'shapes':len(ks)}))
"""
        rc, out, _ = _run_subprocess(
            body, cache,
            env_extra={**env_shape, "PARALLEL_WORKERS": "2"},
        )
        rec = _parse_json_line(out)
        if rc != 0 or rec is None:
            print(_red(f"[parallel-cold] subprocess failed rc={rc}"))
            sys.stdout.write(out[-2000:] + "\n")
            fail = True
        else:
            b = base["jit"].get("parallel_cold_s", {"budget": 900, "warn": 600})
            if not _gate("parallel_cold_s", rec["elapsed_s"],
                         b["budget"], b["warn"], "s", results):
                fail = True

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"shape": sh, "results": results}, indent=2))
        print(f"[jit_bench] wrote {args.json}")

    print()
    print("=" * 60)
    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_warn = sum(1 for r in results if r["status"] == "warn")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    n_skip = sum(1 for r in results if r["status"] == "skip")
    print(f"  JIT bench: {n_pass} pass, {n_warn} warn, "
          f"{n_skip} skip, {n_fail} fail")
    print("=" * 60)

    if fail and not args.soft:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
