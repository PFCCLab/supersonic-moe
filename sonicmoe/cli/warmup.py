"""Offline warmup CLI.

Run once on a single GPU on shared storage to pre-populate Triton + Quack
disk caches; copy the resulting ``--cache-dir`` to all training ranks (or
mount it via NFS/GPFS) to skip per-rank warmup at training start.

Example
-------
    python -m sonicmoe.cli.warmup \
        --E 32 --H 3072 --I 1536 \
        --cache-dir /nfs/jit_cache_e32_h3072_i1536 \
        --total-K 4096

Then in training (any rank), the same ``SONIC_MOE_CACHE_DIR`` plus the
auto-skip in ``warmup_jit`` (sentinel hit) bypasses re-compilation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sonicmoe.cli.warmup",
        description="Pre-compile sonicmoe kernels into a shareable JIT cache.",
    )
    p.add_argument("--E", type=int, required=True, help="num_experts")
    p.add_argument("--H", type=int, required=True, help="hidden_size")
    p.add_argument("--I", type=int, required=True, help="intermediate_size")
    p.add_argument(
        "--total-K", type=int, action="append", default=None,
        help="Tokens per expert workload (default: production sweep "
             "{E*64, E*128, E*256, E*512, E*1024}). Repeatable.",
    )
    p.add_argument("--no-fp8", action="store_true", help="bf16 path only")
    p.add_argument(
        "--also-bf16", action="store_true",
        help="In addition to the FP8 path, also warm the BF16 path "
             "(doubles compile time but covers fall-back kernels).",
    )
    p.add_argument(
        "--cache-dir", type=str, default=None,
        help="Override SONIC_MOE_CACHE_DIR for this run.",
    )
    p.add_argument(
        "--max-workers", type=int, default=0,
        help="Parallel CuTe compile workers (0 = auto). Within-process.",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Number of warmup subprocesses (process-level partitioning of "
             "--total-K shapes across the disk cache).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Recompile even when the warmup sentinel already matches.",
    )
    p.add_argument(
        "--clear", action="store_true",
        help="Wipe the disk cache + sentinel before warming.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    os.environ.setdefault("SONIC_MOE_JIT_VERBOSE", "1")

    from sonicmoe.cache_manager import (
        setup_cache,
        clear_all_caches,
        cache_stats,
        get_cache_root,
    )

    setup_cache(args.cache_dir) if args.cache_dir else setup_cache()

    if args.clear:
        clear_all_caches()
        logging.info("[warmup-cli] Cleared cache root %s", get_cache_root())

    from sonicmoe.jit_warmup import warmup_jit, warmup_jit_parallel

    # Default production sweep covers the seqlen × topk regimes that the
    # dynamic-dim JIT keys still re-tune for (small/medium/large total_K).
    if args.total_K is None:
        ks = [args.E * mult for mult in (64, 128, 256, 512, 1024)]
    else:
        ks = list(args.total_K)
    fp8_modes = []
    if not args.no_fp8:
        fp8_modes.append(True)
    if args.also_bf16 or args.no_fp8:
        fp8_modes.append(False)
    fp8_modes = list(dict.fromkeys(fp8_modes))  # dedupe, preserve order

    ran_any = False
    if args.workers > 1:
        for fp8 in fp8_modes:
            dt = warmup_jit_parallel(
                E=args.E, H=args.H, I=args.I, fp8=fp8,
                total_K_list=ks, workers=args.workers,
                cache_dir=args.cache_dir,
            )
            ran_any = True
            logging.info("[warmup-cli] parallel cold warmup fp8=%s %.1fs",
                         fp8, dt)
        ran = ran_any
    else:
        ran = False
        for fp8 in fp8_modes:
            ran = warmup_jit(
                E=args.E,
                H=args.H,
                I=args.I,
                fp8=fp8,
                total_K_list=ks,
                max_workers=args.max_workers,
                force=args.force,
            ) or ran
    stats = cache_stats()
    logging.info("[warmup-cli] cache_root = %s", get_cache_root())
    logging.info("[warmup-cli] stats      = %s", stats)
    if not ran:
        logging.info("[warmup-cli] no compile work (sentinel hit).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
