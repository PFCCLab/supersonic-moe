"""Unified JIT cache management: Triton, CuTe, in-memory dicts.

Sets up all JIT cache paths to a single user-specified root (NFS/GPFS)
so caches survive container restarts.

Usage:
    SONIC_MOE_CACHE_DIR=/path/to/nfs/cache  # environment variable, or
    from sonicmoe.cache_manager import setup_cache
    setup_cache("/path/to/nfs/cache")        # Python API

Logging:
    Set SONIC_MOE_JIT_VERBOSE=1 to see compile timing / cache hits.
"""

import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path

_log = logging.getLogger("sonicmoe.jit")

DEFAULT_CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".jit_cache",
)

_VERBOSE = os.environ.get("SONIC_MOE_JIT_VERBOSE", "0") == "1"

# Global counter for progress estimation
_compile_count = 0
_compile_start_time: float | None = None
_EXPECTED_KERNEL_COUNT = 45  # ~15 CuTe + ~30 Triton


def _is_verbose() -> bool:
    return _VERBOSE or os.environ.get("SONIC_MOE_JIT_VERBOSE", "0") == "1"


def get_cache_root() -> Path:
    return Path(os.getenv("SONIC_MOE_CACHE_DIR", DEFAULT_CACHE_ROOT))


def setup_cache(cache_root: str | None = None) -> Path:
    """Initialize all JIT caches under a unified root path.

    Subdirectory structure::

        {cache_root}/
          triton/           <- TRITON_CACHE_DIR
          quack/            <- QUACK_CACHE_DIR
          sonicmoe/         <- reserved for future .o persistence
          version.json      <- git hash + timestamp for manual invalidation

    Parameters
    ----------
    cache_root : str, optional
        Override cache root.  Falls back to SONIC_MOE_CACHE_DIR env var,
        then to ``{project_root}/.jit_cache``.

    Returns
    -------
    Path
        The resolved cache root.
    """
    root = Path(cache_root) if cache_root else get_cache_root()
    for sub in ("triton", "quack", "sonicmoe"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    # Triton env vars
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton")
    os.environ["TRITON_HOME"] = str(root)
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"

    # CuTe/quack env vars
    os.environ["QUACK_CACHE_DIR"] = str(root / "quack")

    # Version info
    _write_version_info(root / "version.json")

    if _is_verbose():
        _log.info(f"[JIT] Cache root: {root}")

    return root


def _write_version_info(path: Path):
    """Record cache version info for manual invalidation."""
    import subprocess

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_hash = "unknown"
    info = {
        "git_hash": git_hash,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        path.write_text(json.dumps(info, indent=2))
    except OSError:
        pass


class InstrumentedCompileCache:
    """Drop-in replacement for ``dict`` compile caches with timing + logging.

    L1: in-memory dict (same as original behavior).
    Adds compile timing, cache hit/miss logging, and progress estimation.

    Usage (replaces ``_COMPILE_CACHE: dict = {}``)::

        _COMPILE_CACHE = InstrumentedCompileCache("bfp8_grouped")
        ...
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            compiled = cute.compile(...)
            _COMPILE_CACHE[compile_key] = compiled
    """

    def __init__(self, name: str):
        self.name = name
        self._mem: dict = {}
        self._hits = 0
        self._misses = 0

    def get(self, key, default=None):
        if key in self._mem:
            self._hits += 1
            if _is_verbose():
                _log.info(f"[JIT] {self.name}: cache hit (total: {len(self._mem)} entries)")
            return self._mem[key]
        self._misses += 1
        if _is_verbose():
            _log.info(f"[JIT] {self.name}: cache miss — compiling...")
        return default

    def __setitem__(self, key, value):
        global _compile_count, _compile_start_time
        _compile_count += 1
        if _compile_start_time is None:
            _compile_start_time = time.perf_counter()

        self._mem[key] = value

        if _is_verbose():
            elapsed = time.perf_counter() - _compile_start_time
            _log.info(
                f"[JIT] {self.name}: compiled "
                f"({_compile_count}/~{_EXPECTED_KERNEL_COUNT} kernels, "
                f"{elapsed:.1f}s elapsed)"
            )

    def __getitem__(self, key):
        return self._mem[key]

    def __contains__(self, key):
        hit = key in self._mem
        if hit:
            self._hits += 1
            if _is_verbose():
                _log.info(f"[JIT] {self.name}: cache hit")
        else:
            self._misses += 1
            if _is_verbose():
                _log.info(f"[JIT] {self.name}: cache miss — compiling...")
        return hit

    def __delitem__(self, key):
        del self._mem[key]

    def clear(self):
        self._mem.clear()

    def stats(self) -> dict:
        return {
            "name": self.name,
            "entries": len(self._mem),
            "hits": self._hits,
            "misses": self._misses,
        }

    def __repr__(self):
        return f"InstrumentedCompileCache({self.name!r}, entries={len(self._mem)})"

    def __len__(self) -> int:
        return len(self._mem)

    def __iter__(self):
        return iter(self._mem)


def cache_stats() -> dict:
    """Return disk cache statistics."""
    root = get_cache_root()
    stats = {}
    for sub in ("triton", "quack", "sonicmoe"):
        p = root / sub
        if p.exists():
            files = list(p.rglob("*"))
            stats[sub] = {
                "file_count": sum(1 for f in files if f.is_file()),
                "total_bytes": sum(f.stat().st_size for f in files if f.is_file()),
            }
    return stats


# ── Skip-if-warm sentinel ─────────────────────────────────────────────────────
#
# Triton's TRITON_CACHE_DIR and Quack's QUACK_CACHE_DIR are both file-backed
# disk caches that are safely shared across processes / nodes via NFS. A
# successful warmup leaves them populated with N kernel.cubin / N
# {fn_name}.autotune.json files. We record the (E, H, I, fp8, git_hash,
# kernel_signature_version) tuple on which the warmup ran in a sentinel JSON
# file. On subsequent process starts, ``is_warm()`` checks the sentinel
# matches and that the on-disk file counts have not regressed; if so,
# ``warmup_jit`` can be skipped entirely.
#
# CuTe's in-memory ``_COMPILE_CACHE`` is NOT covered (cute.compile artifacts
# are not picklable; see Phase C investigation in S76 HANDOFF). For CuTe the
# warmup still has to run once per process — but Triton autotune-cache and
# Quack autotune-cache hits eliminate the bulk of the autotune time.

_KERNEL_SIG_VERSION = "v1"


def _warmup_sentinel_path(root: Path | None = None) -> Path:
    if root is None:
        root = get_cache_root()
    return root / "warmup_sentinel.json"


def _git_hash_or_none() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def warmup_signature(E: int, H: int, I: int, fp8: bool) -> dict:
    """Stable identity of a warmup invocation."""
    return {
        "E": E, "H": H, "I": I, "fp8": fp8,
        "kernel_sig": _KERNEL_SIG_VERSION,
        "git_hash": _git_hash_or_none(),
    }


def is_warm(E: int, H: int, I: int, fp8: bool = True,
            *, min_triton_files: int = 1, min_quack_files: int = 0) -> bool:
    """True iff a previous ``warmup_jit`` for this signature is still on disk.

    Consults the sentinel JSON written at the cache root and verifies that
    the Triton disk cache is non-empty (regression guard against sentinel
    surviving a manual ``rm -rf .jit_cache/triton``). When this returns
    True, ``warmup_jit`` may be skipped — the running process will still
    need to populate its own in-memory CuTe cache on first call, but that
    incurs no autotune / ptxas cost (everything reloads from disk).

    NOTE: ``min_quack_files`` defaults to 0 because the CuTe (Quack)
    persistent disk cache is currently BLOCKED upstream (see S76 handoff).
    Once it is restored, raise the default to 1 for tighter regression
    guarding.
    """
    root = get_cache_root()
    sentinel = _warmup_sentinel_path(root)
    if not sentinel.exists():
        return False
    try:
        rec = json.loads(sentinel.read_text())
    except Exception:
        return False
    sig = warmup_signature(E, H, I, fp8)
    # git_hash is informational; mismatch is OK if user opts in.
    must_match = ("E", "H", "I", "fp8", "kernel_sig")
    if any(rec.get(k) != sig[k] for k in must_match):
        return False
    if rec.get("git_hash") != sig["git_hash"] and not os.environ.get(
        "SONIC_MOE_WARMUP_IGNORE_GIT", "0"
    ) == "1":
        return False
    stats = cache_stats()
    triton_n = stats.get("triton", {}).get("file_count", 0)
    quack_n = stats.get("quack", {}).get("file_count", 0)
    if triton_n < min_triton_files or quack_n < min_quack_files:
        return False
    return True


def mark_warm(E: int, H: int, I: int, fp8: bool = True) -> Path:
    """Write the sentinel after a successful ``warmup_jit`` run.

    Atomic rename so concurrent parallel-warmup writers cannot leave a
    partially-written sentinel that would fail JSON decode in ``is_warm``.
    """
    root = get_cache_root()
    sentinel = _warmup_sentinel_path(root)
    sig = warmup_signature(E, H, I, fp8)
    stats = cache_stats()
    sig["cache_stats"] = stats
    sig["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = json.dumps(sig, indent=2)
    tmp = sentinel.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(payload)
    os.replace(tmp, sentinel)
    if _is_verbose():
        _log.info(f"[JIT] Warmup sentinel: {sentinel}")
    return sentinel


def clear_warmup_sentinel() -> None:
    """Force the next ``warmup_jit`` to run from scratch."""
    sentinel = _warmup_sentinel_path()
    if sentinel.exists():
        sentinel.unlink()


def clear_all_caches():
    """Clear all JIT caches (disk only — in-memory caches are per-process)."""
    import shutil

    root = get_cache_root()
    for sub in ("triton", "quack", "sonicmoe"):
        p = root / sub
        if p.exists():
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
    clear_warmup_sentinel()
    if _is_verbose():
        _log.info(f"[JIT] Cleared all caches under {root}")
