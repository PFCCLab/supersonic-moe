#!/usr/bin/env python3
"""SonicMoE Introspection Engine — Zero-Code-Change Model Analysis.

Performs an instrumented dry-run of the MoE forward+backward pass.
Extracts buffer lifecycle, memory trajectory, kernel timing, and
precision data into a structured ``manifest.json`` consumed by the
visualization suite (``python -m visualization``).

**Non-invasive**: no changes to sonicmoe/ source code.  All
instrumentation uses PyTorch's public hook APIs + monkey-patching
of autograd Function boundaries.

Session 46 improvements:
  - 15-iteration warmup (was 3/5) for Triton JIT stability
  - CUDA-event timing alongside wallclock in kernel profiler
  - 5-seed precision audit (was 3)
  - 300s subprocess timeout (was 120) for JIT compilation
  - Compatible with SonicMoEConfig Pythonic config API

Modes
-----
  trace      — shapes / dtypes / lifecycle / memory (~3 s)
  profile    — trace + kernel timing via torch.profiler (~30 s)
  full       — trace + profile + precision audit (~60 s)
  nsys       — nsys GPU-projection profiling (~120 s per shape×mode)
  grid       — parallel 27-shape (3T×3E×3I) nsys profiling across 8 GPUs
  quant-bench — isolated CUDA-event quant kernel benchmark (all variants)
  wgrad-bench — FP8 vs BF16 wgrad end-to-end benchmark with memory
  ncu-bench  — NCU --clock-control=none quant kernel analysis (realistic timing)
  wgrad-force — forced wgrad FP8 at all shapes (bypass I-threshold) + memory breakdown

Usage
-----
    python tools/introspect.py                        # trace mode
    python tools/introspect.py --mode profile         # + kernel timing
    python tools/introspect.py --mode full            # everything
    python tools/introspect.py --mode nsys            # gold-standard GPU timing
    python tools/introspect.py --mode nsys --nsys-shapes 8192,3072,1536,8,8 8192,3072,2048,8,8
    python tools/introspect.py --mode grid --gpu 8    # 27-shape grid on 8 GPUs
    python tools/introspect.py --mode quant-bench     # quant kernel micro-benchmark
    python tools/introspect.py --mode quant-bench --quant-bench-shapes 65536,3072,3072,1536
    python tools/introspect.py --mode wgrad-bench     # FP8 vs BF16 wgrad timing + memory

Output: ``manifest.json`` at repo root.
"""
from __future__ import annotations

import argparse
import collections
import gc
import importlib.util
import inspect
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "manifest.json"
MANIFEST_VERSION = 2
BENCHMARK_FINAL_PATH = ROOT / "benchmark_final.json"
MEM_BREAKDOWN_PATH = ROOT / "mem_breakdown.json"
KERNEL_BREAKDOWN_ROOT_PATH = ROOT / "kernel_breakdown.json"
KERNEL_BREAKDOWN_COMPAT_PATH = ROOT / "reports" / "nsys_final" / "kernel_breakdown.json"
NSYS_BREAKDOWN_PATH = ROOT / "reports" / "nsys_final" / "nsys_gpu_projection.json"

# Default reference shape
SHAPE = {"T": 8192, "H": 3072, "I": 1536, "E": 8, "K": 8}
DEFAULT_SHAPE = dict(SHAPE)
DEFAULT_PRECISION_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_BENCH_REPEATS = 3
DEFAULT_NSYS_WARMUP = 5
DEFAULT_NSYS_ITERS = 20
PERSISTENT_TMP_ROOT = Path(
    os.environ.get("SONIC_MOE_PERSISTENT_TMP_ROOT", str(ROOT / "reports" / "tmp"))
)

# ── Grid benchmark: 3T × 3E × 3I = 27 shapes ──────────────────────────────
GRID_T = [8192, 16384, 32768]
GRID_E = [8, 32, 128]
GRID_I = [1536, 2048, 3072]
GRID_H = 3072  # fixed hidden dim
GRID_K = 8     # fixed topK

# Python binary resolution: prefer the virtualenv that has quack/sonicmoe,
# fall back to sys.executable.  The old hardcoded xfer path is kept as the
# first candidate but is no longer a hard requirement.
_XFER_PYTHON = Path(os.environ.get("SONIC_MOE_XFER_PYTHON", sys.executable))


def _resolve_python_bin() -> str:
    """Return a working Python 3.13 binary that can import quack + sonicmoe."""
    for candidate in [str(_XFER_PYTHON), sys.executable]:
        p = Path(candidate)
        if not p.exists():
            continue
        try:
            subprocess.run(
                [str(p), "-c", "import quack, sonicmoe"],
                capture_output=True, timeout=30,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
            )
            return str(p)
        except Exception:
            continue
    # Last resort
    return sys.executable

# Map _log_stage_memory stage names → visualization phase IDs (0-5)
STAGE_TO_PHASE = {
    "forward:router-metadata": 0,
    "forward:up-proj": 1,
    "forward:fp8-boundary": 1,
    "forward:down-proj-router": 2,
    "backward:down-proj-dgated": 3,
    "backward:down-proj-weight": 3,
    "backward:down-proj-postact-release": 3,
    "backward:up-proj-core": 4,
    "backward:token-reduce": 5,
}

PHASE_NAMES = [
    "Router & Meta",
    "UpProj Fwd",
    "DnProj Fwd",
    "DnProj Bwd",
    "UpBwd (wgrad)",
    "UpBwd (actgrad)",
]

# UpProjection.forward ctx.save_for_backward ordering (8 tensors)
_UP_SAVE_NAMES = [
    "x", "w1", "b1", "expert_frequency_offset",
    "x_gather_idx", "s_scatter_idx", "s_reverse_scatter_idx",
    "num_activated_expert_per_token_offset",
]

# DownProjection.forward ctx.save_for_backward ordering — BF16 (8 tensors)
_DOWN_SAVE_NAMES_BF16 = [
    "z", "w2", "b2", "topk_scores",
    "expert_frequency_offset", "x_gather_idx",
    "s_scatter_idx", "s_reverse_scatter_idx",
]

# DownProjection.forward ctx.save_for_backward ordering — FP8 (9 tensors)
_DOWN_SAVE_NAMES_FP8 = [
    "z_fp8", "z_raw_scales", "w2", "b2", "topk_scores",
    "expert_frequency_offset", "x_gather_idx",
    "s_scatter_idx", "s_reverse_scatter_idx",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TensorRecord:
    """Lifecycle record for a single GPU tensor."""
    name: str
    dtype: str
    shape: list[int]
    size_mib: float
    role: str = "activation"        # activation | weight | grad | scale | index
    create_phase: int = -1
    free_phase: int = -1            # last phase where the tensor is used
    events: list[str] = field(default_factory=list)


@dataclass
class PhaseMemory:
    """Memory snapshot at a phase boundary."""
    phase_id: int
    phase_name: str
    allocated_mib: float = 0.0
    peak_mib: float = 0.0
    reserved_mib: float = 0.0


@dataclass
class KernelRecord:
    """Single kernel timing record."""
    name: str
    category: str
    cuda_time_us: float
    count: int = 1


@dataclass
class ModeManifest:
    """Manifest data for a single mode (bf16 or fp8)."""
    mode: str
    tensors: list[TensorRecord] = field(default_factory=list)
    phase_memory: list[PhaseMemory] = field(default_factory=list)
    kernels: list[KernelRecord] = field(default_factory=list)
    memory_trajectory: dict[str, float] = field(default_factory=dict)
    total_cuda_us: float = 0.0
    wall_clock_ms: float = 0.0
    precision_matrix: list[list[int]] = field(default_factory=list)
    theoretical_memory: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: dtype to precision code (for precision matrix)
# ═══════════════════════════════════════════════════════════════════════════════

_DTYPE_CODE = {
    "torch.bfloat16": 1,
    "torch.float8_e4m3fn": 2,
    "torch.float32": 3,
    "torch.int32": 4,
    "torch.uint8": 5,     # ISA-packed scales
}


def _dtype_to_role(dtype_str: str) -> str:
    """Infer tensor role from dtype string."""
    if "float8" in dtype_str or "uint8" in dtype_str:
        return "scale" if "uint8" in dtype_str else "activation"
    if "int32" in dtype_str:
        return "index"
    if "float32" in dtype_str:
        return "activation"
    return "activation"


def _tensor_size_mib(t) -> float:
    """Tensor size in MiB."""
    if t is None:
        return 0.0
    return t.nelement() * t.element_size() / (1024 ** 2)


def _shape_tuple(shape: dict[str, int]) -> tuple[int, int, int, int, int]:
    return tuple(shape[k] for k in ("T", "H", "I", "E", "K"))


def _is_default_shape() -> bool:
    return _shape_tuple(SHAPE) == _shape_tuple(DEFAULT_SHAPE)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _safe_mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _stat_summary(values: list[float], digits: int = 3) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(_safe_mean(values), digits),
        "std": round(_safe_std(values), digits),
        "min": round(min(values), digits),
        "max": round(max(values), digits),
    }


def _load_python_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_subprocess_env(mode: str, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["USE_QUACK_GEMM"] = "1"
    # Respect shell-level CUDA_VISIBLE_DEVICES (e.g. from parallel launches).
    # Only set it when the parent hasn't already pinned a device.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    if mode == "fp8":
        env["SONIC_MOE_FP8_MODE"] = "perf"
    else:
        env.pop("SONIC_MOE_FP8_MODE", None)
    return env


def _subprocess_env_for_gpu(gpu: int) -> dict[str, str]:
    """Build a subprocess env dict that respects shell-level CUDA_VISIBLE_DEVICES."""
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _persistent_tempdir():
    temp_root = PERSISTENT_TMP_ROOT if PERSISTENT_TMP_ROOT.exists() else None
    return tempfile.TemporaryDirectory(dir=str(temp_root) if temp_root else None)


def _build_model_and_input(device):
    import torch
    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType

    torch.manual_seed(42)
    model = MoE(
        SHAPE["E"], SHAPE["K"], SHAPE["H"], SHAPE["I"],
        ActivationType.SWIGLU, False, 0.02,
    ).to(device).to(torch.bfloat16)
    x = 0.02 * torch.randn(
        SHAPE["T"], SHAPE["H"],
        dtype=torch.bfloat16, device=device, requires_grad=True,
    )
    return model, x


def _warmup_mode(model, x, use_fp8: bool, iters: int = 15) -> None:
    import torch
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

    if use_fp8 and hasattr(model, "refresh_fp8_shadow_weights"):
        model.refresh_fp8_shadow_weights()

    for _ in range(iters):
        xw = x.detach().clone().requires_grad_(True)
        with enable_quack_gemm(True):
            if use_fp8:
                with enable_fp8(True):
                    ow, lw = model(xw, use_fp8=True)
            else:
                ow, lw = model(xw)
        (ow.sum() + lw).backward()
        model.zero_grad(set_to_none=True)
        del ow, lw, xw

    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseTracker:
    """Tracks the current visualization phase by intercepting _log_stage_memory.

    Non-invasive: monkey-patches the module-level functions in
    ``sonicmoe.functional`` to capture stage transitions.
    """

    def __init__(self):
        self.current_phase: int = -1
        self.stage_log: list[tuple[str, int, dict]] = []
        self._memory_at_phase: dict[int, PhaseMemory] = {}
        self._installed = False
        self._orig_log = None
        self._orig_reset = None
        self._orig_debug_enabled = None

    def install(self):
        """Monkey-patch sonicmoe.functional stage-memory functions."""
        import sonicmoe.functional as F
        self._orig_log = F._log_stage_memory
        self._orig_reset = F._reset_stage_memory_probe
        self._orig_debug_enabled = F._stage_memory_debug_enabled

        tracker = self

        def _patched_debug_enabled() -> bool:
            return True  # always enabled during introspection

        def _patched_reset() -> None:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def _patched_log(stage: str) -> None:
            import torch
            torch.cuda.synchronize()
            mib = 1024 ** 2
            phase_id = STAGE_TO_PHASE.get(stage, tracker.current_phase)
            tracker.current_phase = phase_id

            mem = {
                "allocated_mib": torch.cuda.memory_allocated() / mib,
                "peak_mib": torch.cuda.max_memory_allocated() / mib,
                "reserved_mib": torch.cuda.memory_reserved() / mib,
            }
            tracker.stage_log.append((stage, phase_id, mem))

            if phase_id not in tracker._memory_at_phase or \
               mem["peak_mib"] > tracker._memory_at_phase[phase_id].peak_mib:
                tracker._memory_at_phase[phase_id] = PhaseMemory(
                    phase_id=phase_id,
                    phase_name=PHASE_NAMES[phase_id] if phase_id < len(PHASE_NAMES) else f"phase_{phase_id}",
                    allocated_mib=round(mem["allocated_mib"], 2),
                    peak_mib=round(mem["peak_mib"], 2),
                    reserved_mib=round(mem["reserved_mib"], 2),
                )

        F._stage_memory_debug_enabled = _patched_debug_enabled
        F._log_stage_memory = _patched_log
        F._reset_stage_memory_probe = _patched_reset
        self._installed = True

    def uninstall(self):
        """Restore original functions."""
        if not self._installed:
            return
        import sonicmoe.functional as F
        F._log_stage_memory = self._orig_log
        F._reset_stage_memory_probe = self._orig_reset
        F._stage_memory_debug_enabled = self._orig_debug_enabled
        self._installed = False

    def get_phase_memory(self) -> list[PhaseMemory]:
        """Return memory snapshots sorted by phase ID."""
        return [self._memory_at_phase[k] for k in sorted(self._memory_at_phase)]

    def reset(self):
        """Clear state for a new trace run."""
        self.current_phase = -1
        self.stage_log.clear()
        self._memory_at_phase.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor Spy
# ═══════════════════════════════════════════════════════════════════════════════

class TensorSpy:
    """Tracks tensor lifecycle using saved_tensors_hooks + forward/backward wrapping.

    Records: name, dtype, shape, size, create_phase, last_used_phase, role.
    """

    def __init__(self, phase_tracker: PhaseTracker):
        self.phase_tracker = phase_tracker
        self._ptr_to_name: dict[int, str] = {}      # data_ptr → name
        self._records: dict[str, TensorRecord] = {}  # name → TensorRecord
        self._pack_count = 0
        self._unpack_count = 0
        self._prequant_events: list[tuple[str, str, int]] = []  # (key, action, phase)

        # Monkey-patching state
        self._orig_up_fwd = None
        self._orig_up_bwd = None
        self._orig_down_fwd = None
        self._orig_down_bwd = None
        self._prequant_proxy = None
        self._installed = False

    def _record_tensor(self, name: str, tensor, phase: int,
                       role: str = "activation", event: str | None = None):
        """Register or update a tensor record."""
        if tensor is None:
            return
        import torch
        if not isinstance(tensor, torch.Tensor):
            return

        dtype_str = str(tensor.dtype)
        size_mib = round(_tensor_size_mib(tensor), 4)
        shape = list(tensor.shape)

        ptr = tensor.data_ptr()

        # --- Deduplication by data_ptr ---
        # The CUDA caching allocator reuses freed memory, so different tensors
        # at different phases may share a data_ptr.  Distinguish same-tensor
        # (e.g. topk_indices passed as selected_experts) from ptr-reuse by
        # checking shape+dtype match.  If shape/dtype differ, it's a new alloc.
        old_name = self._ptr_to_name.get(ptr)
        if old_name and old_name != name:
            old_rec = self._records.get(old_name)
            if old_rec is not None:
                same_shape_dtype = (old_rec.shape == shape
                                    and old_rec.dtype == dtype_str)
                if not same_shape_dtype:
                    # Different shape/dtype → definitely a data_ptr reuse
                    del self._ptr_to_name[ptr]
                    old_name = None
            if old_name and old_name.startswith(("saved_", "restored_")):
                # Auto-name → meaningful name: absorb old record
                old_rec_pop = self._records.pop(old_name, None)
                if old_rec_pop:
                    if name in self._records:
                        rec = self._records[name]
                        rec.create_phase = min(rec.create_phase,
                                               old_rec_pop.create_phase)
                        rec.free_phase = max(rec.free_phase,
                                             old_rec_pop.free_phase)
                        rec.events.extend(old_rec_pop.events)
                    else:
                        old_rec_pop.name = name
                        if role != "activation":
                            old_rec_pop.role = role
                        self._records[name] = old_rec_pop
            elif old_name and not name.startswith(("saved_", "restored_")):
                # Both meaningful — same physical tensor under different var
                # names (e.g. topk_indices == selected_experts).  Keep the
                # first-registered name and just extend its lifecycle.
                name = old_name

        if name in self._records:
            rec = self._records[name]
            if phase < rec.create_phase or rec.create_phase < 0:
                rec.create_phase = phase
            if phase > rec.free_phase:
                rec.free_phase = phase
            if event:
                rec.events.append(event)
        else:
            self._records[name] = TensorRecord(
                name=name,
                dtype=dtype_str,
                shape=shape,
                size_mib=size_mib,
                role=role,
                create_phase=phase,
                free_phase=phase,
                events=[event] if event else [],
            )
        self._ptr_to_name[ptr] = name

    def _infer_name_from_shape(self, tensor, phase: int) -> str | None:
        """Heuristic naming for tensors not caught by wrapper registration."""
        import torch
        shape = list(tensor.shape)
        dtype = tensor.dtype

        # Phase 0: Router tensors
        if phase == 0:
            if dtype == torch.float32 and len(shape) == 2 and shape[1] <= 32:
                return "topk_scores"   # [T, K] FP32
            if dtype == torch.int32 and len(shape) == 2 and shape[1] <= 32:
                return "topk_indices"  # [T, K] INT32

        # Phase 2: Aux loss intermediates (tiny FP32 tensors from _compute_switch_loss)
        if phase == 2 and dtype == torch.float32:
            if len(shape) == 1 and shape[0] == 1:
                return "aux_loss_norm"
            if len(shape) == 1 and shape[0] <= 32:
                n = sum(1 for k in self._records if k and k.startswith("aux_"))
                return f"aux_loss_{n}"
            if len(shape) == 2 and shape[1] <= 32:
                return "softmax_probs"  # [T, E] FP32 from F.softmax

        return None

    def _make_pack_hook(self):
        spy = self
        def pack_hook(tensor):
            ptr = tensor.data_ptr()
            phase = spy.phase_tracker.current_phase
            name = spy._ptr_to_name.get(ptr)
            if name is None:
                name = spy._infer_name_from_shape(tensor, phase)
            if name is None:
                name = f"saved_{spy._pack_count}"
            spy._record_tensor(name, tensor, phase, event=f"saved@phase{phase}")
            spy._pack_count += 1
            return tensor
        return pack_hook

    def _make_unpack_hook(self):
        spy = self
        def unpack_hook(tensor):
            ptr = tensor.data_ptr()
            phase = spy.phase_tracker.current_phase
            name = spy._ptr_to_name.get(ptr, f"restored_{spy._unpack_count}")
            spy._record_tensor(name, tensor, phase, event=f"restored@phase{phase}")
            spy._unpack_count += 1
            return tensor
        return unpack_hook

    def install(self, model):
        """Install all hooks and monkey-patches."""
        import torch
        from sonicmoe.functional import _UpProjection, _DownProjection

        # 1. Register all model parameters
        for pname, param in model.named_parameters():
            short = self._param_short_name(pname)
            self._record_tensor(short, param.data, phase=0, role="weight")

        # 2. Wrap _UpProjection.forward to capture arg names
        spy = self
        self._orig_up_fwd = _UpProjection.forward

        @staticmethod
        def _traced_up_fwd(ctx, x, w1, b1, expert_frequency_offset,
                           total_expert_freq, K, stream_id, x_gather_idx,
                           s_scatter_idx, s_reverse_scatter_idx,
                           num_activated_expert_per_token_offset,
                           is_varlen_K, activation_type,
                           is_inference_mode_enabled,
                           use_low_precision_postact_buffer):
            phase = 1
            spy.phase_tracker.current_phase = phase
            # Record forward args
            spy._record_tensor("x", x, phase)
            spy._record_tensor("w1", w1, phase, role="weight")
            spy._record_tensor("x_gather_idx", x_gather_idx, phase, role="index")
            spy._record_tensor("s_scatter_idx", s_scatter_idx, phase, role="index")
            spy._record_tensor("s_reverse_scatter_idx", s_reverse_scatter_idx, phase, role="index")
            spy._record_tensor("expert_freq_offset", expert_frequency_offset, phase, role="index")
            spy._record_tensor("num_activated_expert_offset",
                               num_activated_expert_per_token_offset, phase, role="index")

            result = spy._orig_up_fwd(
                ctx, x, w1, b1, expert_frequency_offset,
                total_expert_freq, K, stream_id, x_gather_idx,
                s_scatter_idx, s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                is_varlen_K, activation_type,
                is_inference_mode_enabled,
                use_low_precision_postact_buffer,
            )

            y1, z = result
            spy._record_tensor("y1", y1, phase, event="created@UpProj")
            spy._record_tensor("z", z, phase, event="created@UpProj")
            return result

        _UpProjection.forward = _traced_up_fwd

        # 3. Wrap _UpProjection.backward
        self._orig_up_bwd = _UpProjection.backward

        @staticmethod
        def _traced_up_bwd(ctx, grad_y1, dz):
            spy.phase_tracker.current_phase = 4
            if dz is not None:
                spy._record_tensor("dz", dz, 4, role="grad")

            result = spy._orig_up_bwd(ctx, grad_y1, dz)

            # result = (dx_reduced, dw1, db1, None×12)
            if result[0] is not None:
                spy._record_tensor("dx", result[0], 5, role="grad")
            if result[1] is not None:
                spy._record_tensor("dw1", result[1], 4, role="grad")
            spy.phase_tracker.current_phase = 5
            return result

        _UpProjection.backward = _traced_up_bwd

        # 4. Wrap _DownProjection.forward
        self._orig_down_fwd = _DownProjection.forward

        @staticmethod
        def _traced_down_fwd(ctx, y1, z, w2, b2, topk_scores,
                             selected_experts, expert_frequency_offset,
                             T, K, stream_id, x_gather_idx,
                             s_scatter_idx, s_reverse_scatter_idx,
                             num_activated_expert_per_token_offset,
                             is_varlen_K, activation_type, fp8_protocol):
            phase = 2
            spy.phase_tracker.current_phase = phase
            spy._record_tensor("y1", y1, phase)
            spy._record_tensor("z", z, phase)
            spy._record_tensor("w2", w2, phase, role="weight")
            spy._record_tensor("topk_scores", topk_scores, phase)
            spy._record_tensor("selected_experts", selected_experts, phase, role="index")
            spy._record_tensor("expert_freq_offset", expert_frequency_offset, phase, role="index")
            spy._record_tensor("x_gather_idx", x_gather_idx, phase, role="index")
            spy._record_tensor("s_scatter_idx", s_scatter_idx, phase, role="index")
            spy._record_tensor("s_reverse_scatter_idx", s_reverse_scatter_idx, phase, role="index")

            result = spy._orig_down_fwd(
                ctx, y1, z, w2, b2, topk_scores,
                selected_experts, expert_frequency_offset,
                T, K, stream_id, x_gather_idx,
                s_scatter_idx, s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                is_varlen_K, activation_type, fp8_protocol,
            )

            spy._record_tensor("output", result, phase, event="DnProj_out")
            return result

        _DownProjection.forward = _traced_down_fwd

        # 5. Wrap _DownProjection.backward
        self._orig_down_bwd = _DownProjection.backward

        @staticmethod
        def _traced_down_bwd(ctx, dout):
            spy.phase_tracker.current_phase = 3
            spy._record_tensor("dout", dout, 3, role="grad")

            result = spy._orig_down_bwd(ctx, dout)

            # result = (dy1, dz, dw2, db2, ..., Nones)
            if result[1] is not None:
                spy._record_tensor("dz", result[1], 3, role="grad",
                                   event="created@DnProj_bwd")
            if result[2] is not None:
                spy._record_tensor("dw2", result[2], 3, role="grad")
            return result

        _DownProjection.backward = _traced_down_bwd

        # 6. Monitor _PREQUANTIZED_SCALES via monkey-patched dict
        from sonicmoe.functional import _PREQUANTIZED_SCALES
        self._install_prequant_monitor(_PREQUANTIZED_SCALES)

        self._installed = True

    def _install_prequant_monitor(self, original_dict):
        """Wrap _PREQUANTIZED_SCALES dict to monitor FP8 tensor caching."""
        spy = self
        import sonicmoe.functional as F

        class _MonitoredDict(dict):
            def __setitem__(self, key, value):
                phase = spy.phase_tracker.current_phase
                spy._prequant_events.append((key, "store", phase))
                # Record FP8 tensors from the cache
                import torch
                if isinstance(value, tuple):
                    for i, v in enumerate(value):
                        if isinstance(v, torch.Tensor):
                            suffix = f"_{i}" if len(value) > 1 else ""
                            tname = f"prequant_{key}{suffix}"
                            dtype_str = str(v.dtype)
                            if "float8" in dtype_str:
                                tname = f"{key}" if "fp8" in key else f"{key}_fp8"
                            elif "uint8" in dtype_str:
                                tname = f"{key}_scales"
                            spy._record_tensor(tname, v, phase,
                                               role="scale" if "scale" in tname else "activation",
                                               event=f"prequant_store@phase{phase}")
                super().__setitem__(key, value)

            def pop(self, key, *args):
                phase = spy.phase_tracker.current_phase
                spy._prequant_events.append((key, "consume", phase))
                # Extend lifecycle of tensors stored under this key
                import torch
                value = self.get(key)
                if isinstance(value, tuple):
                    for i, v in enumerate(value):
                        if isinstance(v, torch.Tensor):
                            suffix = f"_{i}" if len(value) > 1 else ""
                            tname = f"prequant_{key}{suffix}"
                            dtype_str = str(v.dtype)
                            if "float8" in dtype_str:
                                tname = f"{key}" if "fp8" in key else f"{key}_fp8"
                            elif "uint8" in dtype_str:
                                tname = f"{key}_scales"
                            spy._record_tensor(tname, v, phase,
                                               role="scale" if "scale" in tname else "activation",
                                               event=f"prequant_consume@phase{phase}")
                return super().pop(key, *args)

        monitored = _MonitoredDict(original_dict)
        monitored.update(original_dict)
        F._PREQUANTIZED_SCALES = monitored
        self._prequant_proxy = monitored

    def uninstall(self):
        """Restore all original methods."""
        if not self._installed:
            return
        from sonicmoe.functional import _UpProjection, _DownProjection
        import sonicmoe.functional as F

        if self._orig_up_fwd is not None:
            _UpProjection.forward = self._orig_up_fwd
        if self._orig_up_bwd is not None:
            _UpProjection.backward = self._orig_up_bwd
        if self._orig_down_fwd is not None:
            _DownProjection.forward = self._orig_down_fwd
        if self._orig_down_bwd is not None:
            _DownProjection.backward = self._orig_down_bwd

        # Restore original PREQUANTIZED_SCALES dict
        if self._prequant_proxy is not None:
            original = dict(self._prequant_proxy)
            F._PREQUANTIZED_SCALES = original
        self._installed = False

    def get_records(self) -> list[TensorRecord]:
        """Return all tracked tensor records sorted by create_phase."""
        recs = sorted(self._records.values(), key=lambda r: (r.create_phase, r.name))
        return recs

    def reset(self):
        """Clear state for a new run."""
        self._ptr_to_name.clear()
        self._records.clear()
        self._pack_count = 0
        self._unpack_count = 0
        self._prequant_events.clear()

    @staticmethod
    def _param_short_name(full_name: str) -> str:
        """Map model parameter name to visualization name."""
        mapping = {
            "router.weight": "router_w",
            "c_fc.weight": "w1",
            "c_fc.bias": "b1",
            "c_proj.weight": "w2",
            "c_proj.bias": "b2",
        }
        return mapping.get(full_name, full_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Precision Matrix Builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_precision_matrix(tensors: list[TensorRecord], n_phases: int = 6) -> list[list[int]]:
    """Build 13×6 precision matrix from tensor records.

    Encoding: 0=absent, 1=BF16, 2=FP8, 3=FP32, 4=INT32, 5=SCALE/UINT8.
    """
    matrix = []
    for t in tensors:
        row = []
        code = _DTYPE_CODE.get(t.dtype, 0)
        for p in range(n_phases):
            if t.create_phase <= p <= t.free_phase:
                row.append(code)
            else:
                row.append(0)
        matrix.append(row)
    return matrix


def _compute_theoretical_memory(
    tensors: list[TensorRecord],
    phase_memory: list[PhaseMemory],
    n_phases: int = 6,
) -> dict[str, dict[str, Any]]:
    """Estimate tracked live memory per phase from tensor lifetimes."""
    phase_map = {pm.phase_id: pm for pm in phase_memory}
    theory: dict[str, dict[str, Any]] = {}
    for phase in range(n_phases):
        alive = [t for t in tensors if t.create_phase <= phase <= t.free_phase]
        total_mib = round(sum(t.size_mib for t in alive), 2)
        top_tensors = sorted(alive, key=lambda t: -t.size_mib)[:5]
        entry: dict[str, Any] = {
            "phase_name": PHASE_NAMES[phase] if phase < len(PHASE_NAMES) else f"phase_{phase}",
            "tracked_live_mib": total_mib,
            "tensor_count": len(alive),
            "top_tensors": [
                {"name": t.name, "size_mib": round(t.size_mib, 2), "dtype": t.dtype}
                for t in top_tensors
            ],
        }
        measured = phase_map.get(phase)
        if measured is not None:
            entry["measured_allocated_mib"] = measured.allocated_mib
            entry["measured_peak_mib"] = measured.peak_mib
            entry["gap_vs_peak_mib"] = round(measured.peak_mib - total_mib, 2)
        theory[str(phase)] = entry
    return theory


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Trajectory
# ═══════════════════════════════════════════════════════════════════════════════

def _capture_memory_trajectory(model, x, use_fp8: bool) -> dict[str, float]:
    """Run forward + backward and capture memory at 4 key checkpoints."""
    import torch
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

    mib = 1024 ** 2

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    pre_fwd = torch.cuda.memory_allocated() / mib

    x_run = x.detach().clone().requires_grad_(True)
    with enable_quack_gemm(True):
        if use_fp8:
            if hasattr(model, "refresh_fp8_shadow_weights"):
                model.refresh_fp8_shadow_weights()
            with enable_fp8(True):
                out, loss_val = model(x_run, use_fp8=True)
        else:
            out, loss_val = model(x_run)
    torch.cuda.synchronize()
    peak_fwd = torch.cuda.max_memory_allocated() / mib

    torch.cuda.reset_peak_memory_stats()
    pre_bwd = torch.cuda.memory_allocated() / mib
    loss = out.sum() + loss_val
    loss.backward()
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated() / mib

    del out, loss, loss_val, x_run
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cleanup = torch.cuda.memory_allocated() / mib

    return {
        "pre_fwd": round(pre_fwd, 2),
        "peak_fwd": round(peak_fwd, 2),
        "pre_bwd": round(pre_bwd, 2),
        "peak_bwd": round(peak_bwd, 2),
        "cleanup": round(cleanup, 2),
        "fwd_peak_above_pre": round(peak_fwd - pre_fwd, 2),
        "bwd_peak_above_pre": round(peak_bwd - pre_bwd, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Trace Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_trace(mode: str, model, x, device,
              phase_tracker: PhaseTracker,
              tensor_spy: TensorSpy) -> ModeManifest:
    """Run an instrumented forward+backward pass, capture tensor lifecycle + memory.

    Parameters
    ----------
    mode : str
        "bf16" or "fp8"
    model : MoE
        The model to trace.
    x : torch.Tensor
        Input tensor [T, H].
    device : torch.device
        CUDA device.
    phase_tracker : PhaseTracker
        Installed phase tracker.
    tensor_spy : TensorSpy
        Installed tensor spy.

    Returns
    -------
    ModeManifest
        Trace data for this mode.
    """
    import torch
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

    use_fp8 = (mode == "fp8")

    # Reset trackers
    phase_tracker.reset()
    tensor_spy.reset()

    # Re-register parameters (they may have been cleared)
    for pname, param in model.named_parameters():
        short = tensor_spy._param_short_name(pname)
        tensor_spy._record_tensor(short, param.data, phase=0, role="weight")

    # Clean CUDA state
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Run instrumented forward + backward with saved_tensors_hooks
    x_run = x.detach().clone().requires_grad_(True)
    if use_fp8 and hasattr(model, "refresh_fp8_shadow_weights"):
        model.refresh_fp8_shadow_weights()

    # Pre-register input so F.linear's autograd save finds the correct name
    tensor_spy._record_tensor("x", x_run, phase=0)

    with torch.autograd.graph.saved_tensors_hooks(
        tensor_spy._make_pack_hook(),
        tensor_spy._make_unpack_hook(),
    ):
        phase_tracker.current_phase = 0
        with enable_quack_gemm(True):
            if use_fp8:
                with enable_fp8(True):
                    out, loss_val = model(x_run, use_fp8=True)
            else:
                out, loss_val = model(x_run)
        torch.cuda.synchronize()

        # Backward
        loss = out.sum() + loss_val
        loss.backward()
        torch.cuda.synchronize()

    # Collect results
    manifest = ModeManifest(mode=mode)
    manifest.tensors = tensor_spy.get_records()
    manifest.phase_memory = phase_tracker.get_phase_memory()
    manifest.precision_matrix = _build_precision_matrix(manifest.tensors)

    # Capture memory trajectory (clean run without hooks)
    del out, loss, loss_val, x_run
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    manifest.memory_trajectory = _capture_memory_trajectory(model, x, use_fp8)
    manifest.theoretical_memory = _compute_theoretical_memory(
        manifest.tensors, manifest.phase_memory
    )

    return manifest


def _run_trace_worker(trace_mode: str) -> dict[str, Any]:
    """Worker entrypoint used by the subprocess-isolated trace runner."""
    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model, x = _build_model_and_input(device)
    _warmup_mode(model, x, use_fp8=(trace_mode == "fp8"))

    phase_tracker = PhaseTracker()
    tensor_spy = TensorSpy(phase_tracker)
    phase_tracker.install()
    tensor_spy.install(model)
    try:
        manifest = run_trace(trace_mode, model, x, device, phase_tracker, tensor_spy)
    finally:
        tensor_spy.uninstall()
        phase_tracker.uninstall()

    return {
        "mode": manifest.mode,
        "tensors": [asdict(t) for t in manifest.tensors],
        "phase_memory": [asdict(pm) for pm in manifest.phase_memory],
        "memory_trajectory": manifest.memory_trajectory,
        "precision_matrix": manifest.precision_matrix,
        "theoretical_memory": manifest.theoretical_memory,
    }


def run_trace_isolated(trace_mode: str, gpu: int) -> ModeManifest:
    """Run one trace in a clean subprocess to avoid FP8 mode contamination."""
    shape_arg = ",".join(str(SHAPE[k]) for k in ("T", "H", "I", "E", "K"))
    env = _build_subprocess_env(trace_mode, gpu)
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--shape", shape_arg,
            "--_worker-trace", trace_mode,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Trace worker [{trace_mode}] failed:\n"
            f"{proc.stderr[-2000:] or proc.stdout[-2000:]}"
        )

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("__TRACE_JSON__"):
            payload = json.loads(line[len("__TRACE_JSON__"):])
            break
    if payload is None:
        raise RuntimeError(f"Trace worker [{trace_mode}] emitted no JSON payload")

    manifest = ModeManifest(mode=payload["mode"])
    manifest.tensors = [TensorRecord(**t) for t in payload.get("tensors", [])]
    manifest.phase_memory = [PhaseMemory(**pm) for pm in payload.get("phase_memory", [])]
    manifest.memory_trajectory = payload.get("memory_trajectory", {})
    manifest.precision_matrix = payload.get("precision_matrix", [])
    manifest.theoretical_memory = payload.get("theoretical_memory", {})
    return manifest


def _run_collect_worker(mode: str, seed: int, out_path: Path) -> None:
    """Worker entrypoint: materialize outputs and grads for precision audit.

    Uses moe_TC_softmax_topk_layer for ALL expert counts — it handles
    route-level padding internally for E>8.  This matches the production
    code path and avoids the manual token-rounding branch.
    """
    import torch
    import sonicmoe.functional as functional
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
    from sonicmoe.functional import moe_TC_softmax_topk_layer
    from sonicmoe.enums import ActivationType

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model, _ = _build_model_and_input(device)
    E, T, H, K = SHAPE["E"], SHAPE["T"], SHAPE["H"], SHAPE["K"]

    functional._ALIGNMENT_ASSUMED = True

    if mode == "fp8":
        model.refresh_fp8_shadow_weights()
        model.stash_bf16_to_cpu()

    torch.manual_seed(seed)
    x = (0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device)).detach().requires_grad_(True)

    w1_p = model.c_fc.weight.permute(1, 2, 0)
    w2_p = model.c_proj.weight.permute(1, 2, 0)

    # Unified path: moe_TC_softmax_topk_layer handles route-level padding
    # internally for all E values (including E>8).
    out, _, _ = moe_TC_softmax_topk_layer(
        x, model.router.weight, w1_p, None, w2_p, None,
        K, model.stream_id, ActivationType.SWIGLU, False,
    )

    out.sum().backward()
    torch.cuda.synchronize()

    payload = {
        "output": out.detach().cpu(),
        "dx": x.grad.detach().cpu() if x.grad is not None else None,
        "dw1": model.c_fc.weight.grad.detach().cpu() if model.c_fc.weight.grad is not None else None,
        "dw2": model.c_proj.weight.grad.detach().cpu() if model.c_proj.weight.grad is not None else None,
    }
    torch.save(payload, out_path)


def run_precision_audit_isolated(gpu: int, seeds: list[int]) -> dict[str, Any]:
    """Precision audit with one subprocess per mode/seed."""
    import torch

    def _collect(mode: str, seed: int, out_path: Path) -> None:
        env = _build_subprocess_env(mode, gpu)
        shape_arg = ",".join(str(SHAPE[k]) for k in ("T", "H", "I", "E", "K"))
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--shape", shape_arg,
                "--_worker-collect", mode,
                "--_worker-seed", str(seed),
                "--_worker-output", str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(ROOT),
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Precision worker [{mode}, seed={seed}] failed:\n"
                f"{proc.stderr[-2000:] or proc.stdout[-2000:]}"
            )

    def _rrmse(a, b):
        diff = (a.float() - b.float())
        denom = b.float().norm().clamp(min=1e-8)
        return float((diff.norm() / denom * 100).item())

    def _cosine(a, b):
        a_flat = a.flatten().double()
        b_flat = b.flatten().double()
        denom = (a_flat.norm() * b_flat.norm()).clamp(min=1e-12)
        val = float((a_flat * b_flat).sum().item() / denom.item())
        return max(-1.0, min(1.0, val))

    metric_store: dict[str, dict[str, list[float]]] = {
        "rrmse_pct": collections.defaultdict(list),
        "cosine_sim": collections.defaultdict(list),
    }

    with _persistent_tempdir() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for seed in seeds:
            bf16_path = tmpdir_path / f"bf16_s{seed}.pt"
            fp8_path = tmpdir_path / f"fp8_s{seed}.pt"
            _collect("bf16", seed, bf16_path)
            _collect("fp8", seed, fp8_path)
            bf16_out = torch.load(bf16_path, weights_only=False)
            fp8_out = torch.load(fp8_path, weights_only=False)

            for key in ["output", "dx", "dw1", "dw2"]:
                ref = bf16_out.get(key)
                test = fp8_out.get(key)
                if ref is None or test is None:
                    continue
                metric_store["rrmse_pct"][key].append(_rrmse(test, ref))
                metric_store["cosine_sim"][key].append(_cosine(test, ref))

    audit = {
        "seeds": seeds,
        "rrmse_pct": {},
        "cosine_sim": {},
        "stats": {"rrmse_pct": {}, "cosine_sim": {}},
    }
    for group in ("rrmse_pct", "cosine_sim"):
        for key, values in metric_store[group].items():
            if not values:
                continue
            digits = 4 if group == "cosine_sim" else 3
            audit[group][key] = round(_safe_mean(values), digits)
            audit["stats"][group][key] = _stat_summary(values, digits=digits)

    return audit


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel Profiler (subprocess-isolated)
# ═══════════════════════════════════════════════════════════════════════════════

_KERNEL_PROFILE_SCRIPT = r'''
import gc, json, os, sys, time, torch
from collections import defaultdict

MODE = os.environ["_PROFILER_MODE"]
T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
device = torch.device("cuda:0")
torch.cuda.set_device(device)

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
torch.manual_seed(42)
model = MoE(E, K, H, I, ActivationType.SWIGLU, False, 0.02).to(device).to(torch.bfloat16)
x = 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)

use_fp8 = (MODE == "fp8")

# Warmup (15 iterations for Triton JIT stability)
for _ in range(15):
    xw = x.detach().clone().requires_grad_(True)
    with enable_quack_gemm(True):
        if use_fp8:
            with enable_fp8(True):
                ow, lw = model(xw, use_fp8=True)
        else:
            ow, lw = model(xw)
    (ow.sum() + lw).backward()
    model.zero_grad(set_to_none=True)
    del ow, lw, xw
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# Profile
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    for _ in range(3):
        xp = x.detach().clone().requires_grad_(True)
        with enable_quack_gemm(True):
            if use_fp8:
                with enable_fp8(True):
                    op, lp = model(xp, use_fp8=True)
            else:
                op, lp = model(xp)
        (op.sum() + lp).backward()
        model.zero_grad(set_to_none=True)
        del op, lp, xp
    torch.cuda.synchronize()

# Wall-clock timing + CUDA-event timing
times = []
cuda_times = []
for _ in range(20):
    xt = x.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start_evt.record()
    with enable_quack_gemm(True):
        if use_fp8:
            with enable_fp8(True):
                ot, lt = model(xt, use_fp8=True)
        else:
            ot, lt = model(xt)
    (ot.sum() + lt).backward()
    end_evt.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)
    cuda_times.append(start_evt.elapsed_time(end_evt))
    model.zero_grad(set_to_none=True)
    del ot, lt, xt
gc.collect()

# Trimmed mean: drop 4 (2 min + 2 max) from 20, mean of 16
cuda_times.sort()
trimmed_cuda = cuda_times[2:-2]

# Aggregate kernels
kernel_agg = defaultdict(lambda: {{"total_us": 0.0, "count": 0}})
for evt in prof.key_averages():
    if hasattr(evt, "self_device_time_total") and evt.self_device_time_total > 0:
        kernel_agg[evt.key]["total_us"] += evt.self_device_time_total / 3  # 3 iters
        kernel_agg[evt.key]["count"] += evt.count // 3

# Sort by time
sorted_kernels = sorted(kernel_agg.items(), key=lambda kv: -kv[1]["total_us"])

total_cuda = sum(v["total_us"] for v in kernel_agg.values())

result = {{
    "mode": MODE,
    "total_cuda_us": round(total_cuda, 1),
    "wall_clock_ms": round(sum(times[5:]) / len(times[5:]), 3),
    "cuda_event_ms": round(sum(trimmed_cuda) / len(trimmed_cuda), 3),
    "kernels": [
        {{"name": k, "cuda_time_us": round(v["total_us"], 2), "count": v["count"]}}
        for k, v in sorted_kernels if v["total_us"] > 1.0
    ],
}}
print("__KERNEL_JSON__" + json.dumps(result))
'''


def _categorize_kernel(name: str) -> str:
    """Map kernel name to human-readable category.

    Works with both torch.profiler names and nsys demangled names.
    Order matters: quant checks MUST precede CUTLASS/GEMM checks because
    CuTe-compiled quant kernels have "cutlass" in their name.
    Designed to minimize "Other" — every significant kernel type gets a label.
    """
    nl = name.lower()
    # ── FP8 quantization family (check BEFORE GEMM/cutlass) ──────────
    if ("blockscaled_quant" in nl or "BlockscaledQuant" in name) and "Gemm" not in name:
        return "Blockscaled Quant"
    if "colwise" in nl and ("quant" in nl or "quantize" in nl):
        return "Blockscaled Quant"
    if "flat_quant" in nl or "FlatQuant" in name:
        return "Flat Quant"
    if "_quantize_and_pack" in nl:
        return "Row Quant"
    if "gather_isa" in nl or "ISAGather" in name:
        return "ISA Scale Gather"
    if "_dual_varlen_quantize" in nl or "_dual_quantize" in nl:
        return "Dual Quant"
    if "dequantize" in nl or "blockscaled_fp8" in nl:
        return "FP8 Dequant"
    # ── GEMM family ──────────────────────────────────────────────────
    if "GemmDefault" in name and "Sm100" in name:
        return "Wgrad GEMM"
    if "GemmGated" in name and "ZeroMat" not in name:
        return "GemmGated (fwd)"
    if "GemmDGated" in name and "ZeroMat" not in name:
        return "GemmDGated (bwd)"
    if "GemmGated" in name and "ZeroMat" in name:
        return "GemmGated ZeroMat (fwd)"
    if "GemmDGated" in name and "ZeroMat" in name:
        return "GemmDGated ZeroMat (bwd)"
    if "cutlass" in nl and "gemm" in nl:
        return "GEMM (other)"
    if "nvjet" in nl or "cublasLt" in nl or "splitKreduce" in nl:
        return "cuBLAS GEMM"
    # ── Activation / routing ─────────────────────────────────────────
    if "swiglu" in nl or "SwiGLU" in name:
        return "SwiGLU"
    if "scatter" in nl and "token" in nl:
        return "Token Scatter"
    if ("gather" in nl and "token" in nl) or "token_gather" in nl:
        return "Token Gather"
    if "topk" in nl or "TC_topk" in nl:
        return "TopK Router"
    if "softmax" in nl:
        return "Softmax"
    if "_bitmatrix" in nl or "_compute_col_partial_sum" in nl:
        return "Router Metadata"
    # ── PyTorch elementwise (break down the former "Other" blob) ─────
    if "elementwise_kernel" in nl and ("128" in name) and (", 4," in name or "<4," in name):
        return "Tensor Copy/Cast"
    if "elementwise_kernel" in nl:
        return "Elementwise Ops"
    if "vectorized_elementwise" in nl:
        if "Fill" in name:
            return "Tensor Fill"
        if "copy" in nl or "bfloat16_copy" in nl:
            return "Tensor Copy/Cast"
        if "add" in nl or "CUDAFunctor_add" in name:
            return "Elementwise Add"
        return "Elementwise Ops"
    if "unrolled_elementwise" in nl:
        return "Tensor Copy/Cast"
    if "reduce_kernel" in nl:
        return "Reduce"
    if "index_elementwise" in nl or "vectorized_gather" in nl:
        return "Index/Gather"
    return "Other"


# ═══════════════════════════════════════════════════════════════════════════════
# nsys GPU-Projection Profiling Engine
# ═══════════════════════════════════════════════════════════════════════════════
#
# Gold-standard performance measurement: runs nsys profile on a steady-state
# workload, then parses the sqlite output to compute GPU-projection time
# (merge overlapping kernel intervals) and per-kernel breakdown.
#
# Unlike torch.profiler, this is immune to CPU contention artifacts and
# gives true GPU busy time.

_NSYS_WORKLOAD_TEMPLATE = textwrap.dedent(r'''
import os, sys, json, torch, gc
import torch.nn.functional as F_torch
sys.path.insert(0, "{root}")
# Inherit CUDA_VISIBLE_DEVICES from parent (set by shell for GPU pinning).
# Only override when running standalone (no parent CVD set).
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ["USE_QUACK_GEMM"] = "1"

mode = "{mode}"

# CRITICAL: set FP8 env vars BEFORE importing sonicmoe, because
# sonicmoe.functional.utils._IS_FP8_ACTIVE is evaluated at module load time
# from SONIC_MOE_FP8_MODE.  Setting after import has no effect on the global.
if mode == "fp8":
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe import MoE, enable_fp8, enable_quack_gemm
from sonicmoe.enums import ActivationType
import sonicmoe.functional as functional
from sonicmoe.functional import count_cumsum, moe_general_routing_inputs

T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
Mtile = 128

# Reset FP8 state cleanly, then let it build up during warmup
functional.clear_all_fp8_weight_caches()
functional._ALIGNMENT_ASSUMED = False
functional._ALIGNMENT_STREAK = 0

torch.manual_seed(42)
device = torch.device("cuda:0")
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_()

use_fp8 = (mode == "fp8")
# Token rounding only needed for FP8 (128-alignment constraint).
# BF16 handles arbitrary cu_seqlens natively.
use_token_rounding = use_fp8 and (E > 8)

# Token rounding: pre-compute 128-aligned routing (frozen across iters)
if use_token_rounding:
    with torch.no_grad():
        router_logits = F_torch.linear(x, moe.router.weight)
        scores = F_torch.softmax(router_logits, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        topk_values, topk_indices = scores.topk(K, dim=-1)
        topk_values /= topk_values.sum(dim=-1, keepdim=True)
        scores.scatter_(-1, topk_indices, topk_values)
        combined = scores.detach().clone() - 1
        combined.scatter_(1, topk_indices, topk_values)
        sorted_idx = combined.argsort(dim=0, descending=True).int()
        expert_freq = count_cumsum(topk_indices.view(-1), E, do_cumsum=True)[0]
        expert_freq_rounded = (torch.ceil(expert_freq / Mtile) * Mtile).int()
        mask = torch.arange(T, device=device, dtype=torch.int32)[:, None].expand(-1, E) < expert_freq_rounded[None, :]
        token_indices_r = sorted_idx[mask]
        expert_indices_r = torch.arange(E, device=device, dtype=torch.int32)[None, :].expand(T, -1)[mask]
        order = token_indices_r.argsort().int()
        token_indices_r = token_indices_r[order]
        expert_indices_r = expert_indices_r[order]
        router_scores_r = scores[token_indices_r, expert_indices_r].contiguous()

# FP8 frontier: refresh + stash (highest performance, all caches retained).
# For memory-constrained: use setup_cpu_optimizer() instead (see docs/HANDOFF.md).
if use_fp8:
    moe.refresh_fp8_shadow_weights()
    moe.stash_bf16_to_cpu()

# Create weight views AFTER setup_cpu_optimizer (avoids keeping bf16 storage
# alive via Python refcount when stash replaces param.data).
w1_p = moe.c_fc.weight.permute(1, 2, 0)
w2_p = moe.c_proj.weight.permute(1, 2, 0)

# Token rounding guarantees 128-alignment; standard path detects it via streak.
# Set alignment assumed AFTER stash (stash needs it False during setup).
functional._ALIGNMENT_ASSUMED = True

from sonicmoe.functional import moe_TC_softmax_topk_layer

def run_iter():
    if use_token_rounding:
        o, ef = moe_general_routing_inputs(
            x, router_scores_r, token_indices_r, expert_indices_r,
            w1_p, None, w2_p, None, E, moe.stream_id, ActivationType.SWIGLU, False,
        )
    else:
        # Direct low-level call, matching official benchmark exactly.
        # Avoids MoE.forward() overhead (config resolution, ExitStack, etc.)
        o, _, _ = moe_TC_softmax_topk_layer(
            x, moe.router.weight, w1_p, None, w2_p, None,
            K, moe.stream_id, ActivationType.SWIGLU, False,
        )
    return o

# Warm up ({warmup} iters)
for _ in range({warmup}):
    out = run_iter()
    out.sum().backward()
    moe.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()

# Measured iterations
torch.cuda.cudart().cudaProfilerStart()
for _ in range({iters}):
    out = run_iter()
    out.sum().backward()
    moe.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("NSYS_DONE", flush=True)
''')


def _nsys_categorize_kernel(name: str) -> str:
    """Categorize nsys kernel names into high-level groups.

    Same logic as _categorize_kernel but also handles nsys-mangled names
    where CuTe-compiled quant kernels have "cutlass" in the demangled name.
    """
    return _categorize_kernel(name)


def _nsys_parse_sqlite(
    db_path: str, num_iters: int,
) -> dict[str, Any]:
    """Parse nsys sqlite export for GPU-projection and kernel breakdown.

    Returns dict with:
      - gpu_projection_us: total GPU busy time (merged overlapping intervals)
      - per_iter_us: gpu_projection_us / num_iters
      - kernel_breakdown: list of {name, category, total_us, count, per_iter_us}
    """
    conn = sqlite3.connect(db_path)

    # Resolve string IDs (nsys stores names as integer references)
    string_map: dict[int, str] = {}
    try:
        for row in conn.execute("SELECT id, value FROM StringIds"):
            string_map[row[0]] = row[1]
    except sqlite3.OperationalError:
        pass

    # Read all GPU kernel events
    kernels: list[tuple[int, int, int, int]] = []  # (start, end, nameId, demangledId)
    try:
        for row in conn.execute(
            "SELECT start, end, demangledName, shortName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            kernels.append((row[0], row[1], row[2], row[3]))
    except sqlite3.OperationalError:
        conn.close()
        return {"error": "No kernel data in sqlite"}

    conn.close()

    if not kernels:
        return {"error": "No kernels found"}

    # Compute GPU-projection (merge overlapping intervals)
    kernels.sort(key=lambda x: x[0])
    merged_ns = 0
    cur_start, cur_end = kernels[0][0], kernels[0][1]
    for start, end, _, _ in kernels[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged_ns += cur_end - cur_start
            cur_start, cur_end = start, end
    merged_ns += cur_end - cur_start
    gpu_projection_us = merged_ns / 1000.0

    # Per-kernel breakdown
    kernel_stats: dict[str, dict[str, Any]] = {}
    for start, end, demangled_id, short_id in kernels:
        # Prefer demangled name, fall back to short name
        name = string_map.get(demangled_id, string_map.get(short_id, f"unknown_{demangled_id}"))
        dur_us = (end - start) / 1000.0
        if name not in kernel_stats:
            kernel_stats[name] = {"total_us": 0.0, "count": 0}
        kernel_stats[name]["total_us"] += dur_us
        kernel_stats[name]["count"] += 1

    breakdown = []
    for name, stats in sorted(kernel_stats.items(), key=lambda x: -x[1]["total_us"]):
        cat = _nsys_categorize_kernel(name)
        breakdown.append({
            "name": name[:120],  # truncate very long kernel names
            "category": cat,
            "total_us": round(stats["total_us"], 1),
            "count": stats["count"],
            "per_iter_us": round(stats["total_us"] / num_iters, 1),
            "per_call_us": round(stats["total_us"] / stats["count"], 1),
        })

    # Category summary
    cat_totals: dict[str, float] = collections.defaultdict(float)
    for k in breakdown:
        cat_totals[k["category"]] += k["per_iter_us"]

    return {
        "gpu_projection_us": round(gpu_projection_us, 1),
        "per_iter_us": round(gpu_projection_us / num_iters, 1),
        "num_iters": num_iters,
        "num_kernels": len(kernels),
        "kernel_breakdown": breakdown[:50],  # top 50 kernels
        "category_summary": dict(sorted(cat_totals.items(), key=lambda x: -x[1])),
    }


# ── Memory measurement subprocess (paired with nsys) ────────────────────────

_MEM_MEASURE_SCRIPT = textwrap.dedent(r'''
import gc, json, os, sys, torch
import torch.nn.functional as F_torch

sys.path.insert(0, "{root}")
# Inherit CUDA_VISIBLE_DEVICES from parent (set by shell for GPU pinning).
# Only override when running standalone (no parent CVD set).
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ["USE_QUACK_GEMM"] = "1"

mode = "{mode}"
if mode == "fp8":
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
from sonicmoe.functional import count_cumsum, moe_general_routing_inputs
import sonicmoe.functional as functional

functional.clear_all_fp8_weight_caches()

T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
Mtile = 128
torch.manual_seed(42)
device = torch.device("cuda:0")
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_()

use_fp8 = (mode == "fp8")
# Token rounding only needed for FP8 (128-alignment constraint).
# BF16 handles arbitrary cu_seqlens natively.
use_token_rounding = use_fp8 and (E > 8)

# Token rounding for E>8
if use_token_rounding:
    with torch.no_grad():
        rl = F_torch.linear(x, moe.router.weight)
        sc = F_torch.softmax(rl, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        tv, ti = sc.topk(K, dim=-1)
        tv /= tv.sum(dim=-1, keepdim=True)
        sc.scatter_(-1, ti, tv)
        cb = sc.detach().clone() - 1
        cb.scatter_(1, ti, tv)
        si = cb.argsort(dim=0, descending=True).int()
        ef = count_cumsum(ti.view(-1), E, do_cumsum=True)[0]
        efr = (torch.ceil(ef / Mtile) * Mtile).int()
        mk = torch.arange(T, device=device, dtype=torch.int32)[:, None].expand(-1, E) < efr[None, :]
        tok_idx = si[mk]
        exp_idx = torch.arange(E, device=device, dtype=torch.int32)[None, :].expand(T, -1)[mk]
        od = tok_idx.argsort().int()
        tok_idx = tok_idx[od]; exp_idx = exp_idx[od]
        rsc = sc[tok_idx, exp_idx].contiguous()
if use_fp8:
    moe.refresh_fp8_shadow_weights()
    moe.stash_bf16_to_cpu()

# Create weight views AFTER stash
if use_token_rounding:
    w1_p = moe.c_fc.weight.permute(1, 2, 0)
    w2_p = moe.c_proj.weight.permute(1, 2, 0)

def run_iter():
    xw = x.detach().clone().requires_grad_(True)
    if use_token_rounding:
        o, ef = moe_general_routing_inputs(
            xw, rsc, tok_idx, exp_idx, w1_p, None, w2_p, None,
            E, moe.stream_id, ActivationType.SWIGLU, False)
    else:
        with enable_quack_gemm(True):
            if use_fp8:
                with enable_fp8(True): o, aux = moe(xw, use_fp8=True)
            else:
                with enable_fp8(False): o, aux = moe(xw)
        o = o
    return xw, o

# Warmup
for _ in range({warmup}):
    xw, o = run_iter()
    o.sum().backward()
    moe.zero_grad(set_to_none=True)
    del xw, o
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# Measure
MiB = 1048576
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize()

base = torch.cuda.memory_allocated(device) / MiB
xw, o = run_iter()
torch.cuda.synchronize()
peak_fwd = torch.cuda.max_memory_allocated(device) / MiB

torch.cuda.reset_peak_memory_stats(device)
o.sum().backward()
torch.cuda.synchronize()
peak_bwd = torch.cuda.max_memory_allocated(device) / MiB

result = {{"mode": mode, "base_mib": round(base, 1),
           "peak_fwd_mib": round(peak_fwd, 1),
           "peak_bwd_mib": round(peak_bwd, 1)}}
print("__MEM_JSON__" + json.dumps(result))
''')


def _run_memory_measure(mode: str, shape: dict, gpu: int, warmup: int = 5) -> dict:
    """Subprocess-isolated memory measurement for one mode+shape."""
    python_bin = _resolve_python_bin()
    script = _MEM_MEASURE_SCRIPT.format(
        root=str(ROOT), gpu=str(gpu), mode=mode, warmup=warmup, **shape,
    )
    env = _subprocess_env_for_gpu(gpu)
    proc = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True, text=True, timeout=600, env=env, cwd=str(ROOT),
    )
    for line in proc.stdout.split("\n"):
        if line.startswith("__MEM_JSON__"):
            return json.loads(line[len("__MEM_JSON__"):])
    return {"error": proc.stderr[-300:] if proc.stderr else "no output"}


def run_nsys_profile(
    gpu: int = 0,
    warmup: int = DEFAULT_NSYS_WARMUP,
    iters: int = DEFAULT_NSYS_ITERS,
    shapes: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Run nsys GPU-projection profiling for BF16 and FP8.

    This is the gold-standard profiling method:
    - Steady-state measurement (no FP8 cache reset between iterations)
    - GPU-projection time (merged kernel intervals, immune to CPU contention)
    - Per-kernel breakdown with category classification

    Args:
        gpu: GPU device index
        warmup: warmup iterations before measurement
        iters: measured iterations
        shapes: list of shape dicts (default: [SHAPE])

    Returns:
        dict with per-shape, per-mode results
    """
    python_bin = _resolve_python_bin()
    nsys_bin = "nsys"

    if shapes is None:
        shapes = [SHAPE]

    # Persistent output directory for nsys-rep files (user-inspectable)
    nsys_output_dir = Path(
        os.environ.get("SONIC_MOE_NSYS_OUTPUT_DIR", str(ROOT / "reports" / "nsys"))
    )
    nsys_output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"shapes": {}}

    for shape in shapes:
        shape_key = f"T{shape['T']}_I{shape['I']}_E{shape['E']}K{shape['K']}"
        shape_results: dict[str, Any] = {"shape": shape}

        # ── nsys profiling: bf16 then fp8 (serial, no GPU contention) ───
        for mode in ("bf16", "fp8"):
            label = f"{mode}/{shape_key}"
            print(f"  nsys profiling [{label}] ({warmup}w+{iters}m) ...", flush=True)

            script = _NSYS_WORKLOAD_TEMPLATE.format(
                root=str(ROOT), gpu=str(gpu), mode=mode,
                warmup=warmup, iters=iters, **shape,
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, prefix=f"nsys_{mode}_{shape_key}_"
            ) as f:
                f.write(script)
                script_path = f.name

            # Output to persistent dir with descriptive name
            ts = time.strftime("%H%M%S")
            rep_name = f"{mode}_{shape_key}_{ts}"
            rep_path = str(nsys_output_dir / rep_name)

            try:
                sub_env = _subprocess_env_for_gpu(gpu)
                cmd = [
                    nsys_bin, "profile",
                    "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop",
                    f"--output={rep_path}",
                    "--export=sqlite",
                    "--force-overwrite=true",
                    python_bin, script_path,
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600,
                    env=sub_env,
                )
                if proc.returncode != 0:
                    print(f"  [WARN] nsys failed for {label}: {proc.stderr[-300:]}", flush=True)
                    shape_results[mode] = {"error": proc.stderr[-500:]}
                    continue

                db_file = f"{rep_path}.sqlite"
                if not os.path.exists(db_file):
                    print(f"  [WARN] sqlite not found for {label}", flush=True)
                    shape_results[mode] = {"error": "sqlite output missing"}
                    continue

                parsed = _nsys_parse_sqlite(db_file, iters)
                parsed["mode"] = mode
                parsed["shape"] = shape
                parsed["nsys_rep"] = f"{rep_path}.nsys-rep"
                shape_results[mode] = parsed
                print(
                    f"    {label}: {parsed.get('per_iter_us', '?')} µs/iter "
                    f"({parsed.get('num_kernels', '?')} kernels)"
                    f"  → {rep_path}.nsys-rep",
                    flush=True,
                )
            except subprocess.TimeoutExpired:
                print(f"  [WARN] nsys timed out for {label}", flush=True)
                shape_results[mode] = {"error": "timeout"}
            finally:
                # Keep nsys-rep + sqlite for user inspection; only delete the temp script
                try:
                    os.unlink(script_path)
                except OSError:
                    pass

        # Compute speedup if both modes succeeded
        bf16_us = shape_results.get("bf16", {}).get("per_iter_us")
        fp8_us = shape_results.get("fp8", {}).get("per_iter_us")
        if bf16_us and fp8_us and fp8_us > 0:
            shape_results["speedup"] = round(bf16_us / fp8_us, 4)
            print(f"    {shape_key} speedup: {shape_results['speedup']}×", flush=True)

        # ── Paired memory measurement (subprocess-isolated) ────────────
        # FP8 = stash mode (the frontier default)
        for mem_mode in ("bf16", "fp8"):
            print(f"  memory [{mem_mode}/{shape_key}] ...", flush=True)
            mem = _run_memory_measure(mem_mode, shape, gpu, warmup=warmup)
            shape_results[f"memory_{mem_mode}"] = mem
            if "error" not in mem:
                print(f"    {mem_mode}: fwd={mem['peak_fwd_mib']:.0f} bwd={mem['peak_bwd_mib']:.0f} MiB",
                      flush=True)

        # ── Category breakdown delta (budget reconciliation) ───────────
        bf16_cats = shape_results.get("bf16", {}).get("category_summary", {})
        fp8_cats = shape_results.get("fp8", {}).get("category_summary", {})
        if bf16_cats and fp8_cats:
            all_cats = sorted(set(list(bf16_cats) + list(fp8_cats)))
            breakdown = {}
            for cat in all_cats:
                b = bf16_cats.get(cat, 0.0)
                f = fp8_cats.get(cat, 0.0)
                breakdown[cat] = {"bf16_us": round(b, 1), "fp8_us": round(f, 1),
                                  "delta_us": round(f - b, 1)}
            shape_results["budget_breakdown"] = breakdown
            savings = sum(d["delta_us"] for d in breakdown.values() if d["delta_us"] < 0)
            overhead = sum(d["delta_us"] for d in breakdown.values() if d["delta_us"] > 0)
            shape_results["budget_savings_us"] = round(savings, 1)
            shape_results["budget_overhead_us"] = round(overhead, 1)
            shape_results["budget_net_us"] = round(savings + overhead, 1)

        results["shapes"][shape_key] = shape_results

    return results


def run_kernel_profile(gpu: int = 0) -> dict[str, Any]:
    """Run kernel profiling in subprocess for both BF16 and FP8.

    Returns dict with "bf16" and "fp8" kernel data.
    """
    python_bin = _resolve_python_bin()

    result = {}
    for mode in ("bf16", "fp8"):
        script = _KERNEL_PROFILE_SCRIPT.format(**SHAPE)
        env = _build_subprocess_env(mode, gpu)
        env["_PROFILER_MODE"] = mode

        print(f"  Kernel profiling [{mode}] in subprocess ...", flush=True)
        try:
            proc = subprocess.run(
                [python_bin, "-c", script],
                capture_output=True, text=True, timeout=300, env=env,
                cwd=str(ROOT),
            )
            # Extract JSON from output
            for line in proc.stdout.split("\n"):
                if line.startswith("__KERNEL_JSON__"):
                    data = json.loads(line[len("__KERNEL_JSON__"):])
                    # Add categories
                    for k in data.get("kernels", []):
                        k["category"] = _categorize_kernel(k["name"])
                    result[mode] = data
                    break
            else:
                print(f"  WARNING: No kernel JSON found in {mode} output", flush=True)
                if proc.stderr:
                    print(f"  stderr (last 500 chars): {proc.stderr[-500:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  WARNING: Kernel profiling [{mode}] timed out", flush=True)
        except Exception as e:
            print(f"  WARNING: Kernel profiling [{mode}] failed: {e}", flush=True)

    return result


def run_rigorous_benchmark(gpu: int, seeds: list[int], repeats: int) -> dict[str, Any] | None:
    """Run the repeated benchmark suite used by README/session figures."""
    if not _is_default_shape():
        print("  [skip] rigorous benchmark only supports the default reference shape", flush=True)
        return None

    bench_path = ROOT / "tools" / "rigorous_benchmark_s42.py"
    if not bench_path.exists():
        print("  [skip] rigorous benchmark script not found", flush=True)
        return None

    mod = _load_python_module(f"rigorous_benchmark_s42_{time.time_ns()}", bench_path)
    all_results: list[dict[str, Any]] = []
    with _persistent_tempdir() as tmpdir:
        for repeat in range(repeats):
            print(f"  Rigorous benchmark repeat {repeat + 1}/{repeats} ...", flush=True)
            for seed in seeds:
                for bench_mode in ("bf16", "fp8", "fp8_stash"):
                    result = mod.run(bench_mode, seed, tmpdir, str(gpu))
                    result["repeat"] = repeat
                    all_results.append(result)

    return {
        "shape": dict(SHAPE),
        "seeds": list(seeds),
        "n_repeats": repeats,
        "warmup": getattr(mod, "WARMUP", None),
        "timing_iters": getattr(mod, "TIMING_ITERS", None),
        "results": all_results,
        "device": all_results[0].get("device", "unknown") if all_results else "unknown",
    }


def _summarize_benchmark_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None

    summary: dict[str, Any] = {
        "shape": report.get("shape", {}),
        "seeds": report.get("seeds", []),
        "n_repeats": report.get("n_repeats", 0),
        "modes": {},
        "comparisons": {},
        "note": (
            "Memory/precision come from subprocess-isolated repeated runs. "
            "Use kernel_breakdown.json for GPU-projection timing."
        ),
    }

    for mode in ("bf16", "fp8", "fp8_stash"):
        entries = [r for r in report.get("results", []) if r.get("mode") == mode]
        if not entries:
            continue
        mode_summary: dict[str, Any] = {
            "n": len(entries),
            "memory_mib": {
                "base": _stat_summary([r["mem"]["base"] for r in entries], digits=2),
                "fwd_peak": _stat_summary([r["mem"]["fwd_peak"] for r in entries], digits=2),
                "bwd_peak": _stat_summary([r["mem"]["bwd_peak"] for r in entries], digits=2),
            },
            "timing_ms": {
                "trimmed": _stat_summary([r["timing"]["trimmed_ms"] for r in entries], digits=3),
            },
        }
        prec_entries = [r.get("precision", {}) for r in entries if r.get("precision")]
        if prec_entries:
            mode_summary["precision"] = {
                "output_rrmse_pct": _stat_summary([p["out_rrmse"] for p in prec_entries], digits=3),
                "output_corr": _stat_summary([p["out_corr"] for p in prec_entries], digits=4),
                "dx_rrmse_pct": _stat_summary([p["dx_rrmse"] for p in prec_entries], digits=3),
                "dx_corr": _stat_summary([p["dx_corr"] for p in prec_entries], digits=4),
            }
        summary["modes"][mode] = mode_summary

    def _mode_mean(mode: str, section: str, field: str) -> float | None:
        mode_data = summary["modes"].get(mode, {})
        section_data = mode_data.get(section, {})
        field_data = section_data.get(field, {})
        return field_data.get("mean")

    bf16_fwd = _mode_mean("bf16", "memory_mib", "fwd_peak")
    bf16_bwd = _mode_mean("bf16", "memory_mib", "bwd_peak")
    bf16_t = _mode_mean("bf16", "timing_ms", "trimmed")
    for mode in ("fp8", "fp8_stash"):
        fwd = _mode_mean(mode, "memory_mib", "fwd_peak")
        bwd = _mode_mean(mode, "memory_mib", "bwd_peak")
        total_ms = _mode_mean(mode, "timing_ms", "trimmed")
        if bf16_fwd is None or bf16_bwd is None or fwd is None or bwd is None:
            continue
        comparison = {
            "fwd_peak_delta_mib": round(fwd - bf16_fwd, 2),
            "bwd_peak_delta_mib": round(bwd - bf16_bwd, 2),
            "fwd_peak_delta_pct": round((fwd - bf16_fwd) / bf16_fwd * 100, 3),
            "bwd_peak_delta_pct": round((bwd - bf16_bwd) / bf16_bwd * 100, 3),
        }
        if bf16_t and total_ms:
            comparison["timing_speedup"] = round(bf16_t / total_ms, 4)
            comparison["timing_delta_ms"] = round(total_ms - bf16_t, 3)
        summary["comparisons"][f"{mode}_vs_bf16"] = comparison

    return summary


def run_rigorous_profiler(gpu: int, repeats: int = 1) -> list[dict[str, Any]] | None:
    """Run the subprocess-isolated profiler used for kernel/memory JSON assets."""
    if not _is_default_shape():
        print("  [skip] rigorous profiler only supports the default reference shape", flush=True)
        return None

    profiler_path = ROOT / "tools" / "rigorous_profiler.py"
    if not profiler_path.exists():
        print("  [skip] rigorous profiler script not found", flush=True)
        return None

    mod = _load_python_module(f"rigorous_profiler_{time.time_ns()}", profiler_path)
    runs: list[dict[str, Any]] = []
    for trial in range(repeats):
        print(f"  Rigorous profiler trial {trial + 1}/{repeats} ...", flush=True)
        bf16 = mod.run_mode("bf16", str(gpu))
        fp8 = mod.run_mode("fp8", str(gpu))
        if bf16 and fp8:
            runs.append({"bf16": bf16, "fp8": fp8})
    return runs or None


def _aggregate_profiler_mode_runs(mode: str, mode_runs: list[dict[str, Any]]) -> dict[str, Any]:
    kernel_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    total_cuda_vals: list[float] = []
    fwd_ms_vals: list[float] = []
    bwd_ms_vals: list[float] = []
    total_ms_vals: list[float] = []
    for run in mode_runs:
        total_cuda_vals.append(run["kernel_profiling"]["total_cuda_us"])
        fwd_ms_vals.append(run["wall_clock_ms"]["median_fwd_ms"])
        bwd_ms_vals.append(run["wall_clock_ms"]["median_bwd_ms"])
        total_ms_vals.append(run["wall_clock_ms"]["median_total_ms"])
        for kernel in run["kernel_profiling"]["kernels"]:
            kernel_groups[kernel["name"]].append(kernel)

    kernels = []
    for name, entries in kernel_groups.items():
        kernels.append({
            "name": name,
            "avg_cuda_us": round(_safe_mean([e["avg_cuda_us"] for e in entries]), 2),
            "median_cuda_us": round(_safe_mean([e["median_cuda_us"] for e in entries]), 2),
            "avg_count": round(_safe_mean([e["avg_count"] for e in entries]), 2),
            "std_us": round(_safe_std([e["avg_cuda_us"] for e in entries]), 2),
        })
    kernels.sort(key=lambda item: -item["avg_cuda_us"])

    ref = mode_runs[0]
    return {
        "label": mode.upper(),
        "profile_trials": len(mode_runs),
        "profile_iters": ref["kernel_profiling"].get("profile_iters", 1),
        "median_fwd_ms": round(_safe_mean(fwd_ms_vals), 4),
        "median_bwd_ms": round(_safe_mean(bwd_ms_vals), 4),
        "median_total_ms": round(_safe_mean(total_ms_vals), 4),
        "total_cuda_us": round(_safe_mean(total_cuda_vals), 2),
        "total_cuda_us_std": round(_safe_std(total_cuda_vals), 2),
        "kernels": kernels,
        "runs": mode_runs,
    }


def _build_manifest_kernel_data(profiler_runs: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not profiler_runs:
        return None
    aggregated = {}
    for mode in ("bf16", "fp8"):
        mode_runs = [run[mode] for run in profiler_runs if run.get(mode)]
        if not mode_runs:
            continue
        agg = _aggregate_profiler_mode_runs(mode, mode_runs)
        aggregated[mode] = {
            "label": agg["label"],
            "profile_trials": agg["profile_trials"],
            "profile_iters": agg["profile_iters"],
            "median_fwd_ms": agg["median_fwd_ms"],
            "median_bwd_ms": agg["median_bwd_ms"],
            "median_total_ms": agg["median_total_ms"],
            "wall_clock_ms": agg["median_total_ms"],
            "total_cuda_us": agg["total_cuda_us"],
            "total_cuda_us_std": agg["total_cuda_us_std"],
            "kernels": agg["kernels"],
        }
    if "bf16" in aggregated and "fp8" in aggregated and aggregated["fp8"]["total_cuda_us"] > 0:
        speedup = round(aggregated["bf16"]["total_cuda_us"] / aggregated["fp8"]["total_cuda_us"], 4)
        aggregated["bf16"]["gpu_projection_speedup"] = 1.0
        aggregated["fp8"]["gpu_projection_speedup"] = speedup
    return aggregated


def _build_mem_breakdown_json(
    profiler_runs: list[dict[str, Any]] | None,
    benchmark_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not profiler_runs:
        return None

    def _avg_snapshot(mode_runs: list[dict[str, Any]], snap_name: str) -> dict[str, float]:
        keys = mode_runs[0]["memory_lifecycle"]["snapshots"][snap_name].keys()
        return {
            key: round(_safe_mean([
                float(run["memory_lifecycle"]["snapshots"][snap_name][key]) for run in mode_runs
            ]), 4)
            for key in keys
        }

    result: dict[str, Any] = {}
    for mode in ("bf16", "fp8"):
        mode_runs = [run[mode] for run in profiler_runs if run.get(mode)]
        if not mode_runs:
            continue
        detailed = {
            snap: _avg_snapshot(mode_runs, snap)
            for snap in ("base", "after_model", "after_input", "pre_fwd", "post_fwd", "pre_bwd", "post_bwd", "cleanup")
        }
        deltas_src = [run["memory_lifecycle"]["deltas_MiB"] for run in mode_runs]
        delta_keys = deltas_src[0].keys()
        deltas = {
            key: round(_safe_mean([float(delta[key]) for delta in deltas_src]), 4)
            for key in delta_keys
        }
        cache_keys = set()
        for run in mode_runs:
            cache_keys.update(run["memory_lifecycle"]["fp8_cache_after_bwd"].keys())
        fp8_cache = {
            key: round(_safe_mean([
                float(run["memory_lifecycle"]["fp8_cache_after_bwd"].get(key, 0.0)) for run in mode_runs
            ]), 4)
            for key in sorted(cache_keys)
        }
        result[mode] = {
            "mode": mode,
            "checkpoints": {
                "base": round(detailed["base"]["allocated_MiB"], 2),
                "after_model": round(detailed["after_model"]["allocated_MiB"], 2),
                "after_input": round(detailed["after_input"]["allocated_MiB"], 2),
                "pre_fwd": round(detailed["pre_fwd"]["allocated_MiB"], 2),
                "post_fwd": round(detailed["post_fwd"]["allocated_MiB"], 2),
                "peak_fwd": round(detailed["post_fwd"]["peak_allocated_MiB"], 2),
                "pre_bwd": round(detailed["pre_bwd"]["allocated_MiB"], 2),
                "post_bwd": round(detailed["post_bwd"]["allocated_MiB"], 2),
                "peak_bwd": round(detailed["post_bwd"]["peak_allocated_MiB"], 2),
                "post_cleanup": round(detailed["cleanup"]["allocated_MiB"], 2),
            },
            "detailed_snapshots": detailed,
            "deltas": deltas,
            "param_sizes_mib": mode_runs[0]["memory_lifecycle"]["param_sizes_MiB"],
            "grad_sizes_mib": mode_runs[0]["memory_lifecycle"]["grad_sizes_MiB"],
            "fp8_caches_after_bwd": fp8_cache,
            "theoretical_sizes_mib": mode_runs[0]["memory_lifecycle"]["theoretical_sizes_MiB"],
        }

    if benchmark_summary and "modes" in benchmark_summary:
        fp8_stash = benchmark_summary["modes"].get("fp8_stash")
        if fp8_stash:
            result["fp8_stash"] = {
                "mode": "fp8_stash",
                "summary": fp8_stash,
                "comparison_vs_bf16": benchmark_summary.get("comparisons", {}).get("fp8_stash_vs_bf16", {}),
            }

    result["_metadata"] = {
        "source": "tools/rigorous_profiler.py",
        "profiler": "torch.cuda.memory_stats()",
        "profile_trials": len(profiler_runs),
        "note": "subprocess-isolated and aggregated across profiler trials",
    }
    return result


def _build_compat_kernel_breakdown(kernel_data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not kernel_data:
        return None

    compat: dict[str, Any] = {}
    for mode, label in (("bf16", "BF16"), ("fp8", "FP8")):
        entry = kernel_data.get(mode)
        if not entry:
            continue
        n_iters = int(entry.get("profile_iters", 1))
        kernels = []
        for kernel in entry.get("kernels", []):
            total_us = kernel["avg_cuda_us"] * n_iters
            count = int(round(kernel["avg_count"] * n_iters))
            avg_us = total_us / count if count else kernel["avg_cuda_us"]
            pct = (kernel["avg_cuda_us"] / entry["total_cuda_us"] * 100) if entry["total_cuda_us"] else 0.0
            kernels.append({
                "name": kernel["name"],
                "total_us": round(total_us),
                "count": count,
                "avg_us": round(avg_us),
                "pct": round(pct, 1),
                "per_iter_us": round(kernel["avg_cuda_us"]),
            })
        compat[label] = {
            "total_us": round(entry["total_cuda_us"] * n_iters),
            "per_iter_us": round(entry["total_cuda_us"]),
            "n_kernels": sum(k["count"] for k in kernels),
            "n_iters": n_iters,
            "kernels": kernels[:12],
        }
    return compat


def _write_json_artifact(path: Path, payload: dict[str, Any] | None) -> None:
    if payload is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Precision Auditor
# ═══════════════════════════════════════════════════════════════════════════════

def run_precision_audit(model, x) -> dict[str, Any]:
    """Compare BF16 vs FP8 outputs: RRMSE and cosine similarity."""
    import torch
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

    def _run(use_fp8: bool):
        x_run = x.detach().clone().requires_grad_(True)
        with enable_quack_gemm(True):
            if use_fp8:
                with enable_fp8(True):
                    out, loss_val = model(x_run, use_fp8=True)
            else:
                out, loss_val = model(x_run)
        loss = out.sum() + loss_val
        loss.backward()
        result = {
            "output": out.detach().float(),
            "dx": x_run.grad.detach().float() if x_run.grad is not None else None,
        }
        # Collect weight gradients
        for pname, p in model.named_parameters():
            if p.grad is not None:
                result[f"grad_{pname}"] = p.grad.detach().float()
        model.zero_grad(set_to_none=True)
        del out, loss, loss_val
        gc.collect()
        return result

    print("  Running BF16 baseline ...", flush=True)
    bf16_out = _run(False)
    print("  Running FP8 frontier ...", flush=True)
    fp8_out = _run(True)

    def _rrmse(a, b):
        if a is None or b is None:
            return None
        diff = (a - b).float()
        return float((diff.norm() / b.float().norm() * 100).item())

    def _cosine(a, b):
        if a is None or b is None:
            return None
        a_flat, b_flat = a.flatten().double(), b.flatten().double()
        denom = (a_flat.norm() * b_flat.norm()).clamp(min=1e-12)
        val = float((a_flat * b_flat).sum().item() / denom.item())
        return max(-1.0, min(1.0, val))

    audit = {"rrmse_pct": {}, "cosine_sim": {}}

    for key in ["output", "dx"]:
        audit["rrmse_pct"][key] = round(_rrmse(fp8_out.get(key), bf16_out.get(key)), 4) \
            if bf16_out.get(key) is not None else None
        audit["cosine_sim"][key] = round(_cosine(fp8_out.get(key), bf16_out.get(key)), 6) \
            if bf16_out.get(key) is not None else None

    # Weight gradients
    for pname in ["c_fc.weight", "c_proj.weight"]:
        gkey = f"grad_{pname}"
        short = "dw1" if "c_fc" in pname else "dw2"
        audit["rrmse_pct"][short] = round(_rrmse(fp8_out.get(gkey), bf16_out.get(gkey)), 4) \
            if bf16_out.get(gkey) is not None else None

    del bf16_out, fp8_out
    gc.collect()
    return audit


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest Writer
# ═══════════════════════════════════════════════════════════════════════════════

def _serialize_manifest(
    bf16_manifest: ModeManifest | None,
    fp8_manifest: ModeManifest | None,
    kernel_data: dict | None,
    precision_audit: dict | None,
    benchmark_summary: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Assemble all data into the final manifest dict."""
    manifest = {
        "version": MANIFEST_VERSION,
        "metadata": metadata or {},
        "modes": {},
    }

    for m in [bf16_manifest, fp8_manifest]:
        if m is None:
            continue
        mode_dict = {
            "tensors": [asdict(t) for t in m.tensors],
            "phase_memory": [asdict(pm) for pm in m.phase_memory],
            "memory_trajectory": m.memory_trajectory,
            "precision_matrix": m.precision_matrix,
            "theoretical_memory": m.theoretical_memory,
        }

        # Merge kernel data if available
        if kernel_data and m.mode in kernel_data:
            kd = kernel_data[m.mode]
            mode_dict["kernels"] = kd.get("kernels", [])
            mode_dict["total_cuda_us"] = kd.get("total_cuda_us", 0)
            mode_dict["wall_clock_ms"] = kd.get("wall_clock_ms", 0)
            if kd.get("total_cuda_us"):
                mode_dict["wall_to_cuda_ratio"] = round(
                    kd.get("wall_clock_ms", 0) * 1000 / kd["total_cuda_us"], 4
                )

        manifest["modes"][m.mode] = mode_dict

    # GPU projection speedup
    if kernel_data and "bf16" in kernel_data and "fp8" in kernel_data:
        bf16_cuda = kernel_data["bf16"].get("total_cuda_us", 0)
        fp8_cuda = kernel_data["fp8"].get("total_cuda_us", 0)
        bf16_wall = kernel_data["bf16"].get("wall_clock_ms", 0)
        fp8_wall = kernel_data["fp8"].get("wall_clock_ms", 0)
        if fp8_cuda > 0:
            manifest["gpu_projection_speedup"] = round(bf16_cuda / fp8_cuda, 4)
        if fp8_wall > 0:
            manifest["wall_clock_speedup"] = round(bf16_wall / fp8_wall, 4)
        if bf16_cuda > 0 and fp8_cuda > 0 and bf16_wall > 0 and fp8_wall > 0:
            manifest["kernel_efficiency"] = {
                "bf16_wall_to_cuda_ratio": round(bf16_wall * 1000 / bf16_cuda, 4),
                "fp8_wall_to_cuda_ratio": round(fp8_wall * 1000 / fp8_cuda, 4),
                "ratio_delta": round(
                    fp8_wall * 1000 / fp8_cuda - bf16_wall * 1000 / bf16_cuda, 4
                ),
            }

    if precision_audit:
        manifest["precision_audit"] = precision_audit
    if benchmark_summary:
        manifest["benchmark_summary"] = benchmark_summary

    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# Quant Kernel Benchmark Engine (CUDA-event, isolated)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Isolated CUDA-event benchmark of every quant kernel variant at arbitrary shapes.
# Reports median, p5, p95, min, max, stddev. Calculates theoretical HBM BW floor.
# Used for NCU-driven optimization feedback loop.

_QUANT_BENCH_SCRIPT = textwrap.dedent(r'''
import gc, json, os, sys, statistics, torch

device = torch.device("cuda:0")
torch.cuda.set_device(device)
sys.path.insert(0, "{root}")

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_activation_blockscaled_fast,
    colwise_quantize_and_pack,
)
try:
    from sonicmoe.quack_utils.cute_blockscaled_quant import colwise_quantize_cute
    HAS_CUTE = True
except ImportError:
    HAS_CUTE = False

SHAPES = {shapes_json}
WARMUP = {warmup}
TRIALS = {trials}
PEAK_HBM_GBps = {peak_hbm_gbps}

def bench(fn, warmup, trials):
    """CUDA-event benchmark with trimmed statistics."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)  # µs
    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    return {{
        "median_us": round(statistics.median(trimmed), 2),
        "mean_us": round(statistics.mean(trimmed), 2),
        "min_us": round(min(times), 2),
        "max_us": round(max(times), 2),
        "p5_us": round(times[max(0, len(times)//20)], 2),
        "p95_us": round(times[min(len(times)-1, len(times)*19//20)], 2),
        "stddev_us": round(statistics.stdev(times), 2) if len(times) > 1 else 0,
        "n_trials": trials,
    }}

def bw_floor_us(total_bytes):
    """Theoretical minimum time at peak HBM bandwidth."""
    return total_bytes / (PEAK_HBM_GBps * 1e3)

results = {{"shapes": {{}}}}

for shape in SHAPES:
    TK = shape["TK"]
    H = shape["H"]
    I2 = shape.get("I2", H)
    I1 = shape.get("I1", I2 // 2)
    tag = f"TK={{TK}}_H={{H}}_I2={{I2}}"
    print(f"  benchmarking {{tag}} ...", flush=True)

    shape_res = {{"TK": TK, "H": H, "I2": I2, "I1": I1, "kernels": {{}}}}

    x = torch.randn(TK, H, dtype=torch.bfloat16, device=device)
    dz = torch.randn(TK, I2, dtype=torch.bfloat16, device=device)
    y1s = torch.randn(TK, I1, dtype=torch.bfloat16, device=device)
    gather_idx = torch.randint(0, TK, (TK,), dtype=torch.int32, device=device)

    # Warmup all paths
    for _ in range(3):
        _ = quantize_activation_blockscaled_fast(x)
        _ = colwise_quantize_and_pack(x, H, TK)
    torch.cuda.synchronize()

    # --- row quant family ---
    x_bytes = TK * H * 2
    dz_bytes = TK * I2 * 2
    y1s_bytes = TK * I1 * 2

    r = bench(lambda: quantize_activation_blockscaled_fast(x), WARMUP, TRIALS)
    r["io_bytes"] = x_bytes + TK * H  # read bf16, write fp8 + scales
    r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
    r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
    shape_res["kernels"]["row_quant_x"] = r

    r = bench(lambda: quantize_activation_blockscaled_fast(dz), WARMUP, TRIALS)
    r["io_bytes"] = dz_bytes + TK * I2
    r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
    r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
    shape_res["kernels"]["row_quant_dz"] = r

    # --- colwise Triton family ---
    r = bench(lambda: colwise_quantize_and_pack(x, H, TK), WARMUP, TRIALS)
    r["io_bytes"] = x_bytes + TK * H
    r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
    r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
    shape_res["kernels"]["colwise_triton_x"] = r

    r = bench(lambda: colwise_quantize_and_pack(x, H, TK, gather_idx=gather_idx), WARMUP, TRIALS)
    r["io_bytes"] = x_bytes + TK * H  # + gather index read
    r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
    r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
    shape_res["kernels"]["colwise_triton_x_gather"] = r

    r = bench(lambda: colwise_quantize_and_pack(dz, I2, TK), WARMUP, TRIALS)
    r["io_bytes"] = dz_bytes + TK * I2
    r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
    r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
    shape_res["kernels"]["colwise_triton_dz"] = r

    r = bench(lambda: colwise_quantize_and_pack(y1s, I1, TK), WARMUP, TRIALS)
    r["io_bytes"] = y1s_bytes + TK * I1
    r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
    r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
    shape_res["kernels"]["colwise_triton_y1s"] = r

    # --- colwise CuTe family ---
    if HAS_CUTE:
        for _ in range(3):
            _ = colwise_quantize_cute(x, H, TK, isa_pack=True)
        torch.cuda.synchronize()

        r = bench(lambda: colwise_quantize_cute(x, H, TK, isa_pack=True), WARMUP, TRIALS)
        r["io_bytes"] = x_bytes + TK * H
        r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
        r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
        shape_res["kernels"]["colwise_cute_x_isa"] = r

        r = bench(lambda: colwise_quantize_cute(dz, I2, TK, isa_pack=True), WARMUP, TRIALS)
        r["io_bytes"] = dz_bytes + TK * I2
        r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
        r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
        shape_res["kernels"]["colwise_cute_dz_isa"] = r

        r = bench(lambda: colwise_quantize_cute(y1s, I1, TK, isa_pack=True), WARMUP, TRIALS)
        r["io_bytes"] = y1s_bytes + TK * I1
        r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
        r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
        shape_res["kernels"]["colwise_cute_y1s_isa"] = r

        r = bench(lambda: colwise_quantize_cute(x, H, TK, gather_idx=gather_idx, isa_pack=True), WARMUP, TRIALS)
        r["io_bytes"] = x_bytes + TK * H
        r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
        r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
        shape_res["kernels"]["colwise_cute_x_gather_isa"] = r

        r = bench(lambda: colwise_quantize_cute(x, H, TK, isa_pack=False), WARMUP, TRIALS)
        r["io_bytes"] = x_bytes + TK * H
        r["bw_floor_us"] = round(bw_floor_us(r["io_bytes"]), 2)
        r["bw_util_pct"] = round(r["bw_floor_us"] / r["median_us"] * 100, 1) if r["median_us"] > 0 else 0
        shape_res["kernels"]["colwise_cute_x_raw"] = r

    # --- Summary: best per-task ---
    kernels = shape_res["kernels"]
    summary = {{}}
    # Best colwise for x (no gather)
    x_cands = {{k: v["median_us"] for k, v in kernels.items() if "colwise" in k and "_x" in k and "gather" not in k}}
    if x_cands:
        best = min(x_cands, key=x_cands.get)
        summary["best_colwise_x_nogather"] = {{"kernel": best, "us": x_cands[best]}}
    # Best colwise for x (with gather)
    x_gather = {{k: v["median_us"] for k, v in kernels.items() if "colwise" in k and "_x" in k and "gather" in k}}
    if x_gather:
        best = min(x_gather, key=x_gather.get)
        summary["best_colwise_x_gather"] = {{"kernel": best, "us": x_gather[best]}}
    # Speedup ratios
    if "colwise_triton_x" in kernels and "colwise_cute_x_isa" in kernels:
        t, c = kernels["colwise_triton_x"]["median_us"], kernels["colwise_cute_x_isa"]["median_us"]
        summary["cute_vs_triton_nogather"] = round(t / c, 3) if c > 0 else 0
    if "colwise_triton_x_gather" in kernels and "colwise_cute_x_gather_isa" in kernels:
        t, c = kernels["colwise_triton_x_gather"]["median_us"], kernels["colwise_cute_x_gather_isa"]["median_us"]
        summary["cute_vs_triton_gather"] = round(t / c, 3) if c > 0 else 0

    shape_res["summary"] = summary

    results["shapes"][tag] = shape_res
    del x, dz, y1s, gather_idx
    gc.collect(); torch.cuda.empty_cache()

print("__QUANT_BENCH_JSON__" + json.dumps(results))
''')


_WGRAD_BENCH_SCRIPT = textwrap.dedent(r'''
import gc, json, os, sys, statistics, torch

device = torch.device("cuda:0")
torch.cuda.set_device(device)
sys.path.insert(0, "{root}")

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

SHAPES = {shapes_json}
WARMUP = {warmup}
TRIALS = {trials}

def bench(fn, warmup, trials):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    return {{
        "median_us": round(statistics.median(trimmed), 2),
        "mean_us": round(statistics.mean(trimmed), 2),
        "min_us": round(min(times), 2),
        "max_us": round(max(times), 2),
        "p5_us": round(times[max(0, len(times)//20)], 2),
        "p95_us": round(times[min(len(times)-1, len(times)*19//20)], 2),
        "n_trials": trials,
    }}

results = {{"shapes": {{}}}}

for shape in SHAPES:
    T, H, I, E, K = shape["T"], shape["H"], shape["I"], shape["E"], shape["K"]
    tag = f"T={{T}}_H={{H}}_I={{I}}_E={{E}}_K={{K}}"
    print(f"  wgrad bench {{tag}} ...", flush=True)

    torch.manual_seed(42)
    model = MoE(E, K, H, I, ActivationType.SWIGLU, False, 0.02).to(device).to(torch.bfloat16)

    shape_res = {{"shape": shape, "modes": {{}}}}

    for mode_name in ("bf16", "fp8"):
        use_fp8 = (mode_name == "fp8")
        # Warmup
        for _ in range(15):
            xw = 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)
            with enable_quack_gemm(True):
                if use_fp8:
                    with enable_fp8(True):
                        ow, lw = model(xw, use_fp8=True)
                else:
                    with enable_fp8(False):
                        ow, lw = model(xw)
            (ow.sum() + lw).backward()
            model.zero_grad(set_to_none=True)
            del ow, lw, xw
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

        # Forward-only timing
        def run_fwd():
            x_t = 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)
            with torch.no_grad():
                with enable_quack_gemm(True):
                    if use_fp8:
                        with enable_fp8(True):
                            o, l = model(x_t, use_fp8=True)
                    else:
                        with enable_fp8(False):
                            o, l = model(x_t)
            del o, l, x_t

        fwd = bench(run_fwd, 5, TRIALS)

        # Full forward+backward timing
        def run_fwd_bwd():
            x_t = 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)
            with enable_quack_gemm(True):
                if use_fp8:
                    with enable_fp8(True):
                        o, l = model(x_t, use_fp8=True)
                else:
                    with enable_fp8(False):
                        o, l = model(x_t)
            (o.sum() + l).backward()
            model.zero_grad(set_to_none=True)
            del o, l, x_t

        fwd_bwd = bench(run_fwd_bwd, 5, TRIALS)

        # Backward = fwd_bwd - fwd
        bwd_median = fwd_bwd["median_us"] - fwd["median_us"]

        shape_res["modes"][mode_name] = {{
            "forward_us": fwd,
            "forward_backward_us": fwd_bwd,
            "backward_est_us": round(bwd_median, 2),
        }}

        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # Compute speedups
    bf16 = shape_res["modes"].get("bf16", {{}})
    fp8 = shape_res["modes"].get("fp8", {{}})
    if bf16 and fp8:
        bf16_fwdbwd = bf16.get("forward_backward_us", {{}}).get("median_us", 0)
        fp8_fwdbwd = fp8.get("forward_backward_us", {{}}).get("median_us", 0)
        bf16_fwd = bf16.get("forward_us", {{}}).get("median_us", 0)
        fp8_fwd = fp8.get("forward_us", {{}}).get("median_us", 0)
        bf16_bwd = bf16.get("backward_est_us", 0)
        fp8_bwd = fp8.get("backward_est_us", 0)
        shape_res["speedup"] = {{
            "fwd_bwd": round(bf16_fwdbwd / fp8_fwdbwd, 4) if fp8_fwdbwd > 0 else 0,
            "fwd": round(bf16_fwd / fp8_fwd, 4) if fp8_fwd > 0 else 0,
            "bwd": round(bf16_bwd / fp8_bwd, 4) if fp8_bwd > 0 else 0,
        }}

    del model
    gc.collect(); torch.cuda.empty_cache()
    results["shapes"][tag] = shape_res

    # Memory measurement (separate clean run)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)
    model = MoE(E, K, H, I, ActivationType.SWIGLU, False, 0.02).to(device).to(torch.bfloat16)
    for mode_name in ("bf16", "fp8"):
        use_fp8 = (mode_name == "fp8")
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect(); torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(device)
        x_t = 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)
        with enable_quack_gemm(True):
            if use_fp8:
                with enable_fp8(True):
                    o, l = model(x_t, use_fp8=True)
            else:
                with enable_fp8(False):
                    o, l = model(x_t)
        fwd_peak = torch.cuda.max_memory_allocated(device)
        (o.sum() + l).backward()
        bwd_peak = torch.cuda.max_memory_allocated(device)
        model.zero_grad(set_to_none=True)
        del o, l, x_t

        results["shapes"][tag]["modes"][mode_name]["memory_mib"] = {{
            "base": round(base / 1048576, 2),
            "fwd_peak": round(fwd_peak / 1048576, 2),
            "bwd_peak": round(bwd_peak / 1048576, 2),
        }}
        gc.collect(); torch.cuda.empty_cache()
    del model
    gc.collect(); torch.cuda.empty_cache()

print("__WGRAD_BENCH_JSON__" + json.dumps(results))
''')

# ═══════════════════════════════════════════════════════════════════════════════
# NCU Kernel Analysis Engine (--clock-control=none for real-world timing)
# ═══════════════════════════════════════════════════════════════════════════════
#
# NCU with clock-control=none gives realistic boost-clock timings that match
# nsys, unlike the default --clock-control=base which locks to base clock.
# This engine profiles individual quant kernels in isolation.

_NCU_QUANT_SCRIPT = textwrap.dedent(r'''
import gc, json, os, sys, torch
device = torch.device("cuda:0")
torch.cuda.set_device(device)
sys.path.insert(0, "{root}")

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_activation_blockscaled_fast,
    colwise_quantize_and_pack,
    fused_transpose_quantize_for_wgrad,
)

TK, H, I2 = {TK}, {H}, {I2}
x = torch.randn(TK, H, dtype=torch.bfloat16, device=device)
dz = torch.randn(TK, I2, dtype=torch.bfloat16, device=device)
gather_idx = torch.randint(0, TK, (TK,), dtype=torch.int32, device=device)

# Warmup
for _ in range(5):
    _ = quantize_activation_blockscaled_fast(x)
    _ = colwise_quantize_and_pack(x, H, TK)
torch.cuda.synchronize()

# Target kernel: {kernel_name}
torch.cuda.cudart().cudaProfilerStart()
for _ in range({ncu_iters}):
    {kernel_call}
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("NCU_DONE", flush=True)
''')

_NCU_KERNEL_REGISTRY = {
    "row_quant_x": "_ = quantize_activation_blockscaled_fast(x)",
    "row_quant_dz": "_ = quantize_activation_blockscaled_fast(dz)",
    "colwise_triton_x": "_ = colwise_quantize_and_pack(x, H, TK)",
    "colwise_triton_x_gather": "_ = colwise_quantize_and_pack(x, H, TK, gather_idx=gather_idx)",
    "colwise_triton_dz": "_ = colwise_quantize_and_pack(dz, I2, TK)",
}


def run_ncu_bench(
    gpu: int = 0,
    kernel_names: list[str] | None = None,
    ncu_iters: int = 3,
    clock_control: str = "none",
) -> dict:
    """Run NCU with --clock-control=none on specified quant kernels.

    Returns per-kernel timing from NCU (boost-clock realistic), plus
    key metrics: SM throughput, HBM throughput, occupancy.
    """
    python_bin = _resolve_python_bin()
    ncu_bin = "ncu"

    if kernel_names is None:
        kernel_names = list(_NCU_KERNEL_REGISTRY.keys())

    TK = SHAPE["T"] * SHAPE["K"]
    H = SHAPE["H"]
    I2 = 2 * SHAPE["I"]

    results: dict[str, Any] = {"kernels": {}, "clock_control": clock_control}

    for kname in kernel_names:
        if kname not in _NCU_KERNEL_REGISTRY:
            print(f"  [skip] unknown kernel: {kname}", flush=True)
            continue

        kernel_call = _NCU_KERNEL_REGISTRY[kname]
        script = _NCU_QUANT_SCRIPT.format(
            root=str(ROOT), TK=TK, H=H, I2=I2,
            kernel_name=kname, kernel_call=kernel_call,
            ncu_iters=ncu_iters,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix=f"ncu_{kname}_"
        ) as f:
            f.write(script)
            script_path = f.name

        print(f"  ncu profiling [{kname}] (clock={clock_control}) ...", flush=True)

        try:
            cmd = [
                ncu_bin,
                f"--clock-control={clock_control}",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--set=full",
                "--csv",
                "-c", str(ncu_iters),
                python_bin, script_path,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                env=_subprocess_env_for_gpu(gpu),
            )

            if proc.returncode != 0:
                print(f"    [WARN] ncu failed for {kname}: {proc.stderr[-300:]}", flush=True)
                results["kernels"][kname] = {"error": proc.stderr[-500:]}
                continue

            # Parse CSV output for Duration and key metrics
            lines = proc.stdout.split("\n")
            kernel_metrics: dict[str, Any] = {"raw_lines": []}
            for line in lines:
                if "Duration" in line or "duration" in line:
                    kernel_metrics["raw_lines"].append(line.strip())
                if "sm__throughput" in line.lower():
                    kernel_metrics["raw_lines"].append(line.strip())
                if "dram__throughput" in line.lower():
                    kernel_metrics["raw_lines"].append(line.strip())
                if "achieved_occupancy" in line.lower() or "sm__warps_active" in line.lower():
                    kernel_metrics["raw_lines"].append(line.strip())

            # Extract duration from CSV: look for numeric values after "Duration" header
            duration_values: list[float] = []
            for line in lines:
                parts = line.split(",")
                for i, part in enumerate(parts):
                    if "duration" in part.strip().lower() and "unit" not in part.lower():
                        # Next field or same field might have the value
                        for j in range(i + 1, min(i + 3, len(parts))):
                            try:
                                val = float(parts[j].strip().strip('"'))
                                if val > 0:
                                    duration_values.append(val)
                            except (ValueError, IndexError):
                                pass

            if duration_values:
                kernel_metrics["duration_us"] = round(sum(duration_values) / len(duration_values), 2)
                kernel_metrics["duration_min_us"] = round(min(duration_values), 2)
                kernel_metrics["duration_max_us"] = round(max(duration_values), 2)
                kernel_metrics["n_samples"] = len(duration_values)

            results["kernels"][kname] = kernel_metrics
            dur = kernel_metrics.get("duration_us", "?")
            print(f"    {kname}: {dur} µs (ncu clock={clock_control})", flush=True)

        except subprocess.TimeoutExpired:
            print(f"    [WARN] ncu timed out for {kname}", flush=True)
            results["kernels"][kname] = {"error": "timeout"}
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Wgrad FP8 Force Mode — bypass I-threshold for full-replacement analysis
# ═══════════════════════════════════════════════════════════════════════════════

_WGRAD_FORCE_FP8_SCRIPT = textwrap.dedent(r'''
import gc, json, os, sys, statistics, torch

device = torch.device("cuda:0")
torch.cuda.set_device(device)
sys.path.insert(0, "{root}")

os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_WGRAD"] = "{wgrad_mode}"

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

SHAPES = {shapes_json}
WARMUP = {warmup}
TRIALS = {trials}

def bench(fn, warmup, trials):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    return {{
        "median_us": round(statistics.median(trimmed), 2),
        "mean_us": round(statistics.mean(trimmed), 2),
        "min_us": round(min(times), 2),
        "max_us": round(max(times), 2),
        "p5_us": round(times[max(0, len(times)//20)], 2),
        "p95_us": round(times[min(len(times)-1, len(times)*19//20)], 2),
        "n_trials": trials,
    }}

def mem_breakdown(model, x_gen, use_fp8, device):
    """Detailed memory breakdown using torch.cuda.memory_stats()."""
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    base_alloc = torch.cuda.memory_allocated(device)
    x_t = x_gen()
    with enable_quack_gemm(True):
        if use_fp8:
            with enable_fp8(True):
                o, l = model(x_t, use_fp8=True)
        else:
            with enable_fp8(False):
                o, l = model(x_t)
    torch.cuda.synchronize()
    post_fwd_alloc = torch.cuda.memory_allocated(device)
    peak_fwd = torch.cuda.max_memory_allocated(device)

    stats_fwd = torch.cuda.memory_stats(device)
    fwd_allocs = stats_fwd.get("allocation.all.allocated", 0)
    fwd_frees = stats_fwd.get("allocation.all.freed", 0)

    torch.cuda.reset_peak_memory_stats(device)
    (o.sum() + l).backward()
    torch.cuda.synchronize()
    post_bwd_alloc = torch.cuda.memory_allocated(device)
    peak_bwd = torch.cuda.max_memory_allocated(device)

    stats_bwd = torch.cuda.memory_stats(device)
    bwd_allocs = stats_bwd.get("allocation.all.allocated", 0)
    bwd_frees = stats_bwd.get("allocation.all.freed", 0)
    # Segment info
    active_blocks = stats_bwd.get("active.all.current", 0)
    active_bytes = stats_bwd.get("active_bytes.all.current", 0)

    model.zero_grad(set_to_none=True)
    del o, l, x_t
    gc.collect(); torch.cuda.empty_cache()

    return {{
        "base_mib": round(base_alloc / 1048576, 2),
        "post_fwd_mib": round(post_fwd_alloc / 1048576, 2),
        "peak_fwd_mib": round(peak_fwd / 1048576, 2),
        "post_bwd_mib": round(post_bwd_alloc / 1048576, 2),
        "peak_bwd_mib": round(peak_bwd / 1048576, 2),
        "fwd_net_allocs": fwd_allocs - fwd_frees,
        "bwd_total_allocs": bwd_allocs,
        "bwd_total_frees": bwd_frees,
        "bwd_active_blocks": active_blocks,
        "bwd_active_bytes_mib": round(active_bytes / 1048576, 2),
    }}

results = {{"shapes": {{}}}}

for shape in SHAPES:
    T, H, I, E, K = shape["T"], shape["H"], shape["I"], shape["E"], shape["K"]
    tag = f"T={{T}}_H={{H}}_I={{I}}_E={{E}}_K={{K}}"
    print(f"  wgrad-force bench {{tag}} (wgrad={wgrad_mode}) ...", flush=True)

    torch.manual_seed(42)
    model = MoE(E, K, H, I, ActivationType.SWIGLU, False, 0.02).to(device).to(torch.bfloat16)

    shape_res = {{"shape": shape, "wgrad_mode": "{wgrad_mode}", "modes": {{}}}}

    for mode_name in ("bf16", "fp8"):
        use_fp8 = (mode_name == "fp8")
        # Warmup
        for _ in range(15):
            xw = 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)
            with enable_quack_gemm(True):
                if use_fp8:
                    with enable_fp8(True):
                        ow, lw = model(xw, use_fp8=True)
                else:
                    with enable_fp8(False):
                        ow, lw = model(xw)
            (ow.sum() + lw).backward()
            model.zero_grad(set_to_none=True)
            del ow, lw, xw
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

        # Full forward+backward timing
        def run_fwd_bwd():
            x_t = 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)
            with enable_quack_gemm(True):
                if use_fp8:
                    with enable_fp8(True):
                        o, l = model(x_t, use_fp8=True)
                else:
                    with enable_fp8(False):
                        o, l = model(x_t)
            (o.sum() + l).backward()
            model.zero_grad(set_to_none=True)
            del o, l, x_t

        fwd_bwd = bench(run_fwd_bwd, 5, TRIALS)

        # Memory breakdown (3 runs, take last)
        x_gen = lambda: 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)
        mem = mem_breakdown(model, x_gen, use_fp8, device)

        shape_res["modes"][mode_name] = {{
            "forward_backward_us": fwd_bwd,
            "memory": mem,
        }}
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # Compute speedups
    bf16 = shape_res["modes"].get("bf16", {{}})
    fp8 = shape_res["modes"].get("fp8", {{}})
    if bf16 and fp8:
        bf16_us = bf16.get("forward_backward_us", {{}}).get("median_us", 0)
        fp8_us = fp8.get("forward_backward_us", {{}}).get("median_us", 0)
        shape_res["speedup"] = round(bf16_us / fp8_us, 4) if fp8_us > 0 else 0
        shape_res["delta_us"] = round(fp8_us - bf16_us, 1)
        # Memory comparison
        bf16_mem = bf16.get("memory", {{}})
        fp8_mem = fp8.get("memory", {{}})
        shape_res["memory_delta"] = {{
            "peak_fwd_delta_mib": round(fp8_mem.get("peak_fwd_mib", 0) - bf16_mem.get("peak_fwd_mib", 0), 2),
            "peak_bwd_delta_mib": round(fp8_mem.get("peak_bwd_mib", 0) - bf16_mem.get("peak_bwd_mib", 0), 2),
        }}

    results["shapes"][tag] = shape_res
    del model; gc.collect(); torch.cuda.empty_cache()

print("__WGRAD_FORCE_JSON__" + json.dumps(results))
''')


def run_wgrad_force_fp8(
    gpu: int = 0,
    shapes: list[dict] | None = None,
    warmup: int = 10,
    trials: int = 30,
    force_wgrad: bool = True,
) -> dict:
    """Run wgrad benchmark with SONIC_MOE_FP8_WGRAD forced on or off.

    This bypasses the I-threshold to measure wgrad FP8 at ALL shapes,
    including I=1536 where it's normally disabled.
    """
    if shapes is None:
        shapes = [dict(SHAPE)]

    python_bin = _resolve_python_bin()
    wgrad_mode = "1" if force_wgrad else "0"

    script = _WGRAD_FORCE_FP8_SCRIPT.format(
        root=str(ROOT),
        shapes_json=json.dumps(shapes),
        warmup=warmup,
        trials=trials,
        wgrad_mode=wgrad_mode,
    )

    env = _subprocess_env_for_gpu(gpu)
    env["USE_QUACK_GEMM"] = "1"

    print(f"  Running wgrad-force benchmark (wgrad={wgrad_mode}) ...", flush=True)
    proc = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True, text=True, timeout=900, env=env,
        cwd=str(ROOT),
    )

    for line in proc.stdout.split("\n"):
        if line.startswith("__WGRAD_FORCE_JSON__"):
            data = json.loads(line[len("__WGRAD_FORCE_JSON__"):])
            return data

    print(f"  WARNING: No wgrad-force JSON found", flush=True)
    if proc.stderr:
        print(f"  stderr (last 1000 chars): {proc.stderr[-1000:]}", flush=True)
    if proc.stdout:
        print(f"  stdout (last 500 chars): {proc.stdout[-500:]}", flush=True)
    return {}


def _print_wgrad_force_summary(data: dict) -> None:
    """Pretty-print wgrad force benchmark results."""
    for shape_key, shape_res in data.get("shapes", {}).items():
        shape = shape_res.get("shape", {})
        wgrad_mode = shape_res.get("wgrad_mode", "?")
        print(f"\n  === Wgrad-Force Benchmark (wgrad={wgrad_mode}): "
              f"T={shape.get('T')}, H={shape.get('H')}, I={shape.get('I')} ===")

        modes = shape_res.get("modes", {})
        bf16 = modes.get("bf16", {})
        fp8 = modes.get("fp8", {})

        bf16_us = bf16.get("forward_backward_us", {}).get("median_us", 0)
        fp8_us = fp8.get("forward_backward_us", {}).get("median_us", 0)
        speedup = shape_res.get("speedup", 0)
        delta = shape_res.get("delta_us", 0)

        print(f"  BF16: {bf16_us:.0f} µs  |  FP8: {fp8_us:.0f} µs  |  "
              f"Speedup: {speedup:.4f}×  |  Delta: {delta:+.0f} µs")

        # Memory
        mem_delta = shape_res.get("memory_delta", {})
        for mode_name in ("bf16", "fp8"):
            mem = modes.get(mode_name, {}).get("memory", {})
            if mem:
                print(f"  [{mode_name}] base={mem.get('base_mib', '?')} MiB  "
                      f"peak_fwd={mem.get('peak_fwd_mib', '?')} MiB  "
                      f"peak_bwd={mem.get('peak_bwd_mib', '?')} MiB  "
                      f"bwd_active={mem.get('bwd_active_bytes_mib', '?')} MiB")
        if mem_delta:
            print(f"  [delta] fwd={mem_delta.get('peak_fwd_delta_mib', '?'):+.1f} MiB  "
                  f"bwd={mem_delta.get('peak_bwd_delta_mib', '?'):+.1f} MiB")


# Target GPU peak HBM bandwidth (GB/s)
_PEAK_HBM_GBPS = 8000


def run_quant_bench(
    gpu: int = 0,
    shapes: list[dict] | None = None,
    warmup: int = 10,
    trials: int = 50,
) -> dict:
    """Run isolated quant kernel benchmark via CUDA events in subprocess."""
    if shapes is None:
        TK = SHAPE["T"] * SHAPE["K"]
        H = SHAPE["H"]
        I = SHAPE["I"]
        shapes = [{"TK": TK, "H": H, "I2": 2 * I, "I1": I}]

    python_bin = _resolve_python_bin()

    script = _QUANT_BENCH_SCRIPT.format(
        root=str(ROOT),
        shapes_json=json.dumps(shapes),
        warmup=warmup,
        trials=trials,
        peak_hbm_gbps=_PEAK_HBM_GBPS,
    )

    env = _subprocess_env_for_gpu(gpu)
    env["USE_QUACK_GEMM"] = "1"
    env["SONIC_MOE_FP8_MODE"] = "perf"

    print("  Running quant kernel benchmark (subprocess) ...", flush=True)
    proc = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True, text=True, timeout=600, env=env,
        cwd=str(ROOT),
    )

    for line in proc.stdout.split("\n"):
        if line.startswith("__QUANT_BENCH_JSON__"):
            data = json.loads(line[len("__QUANT_BENCH_JSON__"):])
            return data

    print(f"  WARNING: No quant bench JSON found", flush=True)
    if proc.stderr:
        print(f"  stderr (last 1000 chars): {proc.stderr[-1000:]}", flush=True)
    if proc.stdout:
        print(f"  stdout (last 500 chars): {proc.stdout[-500:]}", flush=True)
    return {}


def run_wgrad_bench(
    gpu: int = 0,
    shapes: list[dict] | None = None,
    warmup: int = 10,
    trials: int = 30,
) -> dict:
    """Run wgrad FP8 vs BF16 end-to-end benchmark."""
    if shapes is None:
        shapes = [dict(SHAPE)]

    python_bin = _resolve_python_bin()

    script = _WGRAD_BENCH_SCRIPT.format(
        root=str(ROOT),
        shapes_json=json.dumps(shapes),
        warmup=warmup,
        trials=trials,
    )

    env = _subprocess_env_for_gpu(gpu)
    env["USE_QUACK_GEMM"] = "1"

    print("  Running wgrad benchmark (subprocess) ...", flush=True)
    proc = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True, text=True, timeout=900, env=env,
        cwd=str(ROOT),
    )

    for line in proc.stdout.split("\n"):
        if line.startswith("__WGRAD_BENCH_JSON__"):
            data = json.loads(line[len("__WGRAD_BENCH_JSON__"):])
            return data

    print(f"  WARNING: No wgrad bench JSON found", flush=True)
    if proc.stderr:
        print(f"  stderr (last 1000 chars): {proc.stderr[-1000:]}", flush=True)
    if proc.stdout:
        print(f"  stdout (last 500 chars): {proc.stdout[-500:]}", flush=True)
    return {}


def _print_quant_bench_summary(data: dict) -> None:
    """Pretty-print quant bench results to terminal."""
    for shape_key, shape_res in data.get("shapes", {}).items():
        TK = shape_res.get("TK", "?")
        H = shape_res.get("H", "?")
        I2 = shape_res.get("I2", "?")
        print(f"\n  === Quant Kernel Benchmark: TK={TK}, H={H}, I2={I2} ===")
        print(f"  {'Kernel':<45s} {'Median':>8s} {'Min':>8s} {'P95':>8s} {'BW%':>5s}")
        print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
        for kname, kdata in shape_res.get("kernels", {}).items():
            med = kdata.get("median_us", 0)
            mn = kdata.get("min_us", 0)
            p95 = kdata.get("p95_us", 0)
            bw = kdata.get("bw_util_pct", 0)
            print(f"  {kname:<45s} {med:>7.1f}µ {mn:>7.1f}µ {p95:>7.1f}µ {bw:>4.0f}%")

        summary = shape_res.get("summary", {})
        if summary:
            print(f"\n  Summary:")
            for k, v in summary.items():
                if isinstance(v, dict):
                    print(f"    {k}: {v.get('kernel', '?')} = {v.get('us', '?')}µs")
                else:
                    print(f"    {k}: {v}×")


def _print_wgrad_bench_summary(data: dict) -> None:
    """Pretty-print wgrad bench results to terminal."""
    for shape_key, shape_res in data.get("shapes", {}).items():
        shape = shape_res.get("shape", {})
        print(f"\n  === Wgrad Benchmark: T={shape.get('T')}, H={shape.get('H')}, "
              f"I={shape.get('I')}, E={shape.get('E')}, K={shape.get('K')} ===")
        print(f"  {'':20s} {'BF16':>12s} {'FP8':>12s} {'Speedup':>9s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*9}")

        modes = shape_res.get("modes", {})
        bf16 = modes.get("bf16", {})
        fp8 = modes.get("fp8", {})
        speedup = shape_res.get("speedup", {})

        for phase, phase_key in [("Fwd+Bwd", "forward_backward_us"), ("Forward", "forward_us")]:
            bf16_v = bf16.get(phase_key, {}).get("median_us", 0)
            fp8_v = fp8.get(phase_key, {}).get("median_us", 0)
            sp = bf16_v / fp8_v if fp8_v > 0 else 0
            print(f"  {phase:<20s} {bf16_v:>10.0f}µs {fp8_v:>10.0f}µs {sp:>8.3f}×")

        bf16_bwd = bf16.get("backward_est_us", 0)
        fp8_bwd = fp8.get("backward_est_us", 0)
        sp_bwd = bf16_bwd / fp8_bwd if fp8_bwd > 0 else 0
        print(f"  {'Backward (est)':20s} {bf16_bwd:>10.0f}µs {fp8_bwd:>10.0f}µs {sp_bwd:>8.3f}×")

        # Memory
        for mode_name in ("bf16", "fp8"):
            mem = modes.get(mode_name, {}).get("memory_mib", {})
            if mem:
                print(f"  [{mode_name}] peak fwd={mem.get('fwd_peak', '?')} MiB, "
                      f"bwd={mem.get('bwd_peak', '?')} MiB")


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    mode: str = "trace",
    *,
    gpu: int = 0,
    precision_seeds: list[int] | None = None,
    bench_repeats: int = DEFAULT_BENCH_REPEATS,
    profile_trials: int = 1,
    nsys_warmup: int = DEFAULT_NSYS_WARMUP,
    nsys_iters: int = DEFAULT_NSYS_ITERS,
    nsys_shapes: list[dict[str, int]] | None = None,
    nsys_output: str | None = None,
    quant_bench_shapes: list[dict] | None = None,
    quant_bench_trials: int = 50,
    wgrad_bench_shapes: list[dict] | None = None,
    wgrad_bench_trials: int = 30,
) -> dict:
    """Main entry point: run introspection and write manifest.json.

    Parameters
    ----------
    mode : str
        "trace" | "profile" | "full" | "nsys" | "quant-bench" | "wgrad-bench" | "compile-session53"
    """
    # compile-session53: pure data aggregation, no GPU needed
    if mode == "compile-session53":
        return run_compile_session53()

    # grid: parallel multi-GPU nsys profiling
    if mode == "grid":
        return run_grid(
            num_gpus=gpu or 8,  # --gpu doubles as num_gpus for grid mode
            nsys_warmup=nsys_warmup,
            nsys_iters=nsys_iters,
            grid_output=nsys_output,
        )

    import torch

    precision_seeds = precision_seeds or list(DEFAULT_PRECISION_SEEDS)

    print("=" * 60)
    print(f"SonicMoE Introspection Engine  [mode={mode}]")
    print("=" * 60)

    # Setup
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    # Metadata
    gpu_name = torch.cuda.get_device_name(device)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shape": SHAPE,
        "device": gpu_name,
        "gpu_index": gpu,
        "torch_version": torch.__version__,
        "mode": mode,
        "precision_seeds": precision_seeds if mode == "full" else [],
        "bench_repeats": bench_repeats if mode == "full" else 0,
        "profile_trials": profile_trials if mode in ("profile", "full") else 0,
    }
    try:
        import quack
        metadata["quack_version"] = getattr(quack, "__version__", "unknown")
    except ImportError:
        metadata["quack_version"] = "not installed"

    print(f"  Device: {gpu_name}")
    if nsys_shapes:
        labels = ["T%d_E%d" % (s["T"], s["E"]) for s in nsys_shapes]
        print(f"  Shapes: {len(nsys_shapes)} — {labels}")
    else:
        print(f"  Shape: T={SHAPE['T']}, H={SHAPE['H']}, I={SHAPE['I']}, "
              f"E={SHAPE['E']}, K={SHAPE['K']}")

    # ── precision mode: standalone multi-shape precision audit ──
    if mode == "precision":
        shapes = nsys_shapes or [SHAPE]
        results_all = {}
        for i, shape in enumerate(shapes):
            shape_key = f"T{shape['T']}_I{shape['I']}_E{shape['E']}K{shape['K']}"
            saved = dict(SHAPE)
            SHAPE.update(shape)
            print(f"\n  [{i+1}/{len(shapes)}] Precision {shape_key} ({len(precision_seeds)} seeds) ...",
                  flush=True)
            try:
                prec = run_precision_audit_isolated(gpu=gpu, seeds=precision_seeds)
                rr = prec.get("rrmse_pct", {})
                print(f"    output={rr.get('output','?')}%  dx={rr.get('dx','?')}%  "
                      f"dw1={rr.get('dw1','?')}%  dw2={rr.get('dw2','?')}%")
                results_all[shape_key] = prec
            except Exception as ex:
                print(f"    ERROR: {ex}")
                results_all[shape_key] = {"error": str(ex)}
            finally:
                SHAPE.update(saved)
        return results_all

    # ── nsys mode: standalone GPU-projection profiling ──
    if mode == "nsys":
        shapes = nsys_shapes or [SHAPE]
        print(f"\n  nsys GPU-projection profiling ({len(shapes)} shape(s)) ...")
        nsys_data = run_nsys_profile(
            gpu=gpu, warmup=nsys_warmup, iters=nsys_iters, shapes=shapes,
        )
        nsys_data["metadata"] = metadata

        # Write nsys results
        nsys_out = Path(nsys_output) if nsys_output else NSYS_BREAKDOWN_PATH
        nsys_out.parent.mkdir(parents=True, exist_ok=True)
        nsys_out.write_text(json.dumps(nsys_data, indent=2, default=str))
        print(f"\n  → {nsys_out} ({nsys_out.stat().st_size / 1024:.1f} KB)")

        # Print summary table
        print("\n" + "=" * 90)
        print("  nsys GPU-Projection + Memory (FP8 frontier = stash mode)")
        print("=" * 90)
        print(f"  {'Shape':<28s} {'BF16 µs':>8s} {'FP8 µs':>8s} {'Speed':>6s} "
              f"{'BF16 Bwd':>9s} {'FP8 Bwd':>9s} {'MemΔ%':>7s}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*9} {'-'*7}")
        for shape_key, shape_res in nsys_data.get("shapes", {}).items():
            bf16_us = shape_res.get("bf16", {}).get("per_iter_us", 0)
            fp8_us = shape_res.get("fp8", {}).get("per_iter_us", 0)
            speedup = shape_res.get("speedup", 0)
            bf16_bwd = shape_res.get("memory_bf16", {}).get("peak_bwd_mib", 0)
            fp8_bwd = shape_res.get("memory_fp8", {}).get("peak_bwd_mib", 0)
            mem_pct = round((fp8_bwd - bf16_bwd) / bf16_bwd * 100, 1) if bf16_bwd else 0
            print(f"  {shape_key:<28s} {bf16_us:>7.0f}  {fp8_us:>7.0f}  "
                  f"{speedup:>5.3f}× {bf16_bwd:>8.0f}M {fp8_bwd:>8.0f}M {mem_pct:>+6.1f}%")

        # Budget breakdown for each shape
        for shape_key, shape_res in nsys_data.get("shapes", {}).items():
            bb = shape_res.get("budget_breakdown")
            if not bb:
                continue
            print(f"\n  --- {shape_key} FP8 Budget ---")
            print(f"  {'Category':<30s} {'BF16':>7s} {'FP8':>7s} {'Delta':>8s}")
            print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*8}")
            for cat in sorted(bb, key=lambda c: bb[c]["delta_us"]):
                d = bb[cat]
                tag = "SAVE" if d["delta_us"] < -5 else ("COST" if d["delta_us"] > 5 else "")
                print(f"  {cat:<30s} {d['bf16_us']:>6.0f}  {d['fp8_us']:>6.0f}  "
                      f"{d['delta_us']:>+7.0f}  {tag}")
            net = shape_res.get("budget_net_us", 0)
            print(f"  {'NET':<30s} {'':>7s} {'':>7s} {net:>+7.0f}  "
                  f"→ FP8 {'wins' if net < 0 else 'loses'}")

        print("=" * 90)
        return nsys_data

    # ── report mode: comprehensive per-kernel + memory + precision + autograd ──
    if mode == "report":
        shapes = nsys_shapes or [SHAPE]
        official_python = os.environ.get("SONIC_MOE_OFFICIAL_BF16_PYTHON", sys.executable)
        official_root = os.environ.get("SONIC_MOE_OFFICIAL_REPO", str(ROOT))
        nsys_output_dir = Path(
            os.environ.get("SONIC_MOE_NSYS_OUTPUT_DIR", str(ROOT / "reports" / "nsys"))
        )
        nsys_output_dir.mkdir(parents=True, exist_ok=True)

        all_reports = {"metadata": metadata, "shapes": {}}

        for shape in shapes:
            E = shape["E"]
            shape_key = f"T{shape['T']}_I{shape['I']}_E{E}K{shape['K']}"
            print(f"\n{'='*80}")
            print(f"  Report: {shape_key}")
            print(f"{'='*80}")

            shape_report: dict[str, Any] = {"shape": shape}

            # ── 1. nsys BF16 + FP8 (our branch, aligned with official) ────
            print(f"\n  [1/3] nsys profiling (bf16 + fp8) ...", flush=True)
            fp8_nsys = run_nsys_profile(
                gpu=gpu, warmup=nsys_warmup, iters=nsys_iters, shapes=[shape],
            )
            fp8_shape_data = fp8_nsys.get("shapes", {}).get(shape_key, {})
            shape_report["fp8_frontier"] = fp8_shape_data.get("fp8", {})
            shape_report["branch_bf16"] = fp8_shape_data.get("bf16", {})
            shape_report["_nsys_raw"] = fp8_shape_data  # for summary table
            fp8_us = shape_report["fp8_frontier"].get("per_iter_us", 0)
            bf16_branch_us = shape_report["branch_bf16"].get("per_iter_us", 0)
            print(f"    BF16: {bf16_branch_us} µs/iter  FP8: {fp8_us} µs/iter")

            # Speedup (use our branch bf16 — verified aligned with official)
            if bf16_branch_us > 0 and fp8_us > 0:
                shape_report["speedup"] = round(bf16_branch_us / fp8_us, 4)
                print(f"    Speedup: {shape_report['speedup']}×")

            # ── 3. Memory (paired, subprocess) ───────────────────────────
            print(f"\n  [2/3] Memory breakdown ...", flush=True)
            for mem_mode in ("bf16", "fp8"):
                mem = _run_memory_measure(mem_mode, shape, gpu, warmup=3)
                shape_report[f"memory_{mem_mode}"] = mem
                if "error" not in mem:
                    print(f"    {mem_mode}: base={mem['base_mib']:.0f} fwd={mem['peak_fwd_mib']:.0f} bwd={mem['peak_bwd_mib']:.0f} MiB")

            # ── 4. Precision (subprocess-isolated) ───────────────────────
            print(f"\n  [3/3] Precision ({len(precision_seeds)} seeds) ...", flush=True)
            # Set SHAPE to current shape so precision subprocess uses correct params
            saved_shape = dict(SHAPE)
            SHAPE.update(shape)
            try:
                prec = run_precision_audit_isolated(gpu=gpu, seeds=precision_seeds)
                shape_report["precision"] = prec
                rr = prec.get("rrmse_pct", {})
                print(f"    output={rr.get('output','?')}% dx={rr.get('dx','?')}% "
                      f"dw1={rr.get('dw1','?')}% dw2={rr.get('dw2','?')}%")
            except Exception as ex:
                shape_report["precision"] = {"error": str(ex)}
                print(f"    ERROR: {ex}")
            finally:
                SHAPE.update(saved_shape)

            # Budget breakdown (use our branch bf16 since it aligns with official)
            bf16_data = fp8_shape_data.get("bf16", {})
            bf16_cats = bf16_data.get("category_summary", {})
            fp8_cats = shape_report.get("fp8_frontier", {}).get("category_summary", {})
            if bf16_cats and fp8_cats:
                all_cats = sorted(set(list(bf16_cats) + list(fp8_cats)))
                bb = {}
                for cat in all_cats:
                    b, f = bf16_cats.get(cat, 0), fp8_cats.get(cat, 0)
                    bb[cat] = {"bf16": round(b,1), "fp8": round(f,1), "delta": round(f-b,1)}
                shape_report["budget"] = bb

            all_reports["shapes"][shape_key] = shape_report

        # Write report
        report_path = ROOT / "reports" / "session53_full_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(all_reports, indent=2, default=str))
        print(f"\n  → {report_path}")

        # Print final summary
        print(f"\n{'='*90}")
        print(f"  FULL REPORT SUMMARY")
        print(f"{'='*90}")
        print(f"  {'Shape':<28s} {'BF16':>9s} {'FP8':>9s} {'Speed':>7s} "
              f"{'BF16bwd':>8s} {'FP8bwd':>8s} {'Prec':>6s}")
        print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*7} {'-'*8} {'-'*8} {'-'*6}")
        for sk, sr in all_reports["shapes"].items():
            bf = sr.get("fp8_frontier", {}).get("_bf16_per_iter_us",
                 sr.get("official_bf16", {}).get("per_iter_us", 0))
            # Use our branch bf16 from the nsys run if available
            fp8_nsys_data = sr.get("_nsys_raw", {})
            bf16_data_raw = fp8_nsys_data.get("bf16", {})
            if bf16_data_raw:
                bf = bf16_data_raw.get("per_iter_us", bf)
            fp = sr.get("fp8_frontier", {}).get("per_iter_us", 0)
            sp = round(bf / fp, 3) if fp > 0 and bf > 0 else 0
            bf_bwd = sr.get("memory_bf16", {}).get("peak_bwd_mib", 0)
            fp_bwd = sr.get("memory_fp8", {}).get("peak_bwd_mib", 0)
            prec = sr.get("precision", {})
            prec_ok = "PASS" if isinstance(prec, dict) and prec.get("rrmse_pct") and all(
                prec.get("rrmse_pct", {}).get(t, 999) < 10
                for t in ["output", "dx"]
            ) else "?"
            print(f"  {sk:<28s} {bf:>8.0f}  {fp:>8.0f}  {sp:>6.3f}× "
                  f"{bf_bwd:>7.0f}M {fp_bwd:>7.0f}M {prec_ok:>6s}")
        print(f"{'='*90}")
        return all_reports

    # ── quant-bench mode: isolated quant kernel CUDA-event benchmark ──
    if mode == "quant-bench":
        print(f"\n  Quant kernel CUDA-event benchmark ...")
        quant_data = run_quant_bench(
            gpu=gpu,
            shapes=quant_bench_shapes,
            trials=quant_bench_trials,
        )
        quant_data["metadata"] = metadata
        out_path = ROOT / "reports" / "quant_bench.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(quant_data, indent=2, default=str))
        print(f"\n  → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
        _print_quant_bench_summary(quant_data)
        print("=" * 60)
        return quant_data

    # ── wgrad-bench mode: FP8 vs BF16 wgrad end-to-end benchmark ──
    if mode == "wgrad-bench":
        print(f"\n  Wgrad FP8 vs BF16 benchmark ...")
        wgrad_data = run_wgrad_bench(
            gpu=gpu,
            shapes=wgrad_bench_shapes,
            trials=wgrad_bench_trials,
        )
        wgrad_data["metadata"] = metadata
        out_path = ROOT / "reports" / "wgrad_bench.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(wgrad_data, indent=2, default=str))
        print(f"\n  → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
        _print_wgrad_bench_summary(wgrad_data)
        print("=" * 60)
        return wgrad_data

    # ── ncu-bench mode: NCU kernel analysis with clock-control=none ──
    if mode == "ncu-bench":
        print(f"\n  NCU quant kernel analysis (clock-control=none) ...")
        ncu_data = run_ncu_bench(gpu=gpu)
        ncu_data["metadata"] = metadata
        out_path = ROOT / "reports" / "ncu_quant_bench.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(ncu_data, indent=2, default=str))
        print(f"\n  → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
        print("=" * 60)
        return ncu_data

    # ── pad-audit mode: route-level padding precision / perf / memory ──
    if mode == "pad-audit":
        print("\n  Route-Level Padding Audit (E=32, 4-way)")
        print("  =========================================")
        print("  BF16 raw (gold) | BF16+rounding | FP8+padding | FP8+rounding")
        print("  BF16 raw = gold standard. Others compared against it.")
        print()
        pad_data = run_pad_audit(
            gpu=gpu,
            nsys_warmup=nsys_warmup,
            nsys_iters=nsys_iters,
            seeds=precision_seeds,
            T=SHAPE["T"], H=SHAPE["H"], I=SHAPE["I"],
            E=32, K=SHAPE["K"],
        )
        pad_data["metadata"] = metadata

        pad_out = ROOT / "reports" / "pad_audit_results.json"
        pad_out.parent.mkdir(parents=True, exist_ok=True)
        pad_out.write_text(json.dumps(pad_data, indent=2, default=str))

        print(f"\n  → {pad_out} ({pad_out.stat().st_size / 1024:.1f} KB)")
        print("=" * 70)
        return pad_data

    # ── wgrad-force mode: forced wgrad FP8 at all shapes ──
    if mode == "wgrad-force":
        # Run with wgrad forced ON (bypass I-threshold) and OFF for comparison
        all_shapes = wgrad_bench_shapes or [dict(SHAPE)]
        # Also sweep I=2048, I=3072 if only default shape given
        if len(all_shapes) == 1 and all_shapes[0].get("I", 0) == SHAPE["I"]:
            base = dict(all_shapes[0])
            all_shapes = [
                base,
                {**base, "I": 2048},
                {**base, "I": 3072},
            ]

        print(f"\n  Wgrad FP8 force-ON benchmark ({len(all_shapes)} shapes) ...")
        on_data = run_wgrad_force_fp8(
            gpu=gpu, shapes=all_shapes, trials=wgrad_bench_trials, force_wgrad=True,
        )
        on_data["metadata"] = metadata
        on_data["wgrad_forced"] = True

        print(f"\n  Wgrad FP8 force-OFF benchmark ({len(all_shapes)} shapes) ...")
        off_data = run_wgrad_force_fp8(
            gpu=gpu, shapes=all_shapes, trials=wgrad_bench_trials, force_wgrad=False,
        )
        off_data["wgrad_forced"] = False

        combined = {
            "metadata": metadata,
            "wgrad_on": on_data,
            "wgrad_off": off_data,
            "analysis": {},
        }
        # Compare: for each shape, report speedup delta between on and off
        for tag in on_data.get("shapes", {}):
            on_sp = on_data["shapes"][tag].get("speedup", 0)
            off_sp = off_data.get("shapes", {}).get(tag, {}).get("speedup", 0)
            on_us = on_data["shapes"][tag].get("modes", {}).get("fp8", {}).get(
                "forward_backward_us", {}).get("median_us", 0)
            off_us = off_data.get("shapes", {}).get(tag, {}).get("modes", {}).get("fp8", {}).get(
                "forward_backward_us", {}).get("median_us", 0)
            combined["analysis"][tag] = {
                "wgrad_on_speedup": on_sp,
                "wgrad_off_speedup": off_sp,
                "wgrad_on_fp8_us": on_us,
                "wgrad_off_fp8_us": off_us,
                "wgrad_delta_us": round(on_us - off_us, 1) if on_us and off_us else None,
            }

        out_path = ROOT / "reports" / "wgrad_force_bench.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(combined, indent=2, default=str))
        print(f"\n  → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

        print("\n" + "=" * 60)
        print("  Wgrad Force FP8 Analysis")
        print("=" * 60)
        _print_wgrad_force_summary(on_data)
        print("\n  --- Wgrad OFF comparison ---")
        _print_wgrad_force_summary(off_data)

        print("\n  --- Shape-wise wgrad impact ---")
        for tag, analysis in combined["analysis"].items():
            print(f"  {tag}: ON={analysis['wgrad_on_speedup']:.4f}× "
                  f"OFF={analysis['wgrad_off_speedup']:.4f}× "
                  f"delta={analysis.get('wgrad_delta_us', '?')} µs")
        print("=" * 60)
        return combined

    # ── Trace BF16 ──
    print("\n[1/5] Tracing BF16 (subprocess-isolated) ...", flush=True)
    bf16_manifest = run_trace_isolated("bf16", gpu)
    print(f"       → {len(bf16_manifest.tensors)} tensors tracked, "
           f"{len(bf16_manifest.phase_memory)} phase snapshots")

    # ── Trace FP8 ──
    print("\n[2/5] Tracing FP8 (subprocess-isolated) ...", flush=True)
    fp8_manifest = run_trace_isolated("fp8", gpu)
    print(f"       → {len(fp8_manifest.tensors)} tensors tracked, "
           f"{len(fp8_manifest.phase_memory)} phase snapshots")

    kernel_data = None
    benchmark_summary = None
    benchmark_report = None
    profiler_runs = None
    mem_breakdown_payload = None
    compat_kernel_payload = None

    if mode == "profile":
        print("\n[3/5] Kernel profiling (subprocess) ...", flush=True)
        kernel_data = run_kernel_profile(gpu=gpu)
        for m in ("bf16", "fp8"):
            if m in (kernel_data or {}):
                kd = kernel_data[m]
                print(f"       [{m}] {kd['total_cuda_us']:.1f} µs CUDA, "
                       f"{kd['wall_clock_ms']:.2f} ms wall, "
                       f"{len(kd['kernels'])} kernels")
    elif mode == "full":
        print("\n[3/5] Rigorous profiler + benchmark ...", flush=True)
        profiler_runs = run_rigorous_profiler(gpu=gpu, repeats=profile_trials)
        kernel_data = _build_manifest_kernel_data(profiler_runs)
        if kernel_data is None:
            print("       profiler unavailable, falling back to lightweight kernel profile", flush=True)
            kernel_data = run_kernel_profile(gpu=gpu)
        else:
            for m in ("bf16", "fp8"):
                kd = kernel_data[m]
                print(f"       [{m}] {kd['total_cuda_us']:.1f} µs CUDA, "
                      f"{kd['wall_clock_ms']:.2f} ms wall, "
                      f"{len(kd['kernels'])} kernels "
                      f"(trials={kd.get('profile_trials', 1)})")

        benchmark_report = run_rigorous_benchmark(
            gpu=gpu,
            seeds=precision_seeds,
            repeats=bench_repeats,
        )
        benchmark_summary = _summarize_benchmark_report(benchmark_report)
        if benchmark_summary:
            cmp_stash = benchmark_summary.get("comparisons", {}).get("fp8_stash_vs_bf16", {})
            fwd_delta = cmp_stash.get("fwd_peak_delta_mib")
            bwd_delta = cmp_stash.get("bwd_peak_delta_mib")
            if fwd_delta is not None and bwd_delta is not None:
                print(f"       [stash] fwd {fwd_delta:+.1f} MiB, bwd {bwd_delta:+.1f} MiB vs BF16", flush=True)

        mem_breakdown_payload = _build_mem_breakdown_json(profiler_runs, benchmark_summary)
        compat_kernel_payload = _build_compat_kernel_breakdown(kernel_data)
        _write_json_artifact(BENCHMARK_FINAL_PATH, benchmark_report)
        _write_json_artifact(MEM_BREAKDOWN_PATH, mem_breakdown_payload)
        _write_json_artifact(KERNEL_BREAKDOWN_ROOT_PATH, kernel_data)
        _write_json_artifact(KERNEL_BREAKDOWN_COMPAT_PATH, compat_kernel_payload)
    else:
        print("\n[3/5] Kernel profiling SKIPPED (use --mode profile/full)", flush=True)
        # Try loading from existing kernel_breakdown.json
        kern_path = KERNEL_BREAKDOWN_ROOT_PATH
        if kern_path.exists():
            print(f"       → Loading cached data from {kern_path.name}")
            cached = json.loads(kern_path.read_text())
            kernel_data = {}
            for m in ("bf16", "fp8"):
                if m in cached:
                    kernel_data[m] = cached[m]

    # ── Precision audit (if full mode) ──
    precision_audit = None
    if mode == "full":
        print("\n[4/5] Precision audit (subprocess-isolated) ...", flush=True)
        precision_audit = run_precision_audit_isolated(gpu=gpu, seeds=precision_seeds)
        rrmse = precision_audit.get("rrmse_pct", {})
        print(f"       RRMSE: output={rrmse.get('output', '?')}%, "
               f"dx={rrmse.get('dx', '?')}%, "
               f"dw1={rrmse.get('dw1', '?')}%, "
               f"dw2={rrmse.get('dw2', '?')}%")
    else:
        print("\n[4/5] Precision audit SKIPPED (use --mode full)", flush=True)

    # ── Assemble and write manifest ──
    print("\n[5/5] Assembling manifest ...", flush=True)
    manifest = _serialize_manifest(
        bf16_manifest, fp8_manifest, kernel_data, precision_audit,
        benchmark_summary=benchmark_summary, metadata=metadata
    )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    size_kb = MANIFEST_PATH.stat().st_size / 1024
    print(f"  → {MANIFEST_PATH} ({size_kb:.1f} KB)")

    # Auto-generate scoreboard.json from the manifest
    scoreboard_path = ROOT / "tools" / "scoreboard.py"
    if scoreboard_path.exists():
        try:
            print("  Generating scoreboard.json ...")
            spec = importlib.util.spec_from_file_location("scoreboard", scoreboard_path)
            sb_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sb_mod)
            scoreboard = sb_mod.build_scoreboard(manifest)
            scoreboard_out = ROOT / "scoreboard.json"
            scoreboard_out.write_text(json.dumps(scoreboard, indent=2, ensure_ascii=False))
            print("  → scoreboard.json generated")
        except Exception as e:
            print(f"  [warn] scoreboard generation failed: {e}")

    print("=" * 60)
    print("Done. Visualization can now consume manifest.json:")
    print("  python -m visualization")
    print("=" * 60)

    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# pad-audit: route-level padding precision / performance / memory audit
# ═══════════════════════════════════════════════════════════════════════════════
#
# Methodology
# -----------
# Compare T_aligned (e.g. 8192, all segments 128-aligned by construction) vs
# T_unaligned (e.g. 8193, at least 1 expert gets a +1 token, triggers
# route-level padding).  Both use moe_TC_softmax_topk_layer with:
#   - FP8 enabled, NO token rounding (rounding is the other strategy)
#   - Identical E, H, I, K, random seed, model weights
#   - Only T differs by +1
#
# Because the router produces different token counts from T vs T+1, routing
# is NOT bit-identical.  We therefore measure:
#   1. Precision: FP8(padded) vs BF16(same-T) — same routing, isolates padding error
#   2. Performance: nsys GPU-projection µs/iter for aligned vs padded
#   3. Memory: peak forward/backward allocated MiB
#
# For the precision test specifically: we run BF16 and FP8 at the SAME T
# (the unaligned one) through the SAME moe_TC_softmax_topk_layer call path.
# BF16 ignores padding (fp8 not enabled), FP8 pads.  Routing is bit-identical
# because router runs before the FP8 branch.  Any diff = FP8 quant error +
# padding error.  Separately, we run the aligned T to measure pure FP8 quant
# error.  Delta isolates the padding contribution.
#
# E=32 three-way comparison (the real benchmark):
#   - BF16 raw: gold standard, no rounding, no padding
#   - FP8 + padding: same raw routing as BF16, pads metadata for 128-alignment
#   - FP8 + rounding: modifies routing (token counts rounded to 128 multiples)
# BF16 raw is the truth.  RRMSE(fp8_pad, bf16) = pure FP8 quant error.
# RRMSE(fp8_round, bf16) = FP8 quant error + routing perturbation.

_PAD_AUDIT_PRECISION_TEMPLATE = textwrap.dedent(r'''
import gc, json, os, sys, torch
import torch.nn.functional as F_torch
sys.path.insert(0, "{root}")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ["USE_QUACK_GEMM"] = "1"

mode = "{mode}"   # "bf16", "bf16_round", "fp8_pad", "fp8_round"
if mode.startswith("fp8"):
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
import sonicmoe.functional as functional
from sonicmoe.functional import moe_TC_softmax_topk_layer, count_cumsum, moe_general_routing_inputs
from sonicmoe.functional.utils import enable_fp8

T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
seed = {seed}
Mtile = 128

torch.manual_seed(seed)
device = torch.device("cuda:0")
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
torch.cuda.manual_seed(seed)
x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)

functional.clear_all_fp8_weight_caches()
functional._ALIGNMENT_ASSUMED = False
functional._ALIGNMENT_STREAK = 0

use_fp8 = mode.startswith("fp8")
if use_fp8:
    moe.refresh_fp8_shadow_weights()
    moe.stash_bf16_to_cpu()

w1_p = moe.c_fc.weight.permute(1, 2, 0)
w2_p = moe.c_proj.weight.permute(1, 2, 0)

x_run = x.detach().clone().requires_grad_(True)

if mode in ("fp8_round", "bf16_round"):
    # Token rounding: pre-compute 128-aligned routing
    with torch.no_grad():
        rl = F_torch.linear(x_run, moe.router.weight)
        sc = F_torch.softmax(rl, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        tv, ti = sc.topk(K, dim=-1)
        tv /= tv.sum(dim=-1, keepdim=True)
        sc.scatter_(-1, ti, tv)
        cb = sc.detach().clone() - 1
        cb.scatter_(1, ti, tv)
        si = cb.argsort(dim=0, descending=True).int()
        ef = count_cumsum(ti.view(-1), E, do_cumsum=True)[0]
        efr = (torch.ceil(ef / Mtile) * Mtile).int()
        mk = torch.arange(T, device=device, dtype=torch.int32)[:, None].expand(-1, E) < efr[None, :]
        tok_idx = si[mk]
        exp_idx = torch.arange(E, device=device, dtype=torch.int32)[None, :].expand(T, -1)[mk]
        od = tok_idx.argsort().int()
        tok_idx = tok_idx[od]; exp_idx = exp_idx[od]
        rsc = sc[tok_idx, exp_idx].contiguous()
    functional._ALIGNMENT_ASSUMED = True
    with enable_fp8(use_fp8):
        o, ef_out = moe_general_routing_inputs(
            x_run, rsc, tok_idx, exp_idx, w1_p, None, w2_p, None,
            E, moe.stream_id, ActivationType.SWIGLU, False)
else:
    # bf16 or fp8_pad: both use moe_TC_softmax_topk_layer (raw routing)
    with enable_fp8(use_fp8):
        o, rl, ef_out = moe_TC_softmax_topk_layer(
            x_run, moe.router.weight, w1_p, None, w2_p, None,
            K, moe.stream_id, ActivationType.SWIGLU, False)

grad_out = torch.randn(T, H, device=device, dtype=torch.bfloat16,
                       generator=torch.Generator(device=device).manual_seed(seed + 1))
o.backward(grad_out)

result = {{
    "mode": mode, "T": T, "E": E, "seed": seed,
    "o_norm": o.float().norm().item(),
    "o_absmax": o.float().abs().max().item(),
}}
torch.save({{
    "o": o.detach().cpu(),
    "dw1": moe.c_fc.weight.grad.detach().cpu() if moe.c_fc.weight.grad is not None else None,
    "dw2": moe.c_proj.weight.grad.detach().cpu() if moe.c_proj.weight.grad is not None else None,
    "dx": x_run.grad.detach().cpu() if x_run.grad is not None else None,
}}, "{tensor_path}")
print("__PAD_PRECISION__" + json.dumps(result))
''')


def _pad_audit_precision_run(
    root: str, gpu: int, T: int, H: int, I: int, E: int, K: int,
    mode: str, seed: int, tensor_path: str,
) -> dict:
    """Run one subprocess for pad-audit precision."""
    python_bin = _resolve_python_bin()
    script = _PAD_AUDIT_PRECISION_TEMPLATE.format(
        root=root, gpu=gpu, mode=mode, T=T, H=H, I=I, E=E, K=K,
        seed=seed, tensor_path=tensor_path,
    )
    env = _subprocess_env_for_gpu(gpu)
    if mode == "fp8":
        env["SONIC_MOE_FP8_MODE"] = "perf"
    proc = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True, text=True, timeout=600, env=env, cwd=root,
    )
    for line in proc.stdout.split("\n"):
        if line.startswith("__PAD_PRECISION__"):
            return json.loads(line[len("__PAD_PRECISION__"):])
    return {"error": proc.stderr[-500:] if proc.stderr else "no output",
            "stdout": proc.stdout[-300:]}


def _rrmse_tensors(a, b):
    """Compute RRMSE between two tensors (float64 for precision)."""
    a_f, b_f = a.double(), b.double()
    diff_norm = (a_f - b_f).norm().item()
    ref_norm = b_f.norm().item()
    return diff_norm / ref_norm if ref_norm > 1e-12 else float("inf")


def _cosine_tensors(a, b):
    """Cosine similarity (float64)."""
    a_f, b_f = a.double().flatten(), b.double().flatten()
    dot = (a_f * b_f).sum().item()
    return dot / (a_f.norm().item() * b_f.norm().item() + 1e-30)


def _maxdiff_tensors(a, b):
    """Max absolute difference."""
    return (a.float() - b.float()).abs().max().item()


def run_pad_audit(
    gpu: int = 0,
    nsys_warmup: int = DEFAULT_NSYS_WARMUP,
    nsys_iters: int = DEFAULT_NSYS_ITERS,
    seeds: list[int] | None = None,
    T: int = 8192,
    H: int = 3072,
    I: int = 1536,
    E: int = 32,
    K: int = 8,
) -> dict[str, Any]:
    """Route-level padding audit: precision + performance + memory.

    Three-way comparison at E=32 (non-aligned by nature):
      - BF16 raw: gold standard (no rounding, no padding)
      - FP8 + padding: same raw routing, pads metadata for 128-alignment
      - FP8 + rounding: modifies routing (token counts → multiples of 128)

    RRMSE(fp8_pad, bf16) = pure FP8 quantization error.
    RRMSE(fp8_round, bf16) = FP8 quant error + routing perturbation.
    """
    import torch

    root_str = str(ROOT)
    seeds = seeds or [42, 123, 777]
    all_modes = ["bf16", "bf16_round", "fp8_pad", "fp8_round"]

    results: dict[str, Any] = {
        "shape": {"T": T, "H": H, "I": I, "E": E, "K": K},
        "precision": {},
        "performance": {},
        "memory": {},
    }

    # ── 1. PRECISION ──────────────────────────────────────────────────────
    print(f"\n  [1/3] Precision: 4-way vs BF16 raw gold", flush=True)
    print(f"        T={T}, E={E}, K={K}, seeds={seeds}", flush=True)

    per_seed_results = []
    for seed in seeds:
        seed_result = {"seed": seed}
        tensors = {}
        for m in all_modes:
            tpath = str(Path(tempfile.mkdtemp()) / f"pad_audit_{m}_s{seed}.pt")
            info = _pad_audit_precision_run(
                root_str, gpu, T, H, I, E, K, m, seed, tpath,
            )
            if "error" in info:
                print(f"    [FAIL] {m}/seed={seed}: {info['error'][:120]}", flush=True)
                tensors[m] = None
                continue
            try:
                tensors[m] = torch.load(tpath, weights_only=True, map_location="cpu")
            except Exception as e:
                print(f"    [FAIL] {m}/seed={seed}: tensor load: {e}", flush=True)
                tensors[m] = None
            finally:
                try:
                    os.unlink(tpath)
                except OSError:
                    pass

        bf16_t = tensors.get("bf16")
        if bf16_t is None:
            per_seed_results.append(seed_result)
            continue

        for m in all_modes[1:]:  # skip "bf16" (it IS the gold)
            fp8_t = tensors.get(m)
            if fp8_t is None:
                seed_result[m] = {"error": "subprocess failed"}
                continue
            cmp = {}
            for key in ("o", "dw1", "dw2", "dx"):
                a, b = fp8_t.get(key), bf16_t.get(key)
                if a is not None and b is not None and a.shape == b.shape:
                    cmp[key] = {
                        "rrmse": round(_rrmse_tensors(a, b), 8),
                        "cosine": round(_cosine_tensors(a, b), 8),
                        "maxdiff": round(_maxdiff_tensors(a, b), 8),
                    }
                else:
                    reason = "None" if (a is None or b is None) else f"shape {a.shape} vs {b.shape}"
                    cmp[key] = {"skip": reason}
            seed_result[m] = cmp
        per_seed_results.append(seed_result)

    results["precision"]["per_seed"] = per_seed_results

    # Aggregate: mean RRMSE across seeds
    for m in all_modes[1:]:
        agg = {}
        for key in ("o", "dw1", "dw2", "dx"):
            vals = [s[m][key]["rrmse"] for s in per_seed_results
                    if m in s and isinstance(s[m], dict)
                    and key in s[m] and "rrmse" in s[m][key]]
            if vals:
                agg[key] = {
                    "mean_rrmse": round(sum(vals) / len(vals), 8),
                    "max_rrmse": round(max(vals), 8),
                    "n_seeds": len(vals),
                }
        results["precision"][f"{m}_agg"] = agg

    # Print precision summary
    print(f"\n  Precision vs BF16 raw gold (RRMSE, {len(seeds)} seeds)", flush=True)
    col_modes = all_modes[1:]  # bf16_round, fp8_pad, fp8_round
    header = f"  {'Tensor':<8s}" + "".join(f" {m:>12s}" for m in col_modes)
    print(header, flush=True)
    print(f"  {'-'*8}" + "".join(f" {'-'*12}" for _ in col_modes), flush=True)
    for key in ("o", "dw1", "dw2", "dx"):
        row = f"  {key:<8s}"
        for m in col_modes:
            rr = results["precision"].get(f"{m}_agg", {}).get(key, {}).get("mean_rrmse")
            row += f" {rr:>11.6f} " if rr is not None else f" {'N/A':>12s}"
        print(row, flush=True)

    # ── 2. PERFORMANCE (nsys GPU-projection) ──────────────────────────────
    # nsys uses the existing bf16/fp8 modes.  For E>8, fp8 uses rounding.
    # fp8_pad goes through moe_TC_softmax_topk_layer which is the "fp8" nsys mode
    # when use_token_rounding=False.  But the existing template forces rounding
    # for E>8.  We need separate runs.
    print(f"\n  [2/3] nsys performance (warmup={nsys_warmup}, iters={nsys_iters}) ...", flush=True)
    shape = {"T": T, "H": H, "I": I, "E": E, "K": K}

    # Run bf16 + fp8_rounding via existing run_nsys_profile (which rounds for E>8)
    nsys_data = run_nsys_profile(
        gpu=gpu, warmup=nsys_warmup, iters=nsys_iters, shapes=[shape],
    )
    shape_key = f"T{T}_I{I}_E{E}K{K}"
    shape_nsys = nsys_data.get("shapes", {}).get(shape_key, {})
    results["performance"]["bf16_raw_us"] = shape_nsys.get("bf16", {}).get("per_iter_us")
    results["performance"]["fp8_round_us"] = shape_nsys.get("fp8", {}).get("per_iter_us")

    # Run fp8_pad separately: force use_token_rounding=False in nsys template
    # by using a custom script that calls moe_TC_softmax_topk_layer directly
    print(f"  nsys profiling [fp8_pad/{shape_key}] ({nsys_warmup}w+{nsys_iters}m) ...", flush=True)
    _FP8_PAD_NSYS_SCRIPT = textwrap.dedent(r'''
import os, sys, gc, torch
sys.path.insert(0, "{root}")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
import sonicmoe.functional as functional
from sonicmoe.functional import moe_TC_softmax_topk_layer
from sonicmoe.functional.utils import enable_fp8

T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
torch.manual_seed(42)
device = torch.device("cuda:0")
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_()

functional.clear_all_fp8_weight_caches()
functional._ALIGNMENT_ASSUMED = False
functional._ALIGNMENT_STREAK = 0
moe.refresh_fp8_shadow_weights()
moe.stash_bf16_to_cpu()
w1_p = moe.c_fc.weight.permute(1, 2, 0)
w2_p = moe.c_proj.weight.permute(1, 2, 0)

def run_iter():
    with enable_fp8(True):
        o, _, _ = moe_TC_softmax_topk_layer(
            x, moe.router.weight, w1_p, None, w2_p, None,
            K, moe.stream_id, ActivationType.SWIGLU, False)
    return o

for _ in range({warmup}):
    out = run_iter()
    out.sum().backward()
    moe.zero_grad(set_to_none=True)
    if x.grad is not None: x.grad = None
torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()

torch.cuda.cudart().cudaProfilerStart()
for _ in range({iters}):
    out = run_iter()
    out.sum().backward()
    moe.zero_grad(set_to_none=True)
    if x.grad is not None: x.grad = None
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("NSYS_DONE")
''').format(root=str(ROOT), gpu=str(gpu), warmup=nsys_warmup, iters=nsys_iters, **shape)

    python_bin = _resolve_python_bin()
    nsys_output_dir = Path(
        os.environ.get("SONIC_MOE_NSYS_OUTPUT_DIR", str(ROOT / "reports" / "nsys"))
    )
    nsys_output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%H%M%S")
    rep_name = f"fp8pad_{shape_key}_{ts}"
    rep_path = str(nsys_output_dir / rep_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="nsys_fp8pad_") as f:
        f.write(_FP8_PAD_NSYS_SCRIPT)
        script_path = f.name

    try:
        sub_env = _subprocess_env_for_gpu(gpu)
        cmd = [
            "nsys", "profile",
            "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            f"--output={rep_path}", "--export=sqlite",
            "--force-overwrite=true",
            python_bin, script_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=sub_env)
        if proc.returncode == 0:
            db_file = f"{rep_path}.sqlite"
            if os.path.exists(db_file):
                parsed = _nsys_parse_sqlite(db_file, nsys_iters)
                results["performance"]["fp8_pad_us"] = parsed.get("per_iter_us")
                print(f"    fp8_pad/{shape_key}: {parsed.get('per_iter_us', '?')} µs/iter "
                      f"({parsed.get('num_kernels', '?')} kernels)", flush=True)
            else:
                print(f"    [WARN] sqlite missing for fp8_pad", flush=True)
        else:
            print(f"    [WARN] nsys failed for fp8_pad: {proc.stderr[-200:]}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"    [WARN] nsys timed out for fp8_pad", flush=True)
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    # Print performance summary
    bf16_us = results["performance"].get("bf16_raw_us")
    pad_us = results["performance"].get("fp8_pad_us")
    rnd_us = results["performance"].get("fp8_round_us")
    print(f"\n  Performance: nsys GPU-projection µs/iter", flush=True)
    print(f"  {'Mode':<16s} {'µs/iter':>10s} {'vs BF16':>10s}", flush=True)
    print(f"  {'-'*16} {'-'*10} {'-'*10}", flush=True)
    if bf16_us: print(f"  {'BF16 raw':<16s} {bf16_us:>9.0f}  {'1.000×':>10s}", flush=True)
    if pad_us and bf16_us: print(f"  {'FP8 + padding':<16s} {pad_us:>9.0f}  {bf16_us/pad_us:>9.3f}×", flush=True)
    if rnd_us and bf16_us: print(f"  {'FP8 + rounding':<16s} {rnd_us:>9.0f}  {bf16_us/rnd_us:>9.3f}×", flush=True)
    if pad_us and rnd_us:
        delta = (pad_us - rnd_us) / rnd_us * 100
        print(f"\n  FP8 padding vs rounding: {pad_us:.0f} vs {rnd_us:.0f} µs → {delta:+.1f}%", flush=True)

    # ── 3. MEMORY ─────────────────────────────────────────────────────────
    print(f"\n  [3/3] Memory measurement ...", flush=True)
    # bf16_raw and fp8_round via existing _run_memory_measure (E>8 fp8 uses rounding)
    for m in ("bf16", "fp8"):
        mem = _run_memory_measure(m, shape, gpu, warmup=3)
        label = "fp8_round" if m == "fp8" else "bf16_raw"
        results["memory"][label] = mem
        if "error" not in mem:
            print(f"    {label}: fwd={mem['peak_fwd_mib']:.0f} bwd={mem['peak_bwd_mib']:.0f} MiB", flush=True)

    # bf16_round: BF16 with token-rounded routing (no FP8)
    # Reuse the _MEM_MEASURE_SCRIPT but it routes with rounding for E>8 + fp8
    # We need a custom script for bf16_round that does rounding but no FP8.
    # For simplicity, bf16_round memory ≈ bf16_raw (rounding doesn't change tensor sizes much)
    # Just note it in results.
    results["memory"]["bf16_round"] = results["memory"].get("bf16_raw", {}).copy()
    results["memory"]["bf16_round"]["note"] = "≈ bf16_raw (rounding changes routing, not tensor sizes)"

    # fp8_pad memory via custom subprocess
    _FP8_PAD_MEM_SCRIPT = textwrap.dedent(r'''
import gc, json, os, sys, torch
sys.path.insert(0, "{root}")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
from sonicmoe import MoE
from sonicmoe.enums import ActivationType
import sonicmoe.functional as functional
from sonicmoe.functional import moe_TC_softmax_topk_layer
from sonicmoe.functional.utils import enable_fp8

T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
torch.manual_seed(42); device = torch.device("cuda:0")
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)).detach().requires_grad_()
functional.clear_all_fp8_weight_caches()
functional._ALIGNMENT_ASSUMED = False; functional._ALIGNMENT_STREAK = 0
moe.refresh_fp8_shadow_weights(); moe.stash_bf16_to_cpu()
w1_p = moe.c_fc.weight.permute(1, 2, 0)
w2_p = moe.c_proj.weight.permute(1, 2, 0)

def run_iter():
    xw = x.detach().clone().requires_grad_(True)
    with enable_fp8(True):
        o, _, _ = moe_TC_softmax_topk_layer(xw, moe.router.weight, w1_p, None, w2_p, None,
            K, moe.stream_id, ActivationType.SWIGLU, False)
    return xw, o

for _ in range(3):
    xw, o = run_iter(); o.sum().backward()
    moe.zero_grad(set_to_none=True); del xw, o
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

MiB = 1048576
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize()
base = torch.cuda.memory_allocated(device) / MiB
xw, o = run_iter(); torch.cuda.synchronize()
peak_fwd = torch.cuda.max_memory_allocated(device) / MiB
torch.cuda.reset_peak_memory_stats(device)
o.sum().backward(); torch.cuda.synchronize()
peak_bwd = torch.cuda.max_memory_allocated(device) / MiB
print("__MEM_JSON__" + json.dumps({{"mode": "fp8_pad", "base_mib": round(base, 1),
       "peak_fwd_mib": round(peak_fwd, 1), "peak_bwd_mib": round(peak_bwd, 1)}}))
''').format(root=str(ROOT), gpu=str(gpu), **shape)

    proc = subprocess.run(
        [python_bin, "-c", _FP8_PAD_MEM_SCRIPT],
        capture_output=True, text=True, timeout=600,
        env=_subprocess_env_for_gpu(gpu), cwd=str(ROOT),
    )
    for line in proc.stdout.split("\n"):
        if line.startswith("__MEM_JSON__"):
            mem = json.loads(line[len("__MEM_JSON__"):])
            results["memory"]["fp8_pad"] = mem
            print(f"    fp8_pad: fwd={mem['peak_fwd_mib']:.0f} bwd={mem['peak_bwd_mib']:.0f} MiB", flush=True)
            break
    else:
        results["memory"]["fp8_pad"] = {"error": proc.stderr[-200:] if proc.stderr else "no output"}
        print(f"    [FAIL] fp8_pad memory: {proc.stderr[-100:]}", flush=True)

    # Print memory summary table
    print(f"\n  Memory (peak MiB):", flush=True)
    print(f"  {'Mode':<16s} {'Fwd':>8s} {'Bwd':>8s}", flush=True)
    print(f"  {'-'*16} {'-'*8} {'-'*8}", flush=True)
    for label in ("bf16_raw", "bf16_round", "fp8_pad", "fp8_round"):
        mem = results["memory"].get(label, {})
        fwd = mem.get("peak_fwd_mib", 0)
        bwd = mem.get("peak_bwd_mib", 0)
        if fwd: print(f"  {label:<16s} {fwd:>7.0f}  {bwd:>7.0f}", flush=True)

    # ── Theoretical memory breakdown ──────────────────────────────────────
    # Calculate exact delta from padding: which tensors grow and by how much
    MiB = 1048576
    TK_raw = T * K
    # Compute N_pad from routing (approximate: ceil each expert segment to 128)
    n_experts = E
    per_expert_avg = TK_raw / n_experts
    remainder = per_expert_avg % 128
    # Worst case: every expert has remainder > 0
    n_pad_per_expert = (128 - remainder) % 128 if remainder > 0 else 0
    N_pad_approx = int(n_pad_per_expert * n_experts)
    TK_padded = TK_raw + N_pad_approx

    print(f"\n  Theoretical padding memory overhead (E={E}, T={T}, K={K}):", flush=True)
    print(f"  TK_raw={TK_raw}, TK_padded≈{TK_padded}, N_pad≈{N_pad_approx}", flush=True)
    print(f"  {'Tensor':<30s} {'Dtype':>6s} {'Cols':>6s} {'ΔRows':>6s} {'ΔMiB':>8s}", flush=True)
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*8}", flush=True)
    overhead_items = [
        # Forward intermediates (padded_total rows instead of TK rows)
        ("z (pre-activation)", "bf16", 2 * I, N_pad_approx),
        ("y1 (post-SwiGLU)", "bf16", I, N_pad_approx),
        ("z_fp8 (saved for bwd)", "fp8", 2 * I, N_pad_approx),
        ("z_scales (saved for bwd)", "e8m0", 2 * I // 32, N_pad_approx),
        # Backward intermediates
        ("dz (activation grad)", "bf16", 2 * I, N_pad_approx),
        ("y1s (score-weighted y1)", "bf16", I, N_pad_approx),
        ("dz_fp8 (for UpBwd)", "fp8", 2 * I, N_pad_approx),
        # Routing metadata (small)
        ("padded_x_gather_idx", "int32", 1, N_pad_approx),
        ("padded_s_scatter_idx", "int32", 1, N_pad_approx),
        ("padded_scores", "fp32", 1, N_pad_approx),
    ]
    total_overhead = 0.0
    bytes_per = {"bf16": 2, "fp8": 1, "e8m0": 1, "int32": 4, "fp32": 4}
    for name, dtype, cols, drows in overhead_items:
        delta_bytes = drows * cols * bytes_per[dtype]
        delta_mib = delta_bytes / MiB
        total_overhead += delta_mib
        print(f"  {name:<30s} {dtype:>6s} {cols:>6d} {drows:>6d} {delta_mib:>+7.2f}", flush=True)
    print(f"  {'TOTAL THEORETICAL':<30s} {'':>6s} {'':>6s} {'':>6s} {total_overhead:>+7.2f}", flush=True)
    results["memory"]["theoretical_overhead_mib"] = round(total_overhead, 2)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# grid: parallel multi-GPU nsys profiling for the full shape grid
# ═══════════════════════════════════════════════════════════════════════════════

def _grid_generate_shapes() -> list[dict[str, int]]:
    """Generate the 3T × 3E × 3I = 27 shape grid."""
    shapes = []
    for T in GRID_T:
        for E in GRID_E:
            for I in GRID_I:
                shapes.append({"T": T, "H": GRID_H, "I": I, "E": E, "K": GRID_K})
    return shapes


def _grid_estimate_cost(shape: dict) -> float:
    """Heuristic cost for load-balancing: proportional to FLOPs."""
    T, E, I = shape["T"], shape["E"], shape["I"]
    return T * E * I / 1e9


def run_grid(
    num_gpus: int = 8,
    nsys_warmup: int = DEFAULT_NSYS_WARMUP,
    nsys_iters: int = DEFAULT_NSYS_ITERS,
    grid_output: str | None = None,
) -> dict[str, Any]:
    """Run the full 27-shape grid benchmark across multiple GPUs in parallel.

    Each GPU runs ``--mode nsys`` on its assigned shapes sequentially.
    nsys sessions on different GPUs do NOT collide because
    ``CUDA_VISIBLE_DEVICES`` pins each subprocess to one physical GPU.

    Returns merged dict with all shapes.
    """
    python_bin = _resolve_python_bin()
    shapes = _grid_generate_shapes()

    # ── Load-balanced assignment (greedy LPT algorithm) ──
    costs = [(s, _grid_estimate_cost(s)) for s in shapes]
    costs.sort(key=lambda x: x[1], reverse=True)
    gpu_loads = [0.0] * num_gpus
    gpu_shapes: list[list[dict]] = [[] for _ in range(num_gpus)]
    for shape, cost in costs:
        lightest = min(range(num_gpus), key=lambda g: gpu_loads[g])
        gpu_shapes[lightest].append(shape)
        gpu_loads[lightest] += cost

    # ── Output directory ──
    out_dir = Path(grid_output) if grid_output else (ROOT / "reports" / "grid_session53")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  SonicMoE Grid Benchmark — 27-shape parallel nsys profiling")
    print("=" * 80)
    for g in range(num_gpus):
        tags = [f"T{s['T']}_E{s['E']}_I{s['I']}" for s in gpu_shapes[g]]
        print(f"  GPU {g}: {len(gpu_shapes[g])} shapes (cost={gpu_loads[g]:.1f})  {tags}")
    print()

    # ── Launch parallel subprocesses ──
    procs: list[tuple[int, subprocess.Popen, Path, Any]] = []
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for g in range(num_gpus):
        if not gpu_shapes[g]:
            continue
        # Build --nsys-shapes as a single space-separated arg (nargs="+")
        shape_strs = [f"{s['T']},{s['H']},{s['I']},{s['E']},{s['K']}" for s in gpu_shapes[g]]

        gpu_json = out_dir / f"gpu{g}.json"
        cmd = [
            python_bin, str(ROOT / "tools" / "introspect.py"),
            "--mode", "nsys",
            "--gpu", "0",  # always 0: CUDA_VISIBLE_DEVICES pins the physical GPU
            "--nsys-warmup", str(nsys_warmup),
            "--nsys-iters", str(nsys_iters),
            "--nsys-output", str(gpu_json),
            "--nsys-shapes", *shape_strs,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        env["PYTHONPATH"] = str(ROOT) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

        log_file = open(log_dir / f"gpu{g}.log", "w")
        print(f"  Launching GPU {g} ({len(gpu_shapes[g])} shapes) → {gpu_json.name}", flush=True)
        p = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT,
            env=env, cwd=str(ROOT),
        )
        procs.append((g, p, gpu_json, log_file))

    # ── Wait for all ──
    print(f"\n  Waiting for {len(procs)} GPU workers ...", flush=True)
    failed = []
    for g, p, gpu_json, log_file in procs:
        rc = p.wait()
        log_file.close()
        if rc != 0:
            print(f"  [WARN] GPU {g} exited with code {rc}", flush=True)
            failed.append(g)
        else:
            sz = gpu_json.stat().st_size / 1024 if gpu_json.exists() else 0
            print(f"  GPU {g} done → {gpu_json.name} ({sz:.1f} KB)", flush=True)

    # ── Merge all per-GPU JSONs ──
    print(f"\n  Merging results ...", flush=True)
    merged: dict[str, Any] = {
        "metadata": {
            "session": 53,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_gpus": num_gpus,
            "nsys_warmup": nsys_warmup,
            "nsys_iters": nsys_iters,
            "grid_T": GRID_T,
            "grid_E": GRID_E,
            "grid_I": GRID_I,
            "failed_gpus": failed,
        },
        "shapes": {},
    }
    for g in range(num_gpus):
        gpu_json = out_dir / f"gpu{g}.json"
        if not gpu_json.exists():
            continue
        try:
            gpu_data = json.loads(gpu_json.read_text())
            for shape_key, shape_res in gpu_data.get("shapes", {}).items():
                # Validate: shape must have both bf16 and fp8 nsys data
                has_bf16 = "per_iter_us" in shape_res.get("bf16", {})
                has_fp8 = "per_iter_us" in shape_res.get("fp8", {})
                shape_res["_source_gpu"] = g
                shape_res["_complete"] = has_bf16 and has_fp8
                merged["shapes"][shape_key] = shape_res
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [WARN] Failed to parse gpu{g}.json: {e}", flush=True)

    complete = sum(1 for v in merged["shapes"].values() if v.get("_complete"))
    total = len(merged["shapes"])
    merged["metadata"]["shapes_complete"] = complete
    merged["metadata"]["shapes_total"] = total

    # Write merged output
    merged_path = out_dir / "session53_grid_full.json"
    merged_path.write_text(json.dumps(merged, indent=2, default=str))
    print(f"\n  → {merged_path} ({merged_path.stat().st_size / 1024:.1f} KB)")

    # ── Summary table ──
    print(f"\n{'='*100}")
    print(f"  Grid Summary: {complete}/{total} shapes complete")
    print(f"{'='*100}")
    print(f"  {'Shape':<28s} {'BF16 µs':>8s} {'FP8 µs':>8s} {'Speed':>7s} "
          f"{'BF16 Bwd':>9s} {'FP8 Bwd':>9s} {'MemΔ%':>7s} {'Src':>4s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*7} {'-'*9} {'-'*9} {'-'*7} {'-'*4}")
    for sk in sorted(merged["shapes"]):
        sr = merged["shapes"][sk]
        bf16_us = sr.get("bf16", {}).get("per_iter_us", 0)
        fp8_us = sr.get("fp8", {}).get("per_iter_us", 0)
        speedup = sr.get("speedup", 0)
        bf16_bwd = sr.get("memory_bf16", {}).get("peak_bwd_mib", 0)
        fp8_bwd = sr.get("memory_fp8", {}).get("peak_bwd_mib", 0)
        mem_pct = round((fp8_bwd - bf16_bwd) / bf16_bwd * 100, 1) if bf16_bwd else 0
        src = sr.get("_source_gpu", "?")
        ok = "✓" if sr.get("_complete") else "✗"
        print(f"  {sk:<28s} {bf16_us:>7.0f}  {fp8_us:>7.0f}  {speedup:>6.3f}× "
              f"{bf16_bwd:>8.0f}M {fp8_bwd:>8.0f}M {mem_pct:>+6.1f}% GPU{src} {ok}")
    print(f"{'='*100}")

    if failed:
        print(f"\n  ⚠ Failed GPUs: {failed}. Check {log_dir}/gpu*.log for details.")

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# compile-session53: aggregate benchmark artifacts (no GPU needed)
# ═══════════════════════════════════════════════════════════════════════════════

def run_compile_session53() -> dict:
    """Aggregate all Session 53 benchmark JSONs into a single summary.

    No GPU required — purely reads existing data files and writes
    ``reports/session53_summary.json``.
    """
    root = Path(__file__).resolve().parent.parent
    reports = root / "reports"

    summary: dict[str, Any] = {"session": 53, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ── E2E benchmark (authoritative timing + memory) ──
    e2e_path = reports / "e2e_bench_session52.json"
    e2e = json.loads(e2e_path.read_text())
    shapes = {}
    for row in e2e:
        label = row["shape"]
        shapes.setdefault(label, {})
        shapes[label][row["mode"]] = row
    # Compute speedups
    e2e_summary = {}
    for label, modes in shapes.items():
        bf = modes["bf16"]
        fp = modes["fp8"]
        e2e_summary[label] = {
            "bf16_total_ms": bf["total_ms"],
            "fp8_total_ms":  fp["total_ms"],
            "speedup":       round(bf["total_ms"] / fp["total_ms"], 4),
            "bf16_peak_MiB": bf["peak_MiB"],
            "fp8_peak_MiB":  fp["peak_MiB"],
            "mem_delta_pct":  round((fp["peak_MiB"] - bf["peak_MiB"]) / bf["peak_MiB"] * 100, 2),
        }
    summary["e2e"] = e2e_summary

    # ── Quant kernel benchmark ──
    quant_path = reports / "quant_bench_final.json"
    quant = json.loads(quant_path.read_text())
    summary["quant_kernels"] = {
        f"{q['kernel']}@{q['dim']}": round(q["median"], 1) for q in quant
    }

    # ── Kernel profiler breakdown (I=1536) ──
    kb_path = root / "kernel_breakdown.json"
    kb = json.loads(kb_path.read_text())
    for mode in ("bf16", "fp8"):
        total = kb[mode]["total_cuda_us"]
        top5 = []
        for k in kb[mode]["kernels"][:5]:
            short = k["name"][:60]
            top5.append({"name": short, "us": round(k["avg_cuda_us"], 1)})
        summary[f"kernel_breakdown_{mode}"] = {"total_cuda_us": total, "top5": top5}

    # ── Memory breakdown (I=1536) ──
    mem_path = root / "mem_breakdown.json"
    mem = json.loads(mem_path.read_text())
    summary["memory_lifecycle"] = {
        mode: mem[mode]["checkpoints"] for mode in ("bf16", "fp8")
    }
    if "fp8_stash" in mem:
        stash = mem["fp8_stash"]
        summary["memory_lifecycle"]["fp8_stash"] = {
            "bwd_peak_mean": stash["summary"]["memory_mib"]["bwd_peak"]["mean"],
            "timing_speedup": stash["comparison_vs_bf16"]["timing_speedup"],
            "bwd_peak_delta_pct": stash["comparison_vs_bf16"]["bwd_peak_delta_pct"],
        }

    # ── Wgrad benchmark ──
    wgrad_path = reports / "wgrad_bench.json"
    wgrad = json.loads(wgrad_path.read_text())
    wgrad_summary = {}
    for shape_key, shape_data in wgrad.get("shapes", {}).items():
        modes = shape_data.get("modes", {})
        speedup = shape_data.get("speedup", {})
        wgrad_summary[shape_key] = {
            "bf16_fwd_bwd_us": modes.get("bf16", {}).get("forward_backward_us", {}).get("median_us"),
            "fp8_fwd_bwd_us": modes.get("fp8", {}).get("forward_backward_us", {}).get("median_us"),
            "speedup": speedup.get("fwd_bwd"),
        }
    summary["wgrad_bench"] = wgrad_summary

    # Write
    out_path = reports / "session53_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
# Unified Benchmark Mode — performance × precision × memory
# ═══════════════════════════════════════════════════════════════════════════════

# Default 6-shape benchmark list
BENCHMARK_SHAPES_DEFAULT = [
    {"T": 8192,  "H": 3072, "I": 1536, "E": 8,   "K": 8},   # production-like reference baseline
    {"T": 8192,  "H": 3072, "I": 3072, "E": 8,   "K": 8},   # I scaling
    {"T": 32768, "H": 3072, "I": 1536, "E": 8,   "K": 8},   # T scaling
    {"T": 8192,  "H": 3072, "I": 1536, "E": 32,  "K": 8},   # E>8, route-level padding
    {"T": 32768, "H": 3072, "I": 3072, "E": 32,  "K": 8},   # Peak absolute perf
    {"T": 8192,  "H": 3072, "I": 1536, "E": 128, "K": 8},   # Extreme padding
]


def _benchmark_shape_key(shape: dict) -> str:
    return f"T{shape['T']}_I{shape['I']}_E{shape['E']}K{shape['K']}"


def _benchmark_collect_precision(
    shape: dict, seeds: list[int], gpu: int,
) -> dict[str, Any]:
    """Phase 1: subprocess-isolated precision audit for one shape.

    Returns RRMSE as ratio (0.065 = 6.5%), cosine as float64.
    Per-seed breakdown included for reproducibility.
    """
    import torch

    python_bin = sys.executable
    shape_arg = ",".join(str(shape[k]) for k in ("T", "H", "I", "E", "K"))

    def _collect(mode: str, seed: int, out_path: Path) -> None:
        env = _build_subprocess_env(mode, gpu)
        proc = subprocess.run(
            [python_bin, str(Path(__file__).resolve()),
             "--shape", shape_arg,
             "--_worker-collect", mode,
             "--_worker-seed", str(seed),
             "--_worker-output", str(out_path)],
            capture_output=True, text=True, timeout=600,
            cwd=str(ROOT), env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Precision worker [{mode}, seed={seed}] failed:\n"
                f"{proc.stderr[-2000:] or proc.stdout[-2000:]}")

    def _rrmse_ratio(a, b):
        """RRMSE as ratio (consistent with conftest.py). 0.065 = 6.5%."""
        return float(((a.float() - b.float()).norm()
                      / b.float().norm().clamp(min=1e-8)).item())

    def _cosine(a, b):
        af = a.flatten().double()
        bf = b.flatten().double()
        d = (af.norm() * bf.norm()).clamp(min=1e-12)
        return max(-1.0, min(1.0, float((af * bf).sum().item() / d.item())))

    rrmse_per_seed: dict[str, dict[str, float]] = {}
    cosine_per_seed: dict[str, dict[str, float]] = {}
    rrmse_agg: dict[str, list[float]] = collections.defaultdict(list)
    cosine_agg: dict[str, list[float]] = collections.defaultdict(list)

    with _persistent_tempdir() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for seed in seeds:
            bf16_path = tmpdir_path / f"bf16_s{seed}.pt"
            fp8_path = tmpdir_path / f"fp8_s{seed}.pt"
            _collect("bf16", seed, bf16_path)
            _collect("fp8", seed, fp8_path)
            bf16_out = torch.load(bf16_path, weights_only=False)
            fp8_out = torch.load(fp8_path, weights_only=False)

            seed_rrmse, seed_cos = {}, {}
            for key in ["output", "dx", "dw1", "dw2"]:
                ref = bf16_out.get(key)
                test = fp8_out.get(key)
                if ref is None or test is None:
                    continue
                r = _rrmse_ratio(test, ref)
                c = _cosine(test, ref)
                seed_rrmse[key] = round(r, 4)
                seed_cos[key] = round(c, 6)
                rrmse_agg[key].append(r)
                cosine_agg[key].append(c)
            rrmse_per_seed[str(seed)] = seed_rrmse
            cosine_per_seed[str(seed)] = seed_cos

    # Aggregate: mean across seeds
    rrmse_mean = {k: round(_safe_mean(v), 4) for k, v in rrmse_agg.items()}
    cosine_mean = {k: round(_safe_mean(v), 6) for k, v in cosine_agg.items()}

    return {
        "rrmse": rrmse_mean,
        "cosine": cosine_mean,
        "per_seed": {
            s: {"rrmse": rrmse_per_seed[s], "cosine": cosine_per_seed[s]}
            for s in rrmse_per_seed
        },
    }


def _benchmark_collect_memory(
    shape: dict, gpu: int, warmup: int = 5,
) -> dict[str, dict]:
    """Phase 2: subprocess-isolated memory measurement for one shape."""
    result = {}
    for mode in ("bf16", "fp8"):
        mem = _run_memory_measure(mode, shape, gpu, warmup=warmup)
        result[mode] = mem
    # Compute bwd delta
    bf_bwd = result.get("bf16", {}).get("peak_bwd_mib", 0)
    fp_bwd = result.get("fp8", {}).get("peak_bwd_mib", 0)
    if bf_bwd > 0:
        result["bwd_delta_pct"] = round((fp_bwd - bf_bwd) / bf_bwd * 100, 1)
    return result


def _benchmark_collect_perf(
    shape: dict, gpu: int,
    warmup: int = DEFAULT_NSYS_WARMUP, iters: int = DEFAULT_NSYS_ITERS,
) -> dict[str, Any]:
    """Phase 3: nsys GPU-projection for one shape, returns per_iter_us."""
    nsys_data = run_nsys_profile(gpu=gpu, warmup=warmup, iters=iters, shapes=[shape])
    shape_key = _benchmark_shape_key(shape)
    shape_res = nsys_data.get("shapes", {}).get(shape_key, {})
    bf16_us = shape_res.get("bf16", {}).get("per_iter_us")
    fp8_us = shape_res.get("fp8", {}).get("per_iter_us")
    result: dict[str, Any] = {}
    if bf16_us is not None:
        result["bf16_us"] = round(bf16_us, 1)
    if fp8_us is not None:
        result["fp8_us"] = round(fp8_us, 1)
    if bf16_us and fp8_us and fp8_us > 0:
        result["speedup"] = round(bf16_us / fp8_us, 3)
    # Include kernel budget breakdown if available
    for mode in ("bf16", "fp8"):
        cats = shape_res.get(mode, {}).get("category_summary")
        if cats:
            result[f"{mode}_categories"] = {k: round(v, 1) for k, v in cats.items()}
    return result


def run_benchmark(
    shapes: list[dict[str, int]],
    seeds: list[int],
    num_gpus: int = 1,
    nsys_warmup: int = DEFAULT_NSYS_WARMUP,
    nsys_iters: int = DEFAULT_NSYS_ITERS,
) -> dict[str, Any]:
    """Rigorous 3-layer benchmark: precision × memory × performance.

    All phases use subprocess isolation. Phases run serially to guarantee
    GPU is fully idle between them.  Produces a single authoritative JSON.

    RRMSE convention: ratio (0.065 = 6.5%).
    """
    import torch

    gpu = 0  # benchmark pins to GPU 0 for reproducibility

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    gpu_name = "unknown"
    try:
        device = torch.device(f"cuda:{gpu}")
        gpu_name = torch.cuda.get_device_name(device)
    except Exception:
        pass

    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), text=True, timeout=10,
        ).strip()
    except Exception:
        pass

    branch = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(ROOT), text=True, timeout=10,
        ).strip()
    except Exception:
        pass

    metadata = {
        "timestamp": timestamp,
        "hardware": gpu_name,
        "branch": branch,
        "commit": commit,
        "python": sys.executable,
        "seeds": seeds,
        "nsys_warmup": nsys_warmup,
        "nsys_iters": nsys_iters,
        "rrmse_convention": "ratio (0.065 = 6.5%)",
    }

    results: dict[str, Any] = {"metadata": metadata, "shapes": {}}
    n = len(shapes)

    print("=" * 80)
    print(f"  Unified Benchmark — {n} shapes × 3 phases (precision / memory / perf)")
    print(f"  Seeds: {seeds}  |  nsys: {nsys_warmup}w+{nsys_iters}m  |  GPU: {gpu_name}")
    print("=" * 80)

    for i, shape in enumerate(shapes):
        sk = _benchmark_shape_key(shape)
        print(f"\n  [{i+1}/{n}] Shape: {sk}")
        shape_result: dict[str, Any] = {"shape": shape}

        # ── Phase 1: Precision ──────────────────────────────────────────
        print(f"    Phase 1: Precision ({len(seeds)} seeds) ...", flush=True)
        saved = dict(SHAPE)
        SHAPE.update(shape)
        try:
            prec = _benchmark_collect_precision(shape, seeds, gpu)
            shape_result["precision"] = prec
            rr = prec.get("rrmse", {})
            cs = prec.get("cosine", {})
            print(f"      RRMSE  output={rr.get('output','?')} dx={rr.get('dx','?')} "
                  f"dw1={rr.get('dw1','?')} dw2={rr.get('dw2','?')}")
            print(f"      Cosine output={cs.get('output','?')} dx={cs.get('dx','?')} "
                  f"dw1={cs.get('dw1','?')} dw2={cs.get('dw2','?')}")
        except Exception as ex:
            shape_result["precision"] = {"error": str(ex)}
            print(f"      ERROR: {ex}")
        finally:
            SHAPE.update(saved)

        # ── Phase 2: Memory ─────────────────────────────────────────────
        print(f"    Phase 2: Memory ...", flush=True)
        try:
            mem = _benchmark_collect_memory(shape, gpu)
            shape_result["memory"] = mem
            bf_m = mem.get("bf16", {})
            fp_m = mem.get("fp8", {})
            print(f"      BF16 base={bf_m.get('base_mib','?')} fwd={bf_m.get('peak_fwd_mib','?')} "
                  f"bwd={bf_m.get('peak_bwd_mib','?')} MiB")
            print(f"      FP8  base={fp_m.get('base_mib','?')} fwd={fp_m.get('peak_fwd_mib','?')} "
                  f"bwd={fp_m.get('peak_bwd_mib','?')} MiB")
            d = mem.get("bwd_delta_pct")
            if d is not None:
                print(f"      bwd delta: {d:+.1f}%")
        except Exception as ex:
            shape_result["memory"] = {"error": str(ex)}
            print(f"      ERROR: {ex}")

        # ── Phase 3: Performance (nsys) ─────────────────────────────────
        print(f"    Phase 3: Performance (nsys {nsys_warmup}w+{nsys_iters}m) ...", flush=True)
        try:
            perf = _benchmark_collect_perf(shape, gpu, nsys_warmup, nsys_iters)
            shape_result["performance"] = perf
            print(f"      BF16={perf.get('bf16_us','?')} µs  FP8={perf.get('fp8_us','?')} µs  "
                  f"speedup={perf.get('speedup','?')}×")
        except Exception as ex:
            shape_result["performance"] = {"error": str(ex)}
            print(f"      ERROR: {ex}")

        results["shapes"][sk] = shape_result

    # ── Write unified JSON ──────────────────────────────────────────────
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"benchmark_{branch}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    # ── Print summary table ─────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  BENCHMARK SUMMARY — {out_path.name}")
    print(f"{'='*100}")
    print(f"  {'Shape':<28s} {'BF16µs':>7s} {'FP8µs':>7s} {'Speed':>6s} "
          f"{'BF16bwd':>8s} {'FP8bwd':>8s} {'MemΔ%':>6s} "
          f"{'RRMSE(out)':>10s} {'cos(out)':>9s}")
    print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*9}")

    for sk, sr in results["shapes"].items():
        perf = sr.get("performance", {})
        mem = sr.get("memory", {})
        prec = sr.get("precision", {})
        bf_us = perf.get("bf16_us", 0)
        fp_us = perf.get("fp8_us", 0)
        sp = perf.get("speedup", 0)
        bf_bwd = mem.get("bf16", {}).get("peak_bwd_mib", 0)
        fp_bwd = mem.get("fp8", {}).get("peak_bwd_mib", 0)
        m_pct = mem.get("bwd_delta_pct", 0)
        rr_out = prec.get("rrmse", {}).get("output", 0)
        cs_out = prec.get("cosine", {}).get("output", 0)
        print(f"  {sk:<28s} {bf_us:>7.0f} {fp_us:>7.0f} {sp:>5.3f}× "
              f"{bf_bwd:>7.0f}M {fp_bwd:>7.0f}M {m_pct:>+5.1f}% "
              f"{rr_out:>9.4f}  {cs_out:>8.6f}")

    print(f"{'='*100}")
    print(f"  JSON: {out_path}")
    print(f"  RRMSE convention: ratio (multiply ×100 for percentage)")
    return results


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SonicMoE Introspection Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Modes:
              trace     — isolated tensor lifecycle + memory manifest
              profile   — trace + lightweight kernel timing
              full      — trace + repeated benchmark/profiler + precision audit
              nsys      — nsys GPU-projection profiling (gold standard for perf)
              grid      — parallel 27-shape nsys grid across multiple GPUs
              benchmark — unified 3-phase (precision × memory × perf) JSON

            Example:
              python tools/introspect.py --mode full
              python tools/introspect.py --mode nsys --nsys-shapes 8192,3072,1536,8,8 8192,3072,2048,8,8
              python tools/introspect.py --mode grid --gpu 8
              python tools/introspect.py --mode benchmark --benchmark-shapes 8192,3072,1536,8,8 32768,3072,1536,8,8
        """),
    )
    parser.add_argument(
        "--mode", choices=["trace", "profile", "full", "nsys", "grid", "report", "precision", "quant-bench", "wgrad-bench", "ncu-bench", "wgrad-force", "pad-audit", "compile-session53", "benchmark"],
        default="trace",
        help="Introspection depth (default: trace)",
    )
    parser.add_argument(
        "--shape", type=str, default=None,
        help="Override shape as T,H,I,E,K (e.g. '8192,3072,1536,8,8')",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="CUDA device index to target (default: 0)",
    )
    parser.add_argument(
        "--precision-seeds", type=str, default="42,123,777",
        help="Comma-separated seeds for full-mode precision / repeated benchmarks",
    )
    parser.add_argument(
        "--bench-repeats", type=int, default=DEFAULT_BENCH_REPEATS,
        help="Repeated benchmark count in full mode (default: 3)",
    )
    parser.add_argument(
        "--profile-trials", type=int, default=1,
        help="How many times to repeat the rigorous profiler in full mode (default: 1)",
    )
    parser.add_argument(
        "--nsys-warmup", type=int, default=DEFAULT_NSYS_WARMUP,
        help="nsys mode: warmup iterations before measurement (default: 5)",
    )
    parser.add_argument(
        "--nsys-iters", type=int, default=DEFAULT_NSYS_ITERS,
        help="nsys mode: measured iterations (default: 8)",
    )
    parser.add_argument(
        "--nsys-shapes", nargs="+", type=str, default=None,
        help="nsys mode: shapes as T,H,I,E,K (space-separated, e.g. '8192,3072,1536,8,8 8192,3072,2048,8,8')",
    )
    parser.add_argument(
        "--nsys-output", type=str, default=None,
        help="nsys mode: override output JSON path (avoids race when running parallel GPUs)",
    )
    parser.add_argument(
        "--quant-bench-shapes", nargs="+", type=str, default=None,
        help="quant-bench mode: shapes as TK,H,I2,I1 (space-separated)",
    )
    parser.add_argument(
        "--quant-bench-trials", type=int, default=50,
        help="quant-bench mode: CUDA-event trials per kernel (default: 50)",
    )
    parser.add_argument(
        "--wgrad-bench-shapes", nargs="+", type=str, default=None,
        help="wgrad-bench mode: shapes as T,H,I,E,K (space-separated)",
    )
    parser.add_argument(
        "--wgrad-bench-trials", type=int, default=30,
        help="wgrad-bench mode: timing trials per mode (default: 30)",
    )
    parser.add_argument(
        "--benchmark-shapes", nargs="+", type=str, default=None,
        help="benchmark mode: shapes as T,H,I,E,K (space-separated). "
             "Default: 6 canonical shapes covering E=8/32/128.",
    )
    parser.add_argument(
        "--benchmark-seeds", type=str, default="42,123,777",
        help="benchmark mode: comma-separated seeds for precision audit (default: 42,123,777)",
    )
    parser.add_argument("--_worker-trace", choices=["bf16", "fp8"], help=argparse.SUPPRESS)
    parser.add_argument("--_worker-collect", choices=["bf16", "fp8"], help=argparse.SUPPRESS)
    parser.add_argument("--_worker-seed", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.shape:
        parts = [int(x) for x in args.shape.split(",")]
        assert len(parts) == 5, f"Expected T,H,I,E,K but got {len(parts)} values"
        SHAPE["T"], SHAPE["H"], SHAPE["I"], SHAPE["E"], SHAPE["K"] = parts

    if args._worker_trace:
        payload = _run_trace_worker(args._worker_trace)
        print("__TRACE_JSON__" + json.dumps(payload, default=str))
        return

    if args._worker_collect:
        if not args._worker_output:
            raise SystemExit("--_worker-output is required for collect workers")
        _run_collect_worker(args._worker_collect, args._worker_seed, Path(args._worker_output))
        return

    precision_seeds = [int(seed.strip()) for seed in args.precision_seeds.split(",") if seed.strip()]

    # Parse nsys shapes
    nsys_shapes = None
    if args.nsys_shapes:
        nsys_shapes = []
        for s in args.nsys_shapes:
            parts = [int(x) for x in s.split(",")]
            assert len(parts) == 5, f"Expected T,H,I,E,K but got {len(parts)} values in '{s}'"
            nsys_shapes.append(dict(zip(("T", "H", "I", "E", "K"), parts)))

    # Parse quant-bench shapes
    quant_bench_shapes = None
    if args.quant_bench_shapes:
        quant_bench_shapes = []
        for s in args.quant_bench_shapes:
            parts = [int(x) for x in s.split(",")]
            assert len(parts) == 4, f"Expected TK,H,I2,I1 but got {len(parts)} values in '{s}'"
            quant_bench_shapes.append(dict(zip(("TK", "H", "I2", "I1"), parts)))

    # Parse wgrad-bench shapes
    wgrad_bench_shapes = None
    if args.wgrad_bench_shapes:
        wgrad_bench_shapes = []
        for s in args.wgrad_bench_shapes:
            parts = [int(x) for x in s.split(",")]
            assert len(parts) == 5, f"Expected T,H,I,E,K but got {len(parts)} values in '{s}'"
            wgrad_bench_shapes.append(dict(zip(("T", "H", "I", "E", "K"), parts)))

    # ── benchmark mode: unified 3-phase pipeline ──
    if args.mode == "benchmark":
        benchmark_shapes = BENCHMARK_SHAPES_DEFAULT
        if args.benchmark_shapes:
            benchmark_shapes = []
            for s in args.benchmark_shapes:
                parts = [int(x) for x in s.split(",")]
                assert len(parts) == 5, f"Expected T,H,I,E,K but got {len(parts)} in '{s}'"
                benchmark_shapes.append(dict(zip(("T", "H", "I", "E", "K"), parts)))
        benchmark_seeds = [int(x.strip()) for x in args.benchmark_seeds.split(",") if x.strip()]
        run_benchmark(
            shapes=benchmark_shapes,
            seeds=benchmark_seeds,
            num_gpus=args.gpu or 1,
            nsys_warmup=args.nsys_warmup,
            nsys_iters=args.nsys_iters,
        )
        return

    run(
        mode=args.mode,
        gpu=args.gpu,
        precision_seeds=precision_seeds,
        bench_repeats=args.bench_repeats,
        profile_trials=args.profile_trials,
        nsys_warmup=args.nsys_warmup,
        nsys_iters=args.nsys_iters,
        nsys_shapes=nsys_shapes,
        nsys_output=getattr(args, 'nsys_output', None),
        quant_bench_shapes=quant_bench_shapes,
        quant_bench_trials=args.quant_bench_trials,
        wgrad_bench_shapes=wgrad_bench_shapes,
        wgrad_bench_trials=args.wgrad_bench_trials,
    )


if __name__ == "__main__":
    main()
