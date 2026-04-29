"""SonicMoE ↔ ERNIE integration: ``SonicMoEMlpNode`` (FP8 production path).

Drop-in replacement for ERNIE's ``MlpNode``:

    out = node(dispatched_hidden_states, tokens_per_expert,
               dispatched_indices, dispatched_probs)

Gradient contract (matches ``FusionMoePyLayer``):
  * ``dx`` flows back through Paddle autograd (required by
    ``FusedDispatch.backward`` for the reverse A2A).
  * ``dw1 / dw2`` accumulate into per-expert ``main_grad`` buffers via
    a per-instance native-layout view (no per-iter transpose, no globals).
    They do NOT flow through autograd.
  * ``ds`` flows back through Paddle autograd (router-score path).

Design rules (post S74 cleanup — see HANDOFF S74):
  * **No module-level mutable state** in the production path. Each
    ``SonicMoEMlpNode`` instance owns its own weight cache and pending-flush
    flag; pipeline-parallel + multi-layer + interleaved fwd/bwd works out of
    the box because nothing is shared across instances.
  * **No FIFO queues**: deferred wgrad layout conversion is signalled via an
    instance ``_pending_flush`` flag set by ``ctx`` in backward and consumed
    by ``step()``.
  * **No class-variable hacks**: ``topk`` is passed as a regular forward arg.
  * **No deprecated shims**: legacy ``SonicMoEFunc`` (argsort PyLayer),
    ``prepare_sonic_inputs`` (CPU-sync helper), and ``_NATIVE_*`` global
    tombstones have been removed.
  * **No BF16 fallback dead code**: the FP8 wgrad accumulator path always
    returns ``dw1 = dw2 = None`` from ``_UpProjection.backward`` /
    ``_DownProjection.backward`` because ``SonicMoEMlpNode`` always runs
    inside ``enable_fp8(True)``. Backward asserts this contract.
"""

from __future__ import annotations

from typing import Any, Sequence

import paddle
import torch
import triton
import triton.language as tl

from sonicmoe.enums import ActivationType
from sonicmoe.ernie_compat.deepep_metadata import (
    deepep_topk_to_sonic_metadata,
    invalidate_topk_cache,
)
from sonicmoe.functional import (
    _DownProjection,
    _UpProjection,
    _refresh_fp8_config,
    clear_all_fp8_weight_caches,
)
from sonicmoe.functional.utils import enable_fp8
from sonicmoe.quack_utils import precompute_weight_fp8_warmup


# ── PyLayer ctx stub ──────────────────────────────────────────────────────────

class _FakeCtx:
    """Minimal ctx stub for invoking ``_UpProjection`` / ``_DownProjection``
    static-method ``forward`` / ``backward`` directly.

    Supports ``save_for_backward`` / ``saved_tensor`` /
    ``mark_non_differentiable`` / ``set_materialize_grads`` plus arbitrary
    attribute access (the projection helpers stash ``ctx.T``, ``ctx._fp8_cfg``,
    ``ctx._wgrad_w{1,2}_accumulator`` etc).
    """

    def save_for_backward(self, *args): self._saved = args
    def saved_tensor(self): return self._saved
    def mark_non_differentiable(self, *args): pass
    def set_materialize_grads(self, value): pass


# ── Differentiable router-scores reconstruction ──────────────────────────────

@triton.jit
def _scatter_router_grad_kernel(
    GRAD_ptr,    # [TK] grad_output, contiguous
    IDX_ptr,     # [TK] int64 gather indices into [N_total]
    OUT_ptr,     # [N_total] zero-init output
    TK,
    BLOCK: tl.constexpr,
):
    """Plain scatter (no accumulate): out[idx[i]] = grad[i].

    Safe because ``score_src_idx`` is a permutation of distinct positions in
    ``[N_recv * topk]`` (each (token, slot) appears at most once). Eliminates
    the cub::DeviceRadixSort + index_put_with_accumulate cascade triggered by
    PyTorch's generic advanced-indexing backward.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TK
    v = tl.load(GRAD_ptr + offs, mask=mask, other=0.0)
    idx = tl.load(IDX_ptr + offs, mask=mask, other=0)
    tl.store(OUT_ptr + idx, v, mask=mask)


class _GatherRouterScores(torch.autograd.Function):
    """Custom-grad replacement for ``flat_probs[gather_idx]``.

    Forward is the same simple gather (a single Paddle indexing kernel).
    Backward uses a one-shot Triton scatter (no sort, no histogram) instead
    of the generic ``IndexingBackward`` path, which on Paddle dispatches a
    full cub::DeviceRadixSort + scatter + reduction cascade per call.
    """

    @staticmethod
    def forward(ctx, flat_probs: torch.Tensor, gather_idx: torch.Tensor, n_total: int):
        ctx.save_for_backward(gather_idx)
        ctx.n_total = int(n_total)
        ctx.tk = int(gather_idx.shape[0])
        return flat_probs[gather_idx]

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (gather_idx,) = ctx.saved_tensor()
        d = torch.zeros(ctx.n_total, dtype=grad_out.dtype, device=grad_out.device)
        TK = ctx.tk
        if TK > 0:
            BLOCK = 256
            grid = (triton.cdiv(TK, BLOCK),)
            _scatter_router_grad_kernel[grid](
                grad_out.contiguous(), gather_idx, d, TK, BLOCK=BLOCK,
            )
        return d, None, None


@triton.jit
def _build_score_src_idx_kernel(
    INDICES_ptr,  # [N_recv, topk] int32 — dispatched_indices (read-only)
    NAEPT_ptr,    # [N_recv+1] int32
    OUT_ptr,      # [TK] int64
    WORK_ptr,     # [N_recv * TOPK] int32 — scratch (same size as INDICES)
    TOPK: tl.constexpr,
):
    """Build ``score_src_idx``: per-token ascending sort of expert IDs.

    Pure scalar ops — compatible with Triton 3.5.0 + SM103a (no ``tl.min``).
    """
    t = tl.program_id(0)
    start = tl.load(NAEPT_ptr + t)
    end = tl.load(NAEPT_ptr + t + 1)
    n_valid = end - start
    if n_valid == 0:
        return

    base = t * TOPK
    for k in tl.static_range(TOPK):
        e = tl.load(INDICES_ptr + base + k)
        tl.store(WORK_ptr + base + k, tl.where(e >= 0, e, 0x7FFF))

    for j in range(TOPK):
        if j < n_valid:
            best_e: tl.int32 = 0x7FFF + 1
            best_c: tl.int32 = 0
            for k in tl.static_range(TOPK):
                ek = tl.load(WORK_ptr + base + k)
                better = (ek < best_e) | ((ek == best_e) & (k < best_c))
                best_e = tl.where(better, ek, best_e)
                best_c = tl.where(better, k, best_c)
            tl.store(OUT_ptr + start + j, (t * TOPK + best_c).to(tl.int64))
            tl.store(WORK_ptr + base + best_c, 0x7FFF + 1)


def _differentiable_router_scores(
    dispatched_probs: torch.Tensor,    # [N_recv, topk] float32
    dispatched_indices: torch.Tensor,  # [N_recv, topk] int32, -1 = masked
    naept: torch.Tensor,               # [N_recv+1] int32 — prefix-sum of valid counts
    TK: int,                           # number of real (non-padding) entries
    TK_padded: int,                    # TK + padding for 128-alignment
    E: int,                            # number of experts
    score_src_idx: torch.Tensor | None = None,  # [TK] int32 from CUDA kernel
) -> torch.Tensor:
    """Reconstruct router_scores from ``dispatched_probs`` with an autograd
    connection so that ``ds`` flows back to the caller's gate.

    Within each token, entries are ordered by ascending expert ID, matching the
    CUDA metadata kernel's rank-based ordering in ``scatter_and_fixup_kernel``.

    Zero CPU-GPU synchronization. Fast path uses ``score_src_idx`` produced as
    a side-output of ``scatter_and_fixup_kernel``; the Triton kernel is the
    fallback for the Python metadata path.
    """
    N_recv, topk = dispatched_indices.shape
    device = dispatched_indices.device
    if hasattr(device, "type") and not isinstance(device, str):
        idx = getattr(device, "index", None) or 0
        device = f"cuda:{int(idx)}"

    if dispatched_indices.stride(1) != 1:
        raise ValueError("dispatched_indices must be contiguous in last dim")
    if "int32" not in str(dispatched_indices.dtype):
        raise ValueError(f"dispatched_indices: expected int32, got {dispatched_indices.dtype}")
    if naept.stride(0) != 1:
        raise ValueError("naept must be contiguous 1D")
    if naept.shape[0] != N_recv + 1:
        raise ValueError(f"naept: expected shape ({N_recv+1},), got {naept.shape}")

    if TK == 0:
        return dispatched_probs.new_zeros(TK_padded)

    if score_src_idx is not None:
        if score_src_idx.shape[0] != TK:
            raise ValueError(
                f"score_src_idx: expected shape ({TK},), got {score_src_idx.shape}"
            )
        gather_idx = score_src_idx.to(torch.int64)
    else:
        built = torch.empty(TK, dtype=torch.int64, device=device)
        work = torch.empty_like(dispatched_indices)
        _build_score_src_idx_kernel[(N_recv,)](
            dispatched_indices, naept, built, work,
            TOPK=triton.next_power_of_2(topk),
        )
        gather_idx = built

    gathered = _GatherRouterScores.apply(
        dispatched_probs.reshape(-1), gather_idx, dispatched_probs.numel()
    )  # [TK]

    if TK_padded > TK:
        padding = torch.zeros(
            TK_padded - TK, dtype=gathered.dtype, device=device,
        )
        return torch.cat([gathered, padding])
    return gathered


# ── Module-level back-compat helpers (legacy tests / jit_warmup only) ────────
# The production ``SonicMoEMlpNode`` does NOT touch any state below.
# These shims exist solely so the legacy non-instance API used by a few
# benchmarks and by ``jit_warmup`` keeps working with no behavioural change.
# Each call is keyed by ``(data_ptr, _inplace_version)`` of the input weights,
# so multiple expert lists do not collide; the only "global"-flavoured leak is
# the cache dict, which is content-addressed and explicitly invalidated by
# ``invalidate_weight_caches()``.

_LEGACY_W_CACHE: dict[tuple, torch.Tensor] = {}
_LEGACY_PENDING_FLUSH: list = []


def _cache_key(*tensors: torch.Tensor) -> tuple:
    def _ver(t):
        v = getattr(t, '_inplace_version', None)
        if v is not None:
            return v() if callable(v) else v
        return getattr(t, '_version', 0)
    return tuple((t.data_ptr(), _ver(t)) for t in tensors)


def invalidate_weight_caches() -> None:
    """Drop legacy stack/main_grad cache + FP8 weight quant cache + topk cache.

    Production users should call ``SonicMoEMlpNode.step()`` instead, which
    invalidates only that instance's caches. This function is retained for
    legacy benchmarks and the JIT warmup harness.
    """
    _LEGACY_W_CACHE.clear()
    clear_all_fp8_weight_caches()
    invalidate_topk_cache()


def _stack_w1_into(
    cache: dict, experts: Sequence[Any], H: int, I: int,
) -> torch.Tensor:
    """Build (or fetch) the ``[2I, H, E]`` logical view of stacked W1.

    *No main_grad allocation* — call :func:`_alloc_main_grad_w1` lazily from
    the backward path (or :meth:`SonicMoEMlpNode.step`) so memory is only
    paid when gradients are actually accumulated.
    """
    key = ("w1",) + _cache_key(*(e.up_gate_proj.weight for e in experts))
    cached = cache.get(key)
    if cached is not None:
        return cached
    E = len(experts)
    stacked = torch.stack([e.up_gate_proj.weight for e in experts])  # [E, H, 2I]
    gate, up = stacked[:, :, :I], stacked[:, :, I:]
    physical = (
        torch.stack([gate, up], dim=3)
            .reshape(E, H, 2 * I)
            .permute(0, 2, 1)
            .contiguous()
    )  # [E, 2I, H]
    w1 = physical.permute(1, 2, 0)  # logical [2I, H, E]
    w1.stop_gradient = False
    cache[key] = w1
    return w1


def _stack_w2_into(
    cache: dict, experts: Sequence[Any],
) -> torch.Tensor:
    """Build (or fetch) the ``[H, I, E]`` logical view of stacked W2.

    *No main_grad allocation* — see :func:`_stack_w1_into`.
    """
    key = ("w2",) + _cache_key(*(e.down_proj.weight for e in experts))
    cached = cache.get(key)
    if cached is not None:
        return cached
    physical = (
        torch.stack([e.down_proj.weight for e in experts])
            .permute(0, 2, 1)
            .contiguous()
    )  # [E, H, I]
    w2 = physical.permute(1, 2, 0)  # logical [H, I, E]
    w2.stop_gradient = False
    cache[key] = w2
    return w2


def _alloc_main_grad_w1(
    cache: dict, experts: Sequence[Any], H: int, I: int,
) -> torch.Tensor:
    """Lazily allocate the ``[E, H, 2I]`` fp32 ``main_grad`` accumulator for
    W1 and alias each expert's ``up_gate_proj.weight.main_grad`` to its slice.

    Triggered on the first backward (via ``_w1_native_view``) or the first
    call to :meth:`SonicMoEMlpNode.step`. Idempotent within a step; cleared
    by :meth:`SonicMoEMlpNode.invalidate_caches`.
    """
    key = ("w1_mg",) + _cache_key(*(e.up_gate_proj.weight for e in experts))
    cached = cache.get(key)
    if cached is not None:
        return cached
    E = len(experts)
    mg = paddle.zeros([E, H, 2 * I], dtype="float32")
    cache[key] = mg
    for e_idx, exp in enumerate(experts):
        exp.up_gate_proj.weight.main_grad = mg[e_idx]
    return mg


def _alloc_main_grad_w2(
    cache: dict, experts: Sequence[Any],
) -> torch.Tensor:
    """Lazily allocate the ``[E, I, H]`` fp32 ``main_grad`` accumulator for W2."""
    key = ("w2_mg",) + _cache_key(*(e.down_proj.weight for e in experts))
    cached = cache.get(key)
    if cached is not None:
        return cached
    w0 = experts[0].down_proj.weight
    # down_proj.weight shape is [I, H] (Paddle linear: [in, out])
    I, H = int(w0.shape[0]), int(w0.shape[1])
    E = len(experts)
    mg = paddle.zeros([E, I, H], dtype="float32")
    cache[key] = mg
    for e_idx, exp in enumerate(experts):
        exp.down_proj.weight.main_grad = mg[e_idx]
    return mg


def stack_ernie_w1(experts: Sequence[Any], H: int, I: int) -> torch.Tensor:
    """Legacy shim — prefer ``SonicMoEMlpNode``. Returns the ``[2I, H, E]``
    logical view of stacked W1 (cached process-wide for back-compat).
    No main_grad is allocated by this call."""
    return _stack_w1_into(_LEGACY_W_CACHE, experts, H, I)


def stack_ernie_w2(experts: Sequence[Any]) -> torch.Tensor:
    """Legacy shim — prefer ``SonicMoEMlpNode``."""
    return _stack_w2_into(_LEGACY_W_CACHE, experts)


def _flush_native_grads_for(mg1: torch.Tensor, mg2: torch.Tensor) -> None:
    """Convert native-layout ``main_grad`` storage in-place to ERNIE layout.

    * W1 storage is ``[E, 2I, H]`` (gate0, up0, gate1, up1, ...) interleaved;
      ERNIE expects ``[E, H, 2I]`` split-half (all gates, then all ups).
    * W2 storage is ``[E, H, I]``; ERNIE expects ``[E, I, H]``.

    Single ``contiguous()`` scratch per weight; freed after ``copy_``.
    """
    E, H, two_I = mg1.shape
    I = two_I // 2
    native1 = mg1.view(E, two_I, H)
    rhs1 = native1.view(E, I, 2, H).permute(0, 3, 2, 1).contiguous()
    mg1.view(E, H, 2, I).copy_(rhs1)

    E2, I2, H2 = mg2.shape
    native2 = mg2.view(E2, H2, I2)
    rhs2 = native2.permute(0, 2, 1).contiguous()
    mg2.copy_(rhs2)


def flush_native_grads() -> None:
    """Legacy shim — flush every (mg1, mg2) pair queued by the legacy stacker.

    Production code should call ``SonicMoEMlpNode.step()`` instead.
    """
    pending, _LEGACY_PENDING_FLUSH[:] = list(_LEGACY_PENDING_FLUSH), []
    for mg1, mg2 in pending:
        _flush_native_grads_for(mg1, mg2)


# ── SonicMoEMlpNode (production path) ────────────────────────────────────────

class SonicMoEMlpNode:
    """Drop-in replacement for ERNIE's MlpNode (topk DeepEP dispatch path).

    Packages unzip + FP8 FFN + zip into a single callable.  Uses DeepEP's
    topk dispatch metadata for zero-sync metadata conversion.

    All caches are **per-instance** — pipeline parallelism, multi-layer
    models, and interleaved forward / backward across layers are supported
    without any cross-layer state.

    Gradient contract:
      * ``dx`` flows back through Paddle autograd (no detach).
      * ``dw1`` / ``dw2`` accumulate into per-expert ``main_grad`` via a
        native-layout view of a per-instance fp32 buffer; layout is
        converted in-place to ERNIE format on ``step()``.
      * ``ds`` flows back through Paddle autograd.

    Performance properties:
      * Route-level padding (no x-padding cat / copy).
      * Native-layout grad accumulation (1 add_ kernel per weight, no
        per-iter transpose).
      * Fused single-pass FP8 prequantize on first microbatch of each step.

    Parameters
    ----------
    experts
        Per-expert modules, each with ``up_gate_proj.weight [H, 2I]`` and
        ``down_proj.weight [I, H]``.
    n_experts
        Number of local experts (E).
    hidden_size, intermediate_size
        Model dims.  ``up_gate_proj`` has 2*I columns.
    activation_type
        Activation function (default SWIGLU).
    stream_id
        CUDA stream index for FP8 ops.
    """

    def __init__(
        self,
        experts: Sequence[Any],
        n_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation_type: ActivationType = ActivationType.SWIGLU,
        stream_id: int = 0,
    ):
        self._experts = list(experts)
        self._E = n_experts
        self._H = hidden_size
        self._I = intermediate_size
        self._activation_type = activation_type
        self._stream_id = stream_id

        # Per-instance state — no module-level globals.
        self._w_cache: dict[tuple, torch.Tensor] = {}
        self._pending_flush: bool = False
        self._warmed_for_step: bool = False

    # ── Weight layout helpers (instance-scoped) ─────────────────────────────

    def _stacked_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        w1 = _stack_w1_into(self._w_cache, self._experts, self._H, self._I)
        w2 = _stack_w2_into(self._w_cache, self._experts)
        return w1, w2

    def _w1_main_grad(self) -> torch.Tensor:
        """Lazy: returns the ``[E, H, 2I]`` fp32 buffer aliased to every
        ``expert.up_gate_proj.weight.main_grad``.  Allocated on first call."""
        return _alloc_main_grad_w1(self._w_cache, self._experts, self._H, self._I)

    def _w2_main_grad(self) -> torch.Tensor:
        """Lazy: returns the ``[E, I, H]`` fp32 buffer aliased to every
        ``expert.down_proj.weight.main_grad``.  Allocated on first call."""
        return _alloc_main_grad_w2(self._w_cache, self._experts)

    def _w1_native_view(self) -> torch.Tensor:
        """Storage of ``_w1_main_grad`` reinterpreted as ``[E, 2I, H]`` —
        the native CUTLASS wgrad layout."""
        mg = self._w1_main_grad()
        E, H, two_I = mg.shape
        return mg.view(E, two_I, H)

    def _w2_native_view(self) -> torch.Tensor:
        """Storage of ``_w2_main_grad`` reinterpreted as ``[E, H, I]``."""
        mg = self._w2_main_grad()
        E, I, H = mg.shape
        return mg.view(E, H, I)

    def _flush_native_grads(self) -> None:
        if not self._pending_flush:
            return
        _flush_native_grads_for(self._w1_main_grad(), self._w2_main_grad())
        self._pending_flush = False

    def flush_grads(self) -> None:
        """Public alias for the deferred wgrad layout flush.

        Use this in gradient-accumulation harnesses that need to read
        ``main_grad`` between micro-iterations without invalidating the
        weight cache (which ``step()`` does as part of its post-optimizer
        contract).
        """
        self._flush_native_grads()

    # ── Public API ──────────────────────────────────────────────────────────

    def prequantize_weights(self) -> None:
        """Fused single-pass FP8 prequantize for w1/w2 (all four layouts).

        Reads each BF16 weight ONCE and writes both transposed FP8 layouts +
        ISA-packed scales in a single Triton kernel per weight — ~3x faster
        than letting the four ``precompute_weight_fp8_*`` helpers fire lazily
        inside the first microbatch's forward.

        Idempotent within a step: the second call is a cheap cache lookup.
        ``step()`` clears the flag so the next step re-quantizes the freshly
        updated weights.
        """
        if self._warmed_for_step:
            return
        w1, w2 = self._stacked_weights()
        precompute_weight_fp8_warmup(w1, w2)
        self._warmed_for_step = True

    def warmup(self, total_K_list: list[int] | None = None, max_workers: int = 0):
        """Pre-compile all JIT kernels.  Call once after model construction."""
        from sonicmoe.jit_warmup import warmup_jit
        warmup_jit(
            self._E, self._H, self._I,
            device=f"cuda:{torch.cuda.current_device()}",
            fp8=True,
            total_K_list=total_K_list,
            max_workers=max_workers,
        )

    def forward(
        self,
        dispatched_hidden_states: torch.Tensor,
        tokens_per_expert: list[int] | torch.Tensor,
        dispatched_indices: torch.Tensor | None = None,
        dispatched_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run FP8 expert FFN on DeepEP-dispatched tokens.

        Parameters
        ----------
        dispatched_hidden_states : Tensor [N_recv, H] bf16
            Received tokens from DeepEP dispatch (topk layout).
        tokens_per_expert : list[int] or Tensor [E]
            Per-expert token counts from DeepEP ``buffer.dispatch()``.
        dispatched_indices : Tensor [N_recv, topk] int32
            Local expert indices from DeepEP dispatch. ``-1`` = masked.
        dispatched_probs : Tensor [N_recv, topk] float32
            Routing probabilities from DeepEP dispatch.

        Returns
        -------
        Tensor [N_recv, H] bf16
            Expert output, same token ordering as input.
        """
        x = dispatched_hidden_states
        E = self._E
        H = self._H
        I = self._I

        # Topk path: real DeepEP dispatch with multi-expert routing.
        # Identity layout (K=1, pre-sorted) was removed — it had an
        # unfixable dx bug due to expert-sorted ↔ token-order mismatch
        # when total_pad_rows > 0.  All production callers use topk.
        assert dispatched_indices is not None, (
            "dispatched_indices is required (identity layout path removed)"
        )
        assert dispatched_probs is not None, (
            "dispatched_probs required when dispatched_indices is given"
        )
        (
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            _router_scores_data,
            TK_padded,
            total_pad_rows,
            N_recv,
            score_src_idx,
        ) = deepep_topk_to_sonic_metadata(
            dispatched_indices, dispatched_probs,
            tokens_per_expert, E, device=x.device,
        )

        # Build router_scores differentiably from dispatched_probs so that
        # ds gradients flow back to the caller (e.g. DeepEP's gate).
        router_scores = _differentiable_router_scores(
            dispatched_probs, dispatched_indices,
            num_activated_expert_per_token_offset,
            TK_padded - total_pad_rows, TK_padded, E,
            score_src_idx=score_src_idx,
        )
        # T_down = N_recv for the topk path.
        T_down = N_recv
        topk = dispatched_indices.shape[1]

        # Stack weights (per-instance cache) + fused FP8 prequantize on first
        # microbatch of the step.
        self.prequantize_weights()
        w1, w2 = self._stacked_weights()

        # Tensor-input flags for autograd.
        x.stop_gradient = False
        router_scores.stop_gradient = False
        x_gather_idx.stop_gradient = True
        s_scatter_idx.stop_gradient = True
        w1.stop_gradient = True
        w2.stop_gradient = True

        return _SonicMoEDeepEPFunc.apply(
            x,
            router_scores,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            w1, w2,
            self,                       # node — backward routes wgrad here
            E, N_recv, T_down, TK_padded, topk,
            self._activation_type,
            self._stream_id,
        )

    def step(self) -> None:
        """Commit deferred wgrads into per-expert ``main_grad``.

        **Call BEFORE** ``optimizer.step()`` — the optimizer reads
        ``weight.main_grad`` and that buffer is only in the correct
        ERNIE-native layout *after* this flush.

        Training-loop contract::

            for microbatch in microbatches:
                out = node(x, tpe, indices, probs)
                out.backward(grad)
            node.step()                 # ← flush wgrads (this method)
            optimizer.step()            #   reads weight.main_grad
            optimizer.zero_grad()       #   zeros main_grad in place

        Cache invalidation is *not* needed here: ``_w_cache`` and the FP8
        weight cache are keyed by ``(data_ptr, _inplace_version)``, which
        bumps automatically when the optimizer updates the weights, so the
        next forward naturally misses and rebuilds.  Use
        :meth:`invalidate_caches` if you need eager release for memory
        pressure (e.g. parameter swap-out).
        """
        self._flush_native_grads()
        self._warmed_for_step = False

    def invalidate_caches(self) -> None:
        """Eagerly drop this instance's stacked-weight / main_grad cache and
        the process-wide FP8 weight quant + topk caches.  Optional — only
        useful for memory pressure or when weight tensors are swapped out.

        Note: the ``main_grad`` buffer is dropped here, so it will be
        re-allocated (and zeroed) on the next backward.
        """
        self._w_cache.clear()
        clear_all_fp8_weight_caches()
        invalidate_topk_cache()
        self._warmed_for_step = False

    def __call__(
        self,
        dispatched_hidden_states: torch.Tensor,
        tokens_per_expert: list[int] | torch.Tensor,
        dispatched_indices: torch.Tensor | None = None,
        dispatched_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(
            dispatched_hidden_states, tokens_per_expert,
            dispatched_indices, dispatched_probs,
        )


# ── Internal PyLayer (topk DeepEP metadata, production path) ─────────────────

class _SonicMoEDeepEPFunc(paddle.autograd.PyLayer):
    """PyLayer for the topk DeepEP dispatch path (production).

    Uses route-level padding (frontier design): ``x`` is NOT padded.  Padding
    rows gather from row 0 with score=0, contributing nothing to output or
    grads.

    ``T_down = N_recv``: ``_DownProjection`` outputs ``[N_recv, H]`` directly.

    The owning ``SonicMoEMlpNode`` is passed in as a non-tensor positional
    argument and stashed on ``ctx`` so that ``backward`` can route wgrads into
    its per-instance native-layout view and signal the deferred flush.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,            # [T, H] bf16  (grad)
        router_scores: torch.Tensor,             # [TK_padded] float32
        expert_frequency_offset: torch.Tensor,   # [E+1] int32
        x_gather_idx: torch.Tensor,              # [TK_padded] int32
        s_scatter_idx: torch.Tensor,             # [TK_padded] int32
        s_reverse_scatter_idx: torch.Tensor,     # [TK] int32
        num_activated_expert_per_token_offset: torch.Tensor,  # [N_recv+1]
        w1: torch.Tensor,                        # [2I, H, E] bf16
        w2: torch.Tensor,                        # [H, I, E] bf16
        node: "SonicMoEMlpNode",
        n_experts: int = 0,
        N_recv: int = 0,
        T_down: int = 0,
        TK_padded: int = 0,
        topk: int = 1,
        activation_type: ActivationType = ActivationType.SWIGLU,
        stream_id: int = 0,
    ) -> torch.Tensor:
        # ── UpProjection forward (via FakeCtx) ───────────────────────────
        up_ctx = _FakeCtx()
        with enable_fp8(True):
            # Refresh FP8 config inside the enable_fp8(True) block so the
            # snapshot sees ``_IS_FP8_ACTIVE=True`` regardless of the global
            # state any prior caller / test may have left behind.
            _refresh_fp8_config()
            y1, z = _UpProjection.forward(
                up_ctx,
                hidden_states, w1, None,        # x, w1, b1
                expert_frequency_offset, TK_padded, None, stream_id,
                x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                True,                           # is_varlen_K
                activation_type,
                False,                          # is_inference_mode_enabled
                False,                          # use_low_precision_postact_buffer
            )

        # ── DownProjection forward (via FakeCtx) ─────────────────────────
        down_ctx = _FakeCtx()
        with enable_fp8(True):
            out = _DownProjection.forward(
                down_ctx,
                y1, z, w2, None,                # y1, z, w2, b2
                router_scores, s_scatter_idx, expert_frequency_offset,
                T_down, topk, stream_id,
                x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                True,                           # is_varlen_K
                activation_type,
                None,                           # fp8_protocol
            )

        ctx._up_ctx = up_ctx
        ctx._down_ctx = down_ctx
        ctx._node = node
        ctx._I = w1.shape[0] // 2

        return out

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        up_ctx = ctx._up_ctx
        down_ctx = ctx._down_ctx
        node: SonicMoEMlpNode = ctx._node

        # Native-layout views into per-instance main_grad storage.  Caller
        # MUST run ``node.step()`` before reading main_grad — see class
        # docstring for the contract.
        w1_native = node._w1_native_view()  # [E, 2I, H] fp32
        w2_native = node._w2_native_view()  # [E, H, I] fp32
        node._pending_flush = True

        # ── DownProjection backward ──────────────────────────────────────
        down_ctx._wgrad_w2_accumulator = w2_native
        down_grads = _DownProjection.backward(down_ctx, output_grad)
        dz = down_grads[1]
        dw2 = down_grads[2]

        # ds is at index 4 if b2 is present, 3 otherwise.  b2 is always None
        # in this path, so ds is always at index 3 — but keep the guard for
        # safety in case the projection signature evolves.
        ds_idx = 4 if getattr(down_ctx, '_has_b2', False) else 3
        ds = down_grads[ds_idx]

        # ── UpProjection backward ────────────────────────────────────────
        up_ctx._wgrad_w1_accumulator = w1_native
        up_grads = _UpProjection.backward(up_ctx, None, dz)
        dx = up_grads[0]
        dw1 = up_grads[1]

        # FP8 wgrad accumulator path MUST write into the native view directly
        # and return None — ``SonicMoEMlpNode`` always runs in
        # ``enable_fp8(True)``, so the BF16 fallback (which would return
        # non-None tensors) is unreachable.
        assert dw1 is None, (
            "Unexpected non-None dw1 in SonicMoEMlpNode backward — the FP8 "
            "wgrad accumulator must write into _w1_native_view directly. "
            "If this fires, the BF16 fallback was hit unexpectedly."
        )
        assert dw2 is None, (
            "Unexpected non-None dw2 in SonicMoEMlpNode backward — see dw1."
        )

        # Tensor inputs: hidden_states, router_scores, efo, x_gather, s_scatter,
        #                s_reverse_scatter, naept_offset, w1, w2
        return dx, ds, None, None, None, None, None, None, None
