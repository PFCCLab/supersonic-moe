#!/usr/bin/env python
"""Multi-layer correctness for SonicMoEMlpNode + flush_native_grads.

Validates the per-layer redesign of native-grad accumulation. Constructs N
independent MlpNode instances on the same GPU, runs forward+backward through
all of them in sequence (autograd unwinds in reverse), and verifies that
EVERY layer's main_grad matches a single-layer reference run.

This is the regression for the multi-layer global-shadowing bug: the OLD
single-global ``_NATIVE_W{1,2}_GRAD`` design silently dropped grads from
all layers except the last to register, because the global was overwritten
on each ``_ensure_native_grads`` call when ``_NATIVE_GRAD_EXPERTS is not experts``.
"""

import math
import os
import sys

venv = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv"
python_bin = os.path.join(venv, "bin", "python")
if os.path.realpath(sys.prefix) != os.path.realpath(venv):
    print("Switch venv:", venv)
    os.execv(python_bin, [python_bin, *sys.argv])

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")

_REPO = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()
import torch

from sonicmoe.ernie_compat import (
    SonicMoEMlpNode,
    flush_native_grads,
    invalidate_weight_caches,
)
from sonicmoe.ernie_compat import mlp_node_v2 as _mlp_module
import sonicmoe.functional as functional

functional._ALIGNMENT_ASSUMED = True
functional._ALIGNMENT_STREAK = 100


H = 512
I_DIM = 1024
E = 8


class MockExpert:
    def __init__(self, h, i, seed):
        paddle.seed(seed)
        self.up_gate_proj = type("P", (), {
            "weight": paddle.randn([h, 2 * i], dtype="bfloat16") / math.sqrt(h),
        })()
        self.down_proj = type("P", (), {
            "weight": paddle.randn([i, h], dtype="bfloat16") / math.sqrt(i),
        })()
        self.up_gate_proj.weight.stop_gradient = False
        self.down_proj.weight.stop_gradient = False


def _make_layer(seed_base, h=H, i=I_DIM, e=E):
    return [MockExpert(h, i, seed_base + j) for j in range(e)]


def _zero_main_grads(experts):
    for exp in experts:
        for name in ("up_gate_proj", "down_proj"):
            w = getattr(exp, name).weight
            if hasattr(w, "main_grad") and w.main_grad is not None:
                w.main_grad.zero_()


def _flush_all(*nodes):
    """Flush every node's pending native-grad layout (post-S74 per-instance API)."""
    for n in nodes:
        n.flush_grads()


def _snapshot_main_grads(experts):
    """Snapshot main_grad fp32 tensors AS torch tensors (post-flush layout)."""
    out = []
    for exp in experts:
        mg1 = torch.from_dlpack(exp.up_gate_proj.weight.main_grad.detach()).clone()
        mg2 = torch.from_dlpack(exp.down_proj.weight.main_grad.detach()).clone()
        out.append((mg1, mg2))
    return out


def _topk_meta(T, K, E, seed):
    # Match the style of test_mlpnode_precision.py: build dispatched_indices
    # via topk on random scores so all expert IDs are valid GPU int32.
    torch.manual_seed(seed)
    raw_scores = torch.randn(T, E, device="cuda")
    _, top_experts = raw_scores.topk(K, dim=-1)
    indices = top_experts.int()
    probs = torch.rand(T, K, device="cuda") * 0.5 + 0.5
    probs = probs / probs.sum(dim=1, keepdim=True)
    probs = probs.float()
    tpe = [int((indices == e).sum().item()) for e in range(E)]
    return indices, probs, tpe


def _run_layer_once(experts, x, indices, probs, tpe, grad_out):
    """Single-layer backward: zero main_grad, forward, backward, flush.

    Returns the post-flush main_grad snapshot.
    """
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I_DIM)
    # Warmup so wgrad caches are stable.
    for _ in range(3):
        out_w = node.forward(x.clone().detach(), tpe,
                             dispatched_indices=indices, dispatched_probs=probs)
        out_w.backward(grad_out.clone())
    node.flush_grads()
    _zero_main_grads(experts)

    x_in = x.clone().detach()
    x_in.stop_gradient = False
    out = node.forward(x_in, tpe, dispatched_indices=indices, dispatched_probs=probs)
    out.backward(grad_out.clone())
    node.flush_grads()
    return _snapshot_main_grads(experts)


def _run_layer_in_chain(experts, x, indices, probs, tpe, grad_out):
    """Forward only — chain owner controls flush timing.

    Returns (output, x_in_with_grad, node).
    """
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I_DIM)
    # Warmup
    for _ in range(3):
        out_w = node.forward(x.clone().detach(), tpe,
                             dispatched_indices=indices, dispatched_probs=probs)
        out_w.backward(grad_out.clone())
    node.flush_grads()
    _zero_main_grads(experts)

    x_in = x.clone().detach()
    x_in.stop_gradient = False
    out = node.forward(x_in, tpe, dispatched_indices=indices, dispatched_probs=probs)
    return out, x_in, node


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a = a.float().flatten()
    b = b.float().flatten()
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    rrmse = float(((a - b).norm() / (b.norm() + 1e-10)).item())
    return cos, rrmse


def test_two_layer_main_grad_matches_independent():
    """Two MlpNode layers run sequentially → both layers' grads must match a
    single-layer reference run (no global-shadowing).

    Composition:
      x -> layer0 -> y0 -> layer1 -> y1 -> sum -> backward
    Reference: each layer run independently with the same per-layer input/grad.
    """
    print("\n[two_layer_independent_inputs]")
    T = 256
    K = 4
    torch.manual_seed(0)
    x0 = (torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
    grad_out0 = (torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()
    indices0, probs0, tpe0 = _topk_meta(T, K, E, seed=11)

    x1 = (torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
    grad_out1 = (torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()
    indices1, probs1, tpe1 = _topk_meta(T, K, E, seed=22)

    experts0 = _make_layer(seed_base=100)
    experts1 = _make_layer(seed_base=200)

    # ── Reference: each layer run independently ────────────────────────────
    ref0 = _run_layer_once(experts0, x0, indices0, probs0, tpe0, grad_out0)
    ref1 = _run_layer_once(experts1, x1, indices1, probs1, tpe1, grad_out1)

    # ── Combined: both layers used in the SAME backward window ─────────────
    # Independent inputs (not chained) — semantically equivalent to two
    # independent runs, except both share the global flush list. This is the
    # exact failure mode of the old single-global design.
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node0 = SonicMoEMlpNode(experts=experts0, n_experts=E, hidden_size=H, intermediate_size=I_DIM)
    node1 = SonicMoEMlpNode(experts=experts1, n_experts=E, hidden_size=H, intermediate_size=I_DIM)
    # Warmup both.
    for _ in range(3):
        out_w0 = node0.forward(x0.clone().detach(), tpe0,
                               dispatched_indices=indices0, dispatched_probs=probs0)
        out_w0.backward(grad_out0.clone())
        out_w1 = node1.forward(x1.clone().detach(), tpe1,
                               dispatched_indices=indices1, dispatched_probs=probs1)
        out_w1.backward(grad_out1.clone())
    _flush_all(node0, node1)
    _zero_main_grads(experts0)
    _zero_main_grads(experts1)

    x0_in = x0.clone().detach(); x0_in.stop_gradient = False
    x1_in = x1.clone().detach(); x1_in.stop_gradient = False
    out0 = node0.forward(x0_in, tpe0, dispatched_indices=indices0, dispatched_probs=probs0)
    out1 = node1.forward(x1_in, tpe1, dispatched_indices=indices1, dispatched_probs=probs1)
    out0.backward(grad_out0.clone())
    out1.backward(grad_out1.clone())

    # Critical: BEFORE flush, both nodes must be flagged as pending — the
    # per-instance flush flag is the post-S74 replacement for the
    # ``_PENDING_FLUSH_LAYERS`` global FIFO.
    assert node0._pending_flush, "node0 pending flush flag not set after backward"
    assert node1._pending_flush, "node1 pending flush flag not set after backward"

    node0.flush_grads()
    node1.flush_grads()
    assert not node0._pending_flush and not node1._pending_flush
    got0 = _snapshot_main_grads(experts0)
    got1 = _snapshot_main_grads(experts1)

    # ── Compare per expert ─────────────────────────────────────────────────
    failures = []
    for layer_name, ref, got in (("layer0", ref0, got0), ("layer1", ref1, got1)):
        for e_idx, ((mg1_ref, mg2_ref), (mg1_got, mg2_got)) in enumerate(zip(ref, got)):
            cos1, rr1 = _diff(mg1_got, mg1_ref)
            cos2, rr2 = _diff(mg2_got, mg2_ref)
            tag = f"  {layer_name}.expert{e_idx}"
            print(f"{tag} dw1 cos={cos1:.6f} rrmse={rr1:.3e}   dw2 cos={cos2:.6f} rrmse={rr2:.3e}")
            if cos1 < 0.9999 or cos2 < 0.9999 or rr1 > 1e-3 or rr2 > 1e-3:
                failures.append((tag, cos1, rr1, cos2, rr2))

    if failures:
        print(f"\nFAILURES: {len(failures)}")
        for f in failures:
            print(f"  {f}")
        raise AssertionError(f"{len(failures)} expert grads diverged from single-layer ref")
    print("PASS")


def test_chain_two_layers_main_grad_consistency():
    """Chained layers: x -> layer0 -> layer1 -> loss. Verifies that both
    layers' main_grads survive the joint backward + single flush.
    """
    print("\n[two_layer_chained]")
    T = 256
    K = 4
    torch.manual_seed(1)
    x0 = (torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
    grad_out_final = (torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()
    indices0, probs0, tpe0 = _topk_meta(T, K, E, seed=33)
    indices1, probs1, tpe1 = _topk_meta(T, K, E, seed=44)

    experts0 = _make_layer(seed_base=300)
    experts1 = _make_layer(seed_base=400)

    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node0 = SonicMoEMlpNode(experts=experts0, n_experts=E, hidden_size=H, intermediate_size=I_DIM)
    node1 = SonicMoEMlpNode(experts=experts1, n_experts=E, hidden_size=H, intermediate_size=I_DIM)

    # Warmup the chain.
    for _ in range(3):
        x_w = x0.clone().detach()
        x_w.stop_gradient = False
        h0 = node0.forward(x_w, tpe0, dispatched_indices=indices0, dispatched_probs=probs0)
        h1 = node1.forward(h0, tpe1, dispatched_indices=indices1, dispatched_probs=probs1)
        h1.backward(grad_out_final.clone())
    _flush_all(node0, node1)
    _zero_main_grads(experts0)
    _zero_main_grads(experts1)

    x_in = x0.clone().detach(); x_in.stop_gradient = False
    h0 = node0.forward(x_in, tpe0, dispatched_indices=indices0, dispatched_probs=probs0)
    h1 = node1.forward(h0, tpe1, dispatched_indices=indices1, dispatched_probs=probs1)
    h1.backward(grad_out_final.clone())

    assert node0._pending_flush and node1._pending_flush, (
        "Chained backward should flag both nodes for pending flush."
    )

    node0.flush_grads()
    node1.flush_grads()

    # Sanity: both layers' main_grads must be NON-zero (the bug would zero
    # one layer's grads silently).
    for tag, experts in (("layer0", experts0), ("layer1", experts1)):
        for e_idx, exp in enumerate(experts):
            for name in ("up_gate_proj", "down_proj"):
                mg = exp.__dict__[name].weight.main_grad
                mg_t = torch.from_dlpack(mg.detach())
                norm = float(mg_t.float().norm().item())
                if norm < 1e-8:
                    raise AssertionError(
                        f"{tag}.expert{e_idx}.{name}.main_grad is zero — likely "
                        f"clobbered by other layer (multi-layer global bug)."
                    )
    print(f"PASS — both layers have non-zero main_grad after joint flush.")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-parallel / 1F1B-style: fully decoupled, arbitrarily interleaved
# fwd/bwd across N independent MoE layers. Mirrors what real PP schedules do.
# ─────────────────────────────────────────────────────────────────────────────
def _build_layers(n_layers: int, T: int, K: int):
    """Build N independent layers + per-layer (x, grad_out, indices, probs, tpe).

    Each layer has its own weights, activations, and routing — there is no
    chain dependency, so we can replay any fwd/bwd permutation.
    """
    layers = []
    for li in range(n_layers):
        torch.manual_seed(7000 + li)
        x = (torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
        grad_out = (torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()
        indices, probs, tpe = _topk_meta(T, K, E, seed=8000 + li)
        experts = _make_layer(seed_base=9000 + li * 17)
        layers.append({
            "experts": experts,
            "x": x,
            "grad_out": grad_out,
            "indices": indices,
            "probs": probs,
            "tpe": tpe,
        })
    return layers


def _run_interleaved(layers, schedule):
    """Execute ``schedule`` (list of ('F'|'B', layer_idx)) and return per-layer
    main_grad snapshots after a single joint flush.

    Each F creates a fresh node + x_in for that layer; each B does .backward
    on the matching forward output.
    """
    n = len(layers)
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    nodes = [
        SonicMoEMlpNode(experts=L["experts"], n_experts=E, hidden_size=H, intermediate_size=I_DIM)
        for L in layers
    ]
    # Warmup: do one fwd+bwd per layer and flush so caches are stable.
    for li, L in enumerate(layers):
        out_w = nodes[li].forward(L["x"].clone().detach(), L["tpe"],
                                  dispatched_indices=L["indices"], dispatched_probs=L["probs"])
        out_w.backward(L["grad_out"].clone())
    _flush_all(*nodes)
    for L in layers:
        _zero_main_grads(L["experts"])

    pending_outs: dict[int, torch.Tensor] = {}
    for op, li in schedule:
        L = layers[li]
        if op == "F":
            x_in = L["x"].clone().detach(); x_in.stop_gradient = False
            out = nodes[li].forward(x_in, L["tpe"],
                                    dispatched_indices=L["indices"], dispatched_probs=L["probs"])
            assert li not in pending_outs, f"double F on layer{li}"
            pending_outs[li] = out
        else:
            assert op == "B"
            out = pending_outs.pop(li)
            out.backward(L["grad_out"].clone())

    assert not pending_outs, f"unmatched forwards: {sorted(pending_outs)}"
    _flush_all(*nodes)
    return [_snapshot_main_grads(L["experts"]) for L in layers]


def test_pipeline_parallel_interleaved():
    """N=3 layers with several non-trivial F/B interleavings (PP-style).

    Schedules tested (each must produce identical per-layer grads to a clean
    sequential baseline):
      A: F0 F1 F2 B2 B1 B0    — strict 1F1B (canonical)
      B: F0 F1 F2 B0 B1 B2    — fwd-first-bwd-first
      C: F0 F1 B0 F2 B1 B2    — fully interleaved
      D: F2 F0 F1 B1 B2 B0    — out-of-order fwds
    """
    print("\n[pipeline_parallel_interleaved]")
    T, K, N = 256, 4, 3
    layers = _build_layers(N, T, K)
    # Reference: each layer run once independently (clean, no interleave).
    refs = []
    for li, L in enumerate(layers):
        refs.append(_run_layer_once(L["experts"], L["x"], L["indices"], L["probs"], L["tpe"], L["grad_out"]))

    schedules = {
        "A_1F1B":          [("F", 0), ("F", 1), ("F", 2), ("B", 2), ("B", 1), ("B", 0)],
        "B_FFB_FBB":       [("F", 0), ("F", 1), ("F", 2), ("B", 0), ("B", 1), ("B", 2)],
        "C_interleaved":   [("F", 0), ("F", 1), ("B", 0), ("F", 2), ("B", 1), ("B", 2)],
        "D_out_of_order":  [("F", 2), ("F", 0), ("F", 1), ("B", 1), ("B", 2), ("B", 0)],
    }

    failures = []
    for name, sched in schedules.items():
        # Re-build layers each schedule to reset weights+state cleanly.
        layers_run = _build_layers(N, T, K)
        got = _run_interleaved(layers_run, sched)
        for li in range(N):
            for e_idx, ((mg1_ref, mg2_ref), (mg1_got, mg2_got)) in enumerate(zip(refs[li], got[li])):
                cos1, rr1 = _diff(mg1_got, mg1_ref)
                cos2, rr2 = _diff(mg2_got, mg2_ref)
                if cos1 < 0.9999 or cos2 < 0.9999 or rr1 > 1e-3 or rr2 > 1e-3:
                    failures.append((name, li, e_idx, cos1, rr1, cos2, rr2))
        print(f"  {name}: layers={N} all-experts OK")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f"  schedule={f[0]} layer{f[1]} expert{f[2]} dw1 cos={f[3]:.4f} rr={f[4]:.2e}  dw2 cos={f[5]:.4f} rr={f[6]:.2e}")
        raise AssertionError(f"{len(failures)} interleaved-grad mismatches across schedules")
    print("PASS")


def test_multistep_pp_accumulation():
    """Verify per-layer main_grad correctness across multiple optimizer steps
    AND across multiple micro-batches within one step (gradient accumulation).

    Two scenarios:
    (A) ``grad_accum=N_MICRO`` micro-batches → ONE flush → optimizer step.
        main_grad must equal Σ over micro-batches of single-batch ref.
    (B) ``N_STEPS`` independent optimizer steps; each step does
        zero_grad → fwd+bwd → flush. Each step's flushed main_grad must
        independently match its single-batch ref. Catches state leakage
        across optimizer steps (e.g., scratch buffers that survive flush).
    """
    print("\n[multistep_pp_accumulation]")
    T, K, N_LAYERS, N_MICRO, N_STEPS = 256, 4, 3, 4, 3
    layers = _build_layers(N_LAYERS, T, K)

    # Per-(step,micro) grad_outs.
    micro_grad_outs = []
    for s in range(N_STEPS):
        per_micro = []
        for m in range(N_MICRO):
            per_layer = []
            for li in range(N_LAYERS):
                torch.manual_seed(31_000 + s * 10_000 + m * 1000 + li)
                per_layer.append((torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous())
            per_micro.append(per_layer)
        micro_grad_outs.append(per_micro)

    # ── Reference: per-(step,micro,layer) single-layer grad ──────────────────
    refs = {}  # (s, li) -> sum-over-micros of per-micro single-layer grads
    for s in range(N_STEPS):
        for m in range(N_MICRO):
            for li in range(N_LAYERS):
                L = layers[li]
                r = _run_layer_once(L["experts"], L["x"], L["indices"], L["probs"],
                                    L["tpe"], micro_grad_outs[s][m][li])
                if (s, li) not in refs:
                    refs[(s, li)] = [(mg1.clone(), mg2.clone()) for (mg1, mg2) in r]
                else:
                    for e_idx, (mg1, mg2) in enumerate(r):
                        refs[(s, li)][e_idx] = (
                            refs[(s, li)][e_idx][0] + mg1,
                            refs[(s, li)][e_idx][1] + mg2,
                        )

    # ── Actual: PP-interleaved schedule per micro-step, ONE flush per step ──
    layers_run = _build_layers(N_LAYERS, T, K)
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    nodes = [
        SonicMoEMlpNode(experts=L["experts"], n_experts=E, hidden_size=H, intermediate_size=I_DIM)
        for L in layers_run
    ]
    # JIT/cache warmup so step 0 isn't penalized by compilation.
    for li, L in enumerate(layers_run):
        out_w = nodes[li].forward(L["x"].clone().detach(), L["tpe"],
                                  dispatched_indices=L["indices"], dispatched_probs=L["probs"])
        out_w.backward(micro_grad_outs[0][0][li].clone())
    _flush_all(*nodes)
    for L in layers_run:
        _zero_main_grads(L["experts"])

    schedules = [
        [("F", 0), ("F", 1), ("F", 2), ("B", 2), ("B", 1), ("B", 0)],
        [("F", 0), ("F", 1), ("B", 0), ("F", 2), ("B", 1), ("B", 2)],
        [("F", 2), ("F", 0), ("F", 1), ("B", 1), ("B", 2), ("B", 0)],
        [("F", 1), ("F", 0), ("F", 2), ("B", 0), ("B", 2), ("B", 1)],
    ]

    failures = []
    for s in range(N_STEPS):
        # zero_grad at the start of each optimizer step (real training semantics).
        for L in layers_run:
            _zero_main_grads(L["experts"])
        # N_MICRO micro-batches, each with its own interleaved schedule.
        for m in range(N_MICRO):
            sched = schedules[(s + m) % len(schedules)]
            pending: dict[int, torch.Tensor] = {}
            for op, li in sched:
                L = layers_run[li]
                if op == "F":
                    x_in = L["x"].clone().detach(); x_in.stop_gradient = False
                    out = nodes[li].forward(x_in, L["tpe"],
                                            dispatched_indices=L["indices"], dispatched_probs=L["probs"])
                    pending[li] = out
                else:
                    pending.pop(li).backward(micro_grad_outs[s][m][li].clone())
        _flush_all(*nodes)  # ONE flush per step (after all micro-batches).

        # Verify this step's main_grad equals sum over micros of single-layer refs.
        for li in range(N_LAYERS):
            got = _snapshot_main_grads(layers_run[li]["experts"])
            for e_idx in range(E):
                cos1, rr1 = _diff(got[e_idx][0], refs[(s, li)][e_idx][0])
                cos2, rr2 = _diff(got[e_idx][1], refs[(s, li)][e_idx][1])
                if cos1 < 0.9999 or cos2 < 0.9999 or rr1 > 1e-3 or rr2 > 1e-3:
                    failures.append((s, li, e_idx, cos1, rr1, cos2, rr2))
        print(f"  step{s}: {N_LAYERS} layers × {E} experts × {N_MICRO} micro-batches → flushed main_grad OK")

    if failures:
        for f in failures[:8]:
            print(f"  step{f[0]} layer{f[1]} expert{f[2]} dw1 cos={f[3]:.4f} rr={f[4]:.2e}  dw2 cos={f[5]:.4f} rr={f[6]:.2e}")
        raise AssertionError(f"{len(failures)} multi-step accumulation mismatches")
    print("PASS")


if __name__ == "__main__":
    test_two_layer_main_grad_matches_independent()
    test_chain_two_layers_main_grad_consistency()
    test_pipeline_parallel_interleaved()
    test_multistep_pp_accumulation()
    print("\n" + "=" * 60)
    print("ALL MULTI-LAYER TESTS PASSED")
    print("=" * 60)
