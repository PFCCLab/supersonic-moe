"""
Precision: Path A (direct .apply(), is_varlen_K=False)
       vs  Path B (SonicMoEMlpNode, is_varlen_K=True, route-level padding)

Same weights, same routing, same input, same out_grad.
Compares: output, dx, ds, dw1, dw2.
"""
import paddle
paddle.compat.enable_torch_proxy(scope={"sonicmoe", "quack", "triton"}, silent=True)

import torch, numpy as np
from sonicmoe.enums import ActivationType
from sonicmoe.functional import (
    general_routing_router_metadata,
    _UpProjection, _DownProjection,
    clear_all_fp8_weight_caches, _refresh_fp8_config,
)
from sonicmoe.functional.utils import enable_fp8
from sonicmoe.ernie_compat.mlp_node_v2 import (
    SonicMoEMlpNode, invalidate_weight_caches, flush_native_grads,
)
import sonicmoe.ernie_compat.mlp_node_v2 as _m
import paddle.nn.functional as F


def metrics(a, b, label):
    a_f = a.astype("float32").numpy() if hasattr(a, 'numpy') else a.detach().float().numpy()
    b_f = b.astype("float32").numpy() if hasattr(b, 'numpy') else b.detach().float().numpy()
    d = a_f - b_f
    rr = 100.0 * np.sqrt(np.mean(d**2)) / (np.sqrt(np.mean(b_f**2)) + 1e-12)
    cos = np.dot(a_f.flat, b_f.flat) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12)
    status = "PASS" if rr < 1.0 and cos > 0.999 else "WARN"
    print(f"  {label:10s} RRMSE={rr:8.4f}%  cos={cos:.6f}  [{status}]")


class _FL:
    def __init__(self, w): self.weight = w
class _FE:
    def __init__(self, w1, w2):
        self.up_gate_proj = _FL(w1)
        self.down_proj = _FL(w2)


def run():
    E, H, I, K, T = 8, 1536, 1024, 8, 4096
    paddle.seed(42)
    print(f"T={T} H={H} I={I} E={E} K={K}\n")

    # ── Shared data ──────────────────────────────────────────────────────
    x_data = paddle.randn([T, H], dtype="bfloat16")
    out_grad = paddle.randn([T, H], dtype="bfloat16")
    gate_w = paddle.randn([E, H], dtype="float32")
    w1_per_e = [paddle.randn([H, 2*I], dtype="bfloat16") for _ in range(E)]
    w2_per_e = [paddle.randn([I, H], dtype="bfloat16") for _ in range(E)]

    # ── Shared routing (deterministic) ───────────────────────────────────
    with paddle.no_grad():
        logits = F.linear(x_data.cast("float32"), gate_w.T)
        gates = F.softmax(logits, axis=-1)
        topk_w, topk_i = paddle.topk(gates, K, axis=-1)
        topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)

    # ── Stacked weights (SonicMoE layout) ────────────────────────────────
    def make_stacked(w1_list, w2_list):
        st1 = paddle.stack(w1_list)  # [E,H,2I]
        g, u = st1[:,:,:I], st1[:,:,I:]
        phys = paddle.stack([g,u], axis=3).reshape([E,H,2*I]).transpose([0,2,1])
        w1 = phys.transpose([1,2,0])        # [2I,H,E]
        st2 = paddle.stack(w2_list)          # [E,I,H]
        w2 = st2.transpose([0,2,1]).transpose([1,2,0])  # [H,I,E]
        return w1, w2

    # ═══════════ Path A: direct .apply(), is_varlen_K=False ══════════════
    print("--- Path A ---")
    clear_all_fp8_weight_caches()

    x_a = paddle.to_tensor(x_data.numpy(), dtype="bfloat16", stop_gradient=False)
    ts_a = paddle.to_tensor(topk_w.numpy(), dtype="float32", stop_gradient=False)
    w1_a_raw, w2_a_raw = make_stacked(w1_per_e, w2_per_e)
    w1_a = paddle.to_tensor(w1_a_raw.numpy(), dtype="bfloat16", stop_gradient=False)
    w2_a = paddle.to_tensor(w2_a_raw.numpy(), dtype="bfloat16", stop_gradient=False)

    tok_ids = paddle.arange(T, dtype="int32").unsqueeze(1).expand([T,K]).reshape([-1])
    exp_ids = topk_i.reshape([-1]).cast("int32")
    sf = ts_a.reshape([-1])
    (_, efo, xgi, ssi, srsi, naept) = general_routing_router_metadata(sf, tok_ids, exp_ids, T, E)
    ssi.stop_gradient = True

    # warm-up
    with enable_fp8(True):
        _refresh_fp8_config()
        y1,z = _UpProjection.apply(x_a,w1_a,None,efo,T*K,K,0,xgi,ssi,srsi,naept,False,ActivationType.SWIGLU,False,False)
        oa = _DownProjection.apply(y1,z,w2_a,None,ts_a,ssi,efo,T,K,0,xgi,ssi,srsi,naept,False,ActivationType.SWIGLU,None)

    # real pass (recreate leaf tensors)
    x_a = paddle.to_tensor(x_data.numpy(), dtype="bfloat16", stop_gradient=False)
    ts_a = paddle.to_tensor(topk_w.numpy(), dtype="float32", stop_gradient=False)
    w1_a = paddle.to_tensor(w1_a_raw.numpy(), dtype="bfloat16", stop_gradient=False)
    w2_a = paddle.to_tensor(w2_a_raw.numpy(), dtype="bfloat16", stop_gradient=False)
    sf = ts_a.reshape([-1])
    (_, efo, xgi, ssi, srsi, naept) = general_routing_router_metadata(sf, tok_ids, exp_ids, T, E)
    ssi.stop_gradient = True

    with enable_fp8(True):
        _refresh_fp8_config()
        y1,z = _UpProjection.apply(x_a,w1_a,None,efo,T*K,K,0,xgi,ssi,srsi,naept,False,ActivationType.SWIGLU,False,False)
        out_a = _DownProjection.apply(y1,z,w2_a,None,ts_a,ssi,efo,T,K,0,xgi,ssi,srsi,naept,False,ActivationType.SWIGLU,None)

    out_a.backward(out_grad)
    dx_a, ds_a, dw1_a, dw2_a = x_a.grad, ts_a.grad, w1_a.grad, w2_a.grad
    print(f"  out={out_a.norm().item():.2f}  dx={dx_a.norm().item():.4f}  "
          f"ds={'None' if ds_a is None else f'{ds_a.norm().item():.4f}'}  "
          f"dw1={dw1_a.norm().item():.4f}  dw2={dw2_a.norm().item():.4f}")

    # ═══════════ Path B: SonicMoEMlpNode ═════════════════════════════════
    print("\n--- Path B ---")
    clear_all_fp8_weight_caches()
    invalidate_weight_caches()
    def make_experts(w1_list, w2_list):
        exps = []
        for i in range(E):
            w1 = paddle.to_tensor(w1_list[i].numpy(), dtype="bfloat16", stop_gradient=False)
            w2 = paddle.to_tensor(w2_list[i].numpy(), dtype="bfloat16", stop_gradient=False)
            exps.append(_FE(w1, w2))
        return exps

    experts = make_experts(w1_per_e, w2_per_e)
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H,
                            intermediate_size=I, activation_type=ActivationType.SWIGLU)
    x_b = paddle.to_tensor(x_data.numpy(), dtype="bfloat16", stop_gradient=False)
    di = topk_i.cast("int32"); dp = topk_w.cast("float32")
    tpe = paddle.bincount(di.reshape([-1]).cast("int64"), minlength=E).tolist()

    # warm-up
    with enable_fp8(True):
        _refresh_fp8_config()
        _ = node(x_b, tpe, di, dp)

    # real pass
    invalidate_weight_caches()
    experts_b = make_experts(w1_per_e, w2_per_e)
    node_b = SonicMoEMlpNode(experts=experts_b, n_experts=E, hidden_size=H,
                              intermediate_size=I, activation_type=ActivationType.SWIGLU)
    x_b = paddle.to_tensor(x_data.numpy(), dtype="bfloat16", stop_gradient=False)
    with enable_fp8(True):
        _refresh_fp8_config()
        out_b = node_b(x_b, tpe, di, dp)

    out_b.backward(out_grad)
    flush_native_grads()
    dx_b = x_b.grad
    print(f"  out={out_b.norm().item():.2f}  dx={dx_b.norm().item():.4f}")

    # ═══════════ Compare ═════════════════════════════════════════════════
    print("\n=== Path A vs Path B ===")
    metrics(out_a, out_b, "output")
    metrics(dx_a, dx_b, "dx")

    if ds_a is not None:
        print(f"  ds (A only): norm={ds_a.norm().item():.4f}")
    else:
        print(f"  ds: Path A returned None — check topk_scores leaf status")

    # dw1: A=[2I,H,E] interleaved, B=per-expert main_grad [H,2I] split-half
    mg1 = [e.up_gate_proj.weight.main_grad for e in experts_b]
    if mg1[0] is not None:
        s = paddle.stack(mg1)  # [E,H,2I] split-half
        g,u = s[:,:,:I], s[:,:,I:]
        dw1_b = paddle.stack([g,u],axis=3).reshape([E,H,2*I]).transpose([0,2,1]).transpose([1,2,0])
        metrics(dw1_a.cast("float32"), dw1_b, "dw1")

    mg2 = [e.down_proj.weight.main_grad for e in experts_b]
    if mg2[0] is not None:
        dw2_b = paddle.stack(mg2).transpose([2,1,0])  # [H,I,E]
        metrics(dw2_a.cast("float32"), dw2_b, "dw2")

    invalidate_weight_caches(); clear_all_fp8_weight_caches()
    print("\nDONE")


if __name__ == "__main__":
    run()
