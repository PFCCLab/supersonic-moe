"""Small-H token gather-sum kernel is bitwise equal to the BLOCK_K=1 kernel."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def torch():
    paddle = pytest.importorskip("paddle")
    paddle.enable_compat()
    t = pytest.importorskip("torch")
    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    return t


def _case(torch, T, H, max_k, seed):
    torch.manual_seed(seed)
    cnt = torch.randint(0, max_k + 1, (T,), device="cuda")
    cnt[: min(T, 3)] = torch.tensor([0, 1, max_k][: min(T, 3)], device="cuda")
    TK = int(cnt.sum().item())
    off = torch.zeros(T + 1, dtype=torch.int32, device="cuda")
    off[1:] = torch.cumsum(cnt, 0).to(torch.int32)
    perm = torch.randperm(TK, device="cuda").to(torch.int32)
    x = torch.randn(TK, H, device="cuda").to(torch.bfloat16)
    w = torch.rand(TK, device="cuda")
    return x, w, perm, off


@pytest.mark.parametrize("H", [128, 256, 384, 512])
@pytest.mark.parametrize("T", [1, 777, 4096])
@pytest.mark.parametrize("use_w", [False, True])
@pytest.mark.parametrize("skip_single", [False, True])
def test_small_h_bitwise(torch, monkeypatch, H, T, use_w, skip_single):
    monkeypatch.setenv("SONIC_MOE_DETERMINISTIC_AUTOTUNE", "1")
    from sonicmoe.functional import reduction_over_k_gather as R

    assert R._small_h_config(H) is not None
    x, w, perm, off = _case(torch, T, H, 6, T * 1000 + H)
    ww = w if use_w else None
    outs = []
    for flag in ("0", "1"):
        monkeypatch.setenv("SONIC_MOE_TGS_SMALL_H", flag)
        o = torch.full((T, H), 3.0, dtype=torch.bfloat16, device="cuda")
        R.token_gather_and_sum_varlen_K_triton(x, ww, o, perm, off, T, 6, H, True, skip_single)
        outs.append(o)
    assert bool((outs[0].view(torch.int16) == outs[1].view(torch.int16)).all().item())


def test_large_h_uses_default_kernel(monkeypatch):
    monkeypatch.delenv("SONIC_MOE_TGS_SMALL_H", raising=False)
    from sonicmoe.functional import reduction_over_k_gather as R

    assert R._small_h_config(768) is None
    assert R._small_h_config(512) == (1, 512, 2)
    monkeypatch.setenv("SONIC_MOE_TGS_SMALL_H", "0")
    assert R._small_h_config(256) is None
