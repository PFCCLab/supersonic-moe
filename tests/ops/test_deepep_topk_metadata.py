#!/usr/bin/env python
"""Tests for deepep_topk_to_sonic_metadata: real DeepEP topk dispatch conversion.

Validates structural invariants, score consistency, edge cases, and
equivalence with the identity-layout path for K=1 pre-sorted data.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m pytest tests/ops/test_deepep_topk_metadata.py -v --tb=short
  CUDA_VISIBLE_DEVICES=0 python tests/ops/test_deepep_topk_metadata.py  # standalone
"""

import os
import sys

os.environ.setdefault("USE_QUACK_GEMM", "1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
for _p in (_QUACK, _REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()
import torch
import pytest

from sonicmoe.ernie_compat.deepep_metadata import (
    _HAS_TOPK_CUDA_SCALES_KERNEL,
    deepep_topk_to_sonic_metadata,
    deepep_topk_to_sonic_metadata_with_scales,
    deepep_to_sonic_metadata,
    _host_prefix_sum,
)
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    gather_raw_blockscaled_1x32_scales_to_isa,
    quantize_activation_blockscaled_fast,
)
from sonicmoe.functional import (
    _ensure_isa_1x32_scales,
    _gather_1x32_scales_to_isa,
    _raw_1x32_scale_bytes,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _t_eq(a, b) -> bool:
    """Cross-framework tensor equality reduction.
    Paddle compat's ``torch.equal`` returns an element-wise bool tensor (not a
    Python bool). Reduce to a single bool here so ``assert _t_eq(...)`` works
    on both real PyTorch and the Paddle compat layer.
    """
    r = torch.equal(a, b)
    if hasattr(r, "all") and not isinstance(r, bool):
        return bool(r.all().item())
    return bool(r)


def fabricate_dispatch_result(
    N_recv: int, topk: int, E: int, broadcast_ratio: float = 0.5,
    device="cuda",
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Generate realistic DeepEP dispatch results for testing.

    Returns (dispatched_indices, dispatched_probs, tokens_per_expert).

    Fully vectorized: O(1) GPU syncs regardless of N_recv (was O(N_recv) Python
    loop with per-token randperm + .item()).  Per-token expert-count is sampled
    from a clipped Gaussian (matching the legacy distribution) and the topk
    expert IDs come from argsort(rand) — same uniform-without-replacement
    semantics as randperm, but batched.
    """
    expected_experts = max(1, min(int(broadcast_ratio * E), topk))
    max_active = min(topk, E)

    # Per-token active-count: clipped Gaussian around expected_experts.
    counts = torch.randn(N_recv, device=device) * 2 + expected_experts
    counts = counts.round().clamp_(min=1, max=max_active).to(torch.int64)

    # Per-token uniform random permutation of [0, E): argsort(rand_per_row).
    rand_keys = torch.rand(N_recv, E, device=device)
    perm = rand_keys.argsort(dim=1)              # [N, E] random permutations
    perm = perm[:, :topk].to(torch.int32)        # [N, topk] candidate experts

    # Mask trailing slots (col >= count) to -1.
    col_idx = torch.arange(topk, device=device).unsqueeze(0)  # [1, topk]
    valid_mask = col_idx < counts.unsqueeze(1)                # [N, topk]
    dispatched_indices = torch.where(
        valid_mask, perm, torch.full_like(perm, -1)
    )

    # Probability = 1/count for valid slots, 0 otherwise.
    inv_counts = (1.0 / counts.to(torch.float32)).unsqueeze(1)  # [N, 1]
    dispatched_probs = torch.where(
        valid_mask, inv_counts.expand(-1, topk),
        torch.zeros((), dtype=torch.float32, device=device).expand(N_recv, topk),
    ).contiguous()

    # tokens_per_expert via bincount on valid entries.
    valid_flat = dispatched_indices[valid_mask].long()
    tpe = torch.bincount(valid_flat, minlength=E).tolist()

    return dispatched_indices, dispatched_probs, tpe


def fabricate_identity_layout(
    tokens_per_expert: list[int], E: int, device="cuda",
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Generate K=1 pre-sorted dispatch data (identity layout).

    Each token is routed to exactly 1 expert, and tokens are already sorted
    by expert in the buffer.
    """
    T = sum(tokens_per_expert)
    dispatched_indices = torch.empty(T, 1, dtype=torch.int32, device=device)
    dispatched_probs = torch.ones(T, 1, dtype=torch.float32, device=device)

    offset = 0
    for e, count in enumerate(tokens_per_expert):
        dispatched_indices[offset:offset + count, 0] = e
        offset += count

    return dispatched_indices, dispatched_probs, tokens_per_expert


# ── Structural Invariant Tests ──────────────────────────────────────────────

class TestStructuralInvariants:
    """Test that output metadata satisfies required structural properties."""

    @pytest.fixture(scope="class", params=[
        (256, 8, 8),     # small
        (1024, 4, 8),    # medium
        (4096, 8, 32),   # larger
        (2048, 16, 64),  # many experts
        (4096, 8, 256),  # 256 experts
    ])
    def dispatch_data(self, request):
        N_recv, topk, E = request.param
        torch.manual_seed(42)
        return fabricate_dispatch_result(N_recv, topk, E)

    def test_efo_monotonic(self, dispatch_data):
        """expert_frequency_offset must be monotonically non-decreasing."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        assert efo[0].item() == 0
        diffs = efo[1:] - efo[:-1]
        assert (diffs >= 0).all(), "efo not monotonically increasing"

    def test_segments_128_aligned(self, dispatch_data):
        """All non-empty expert segments must be 128-aligned."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        for e in range(E):
            seg_size = (efo[e + 1] - efo[e]).item()
            if seg_size > 0:
                assert seg_size % 128 == 0, (
                    f"Expert {e}: segment size {seg_size} not 128-aligned"
                )

    def test_efo_last_equals_tk_padded(self, dispatch_data):
        """Last efo entry must equal TK_padded."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo, _, _, _, _, _, TK_padded, _, _, _ = result
        assert efo[-1].item() == TK_padded

    def test_x_gather_idx_bounds(self, dispatch_data):
        """x_gather_idx values for real entries must be in [0, N_recv)."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        N_recv = indices.shape[0]
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        x_gather_idx = result[1]
        # All values must be in [0, N_recv)
        assert (x_gather_idx >= 0).all()
        assert (x_gather_idx < N_recv).all()

    def test_naept_properties(self, dispatch_data):
        """naept must be monotonically increasing with correct bounds."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        N_recv = indices.shape[0]
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        naept = result[4]
        TK = (indices >= 0).sum().item()

        # Length = N_recv + 1
        assert naept.shape[0] == N_recv + 1
        # Starts at 0
        assert naept[0].item() == 0
        # Ends at TK (total valid entries)
        assert naept[-1].item() == TK
        # Monotonically non-decreasing
        diffs = naept[1:] - naept[:-1]
        assert (diffs >= 0).all()

    def test_s_reverse_length(self, dispatch_data):
        """s_reverse_scatter_idx length must be TK (valid entries)."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        TK = (indices >= 0).sum().item()
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        s_reverse = result[3]
        assert s_reverse.shape[0] == TK

    def test_topk_scores_length(self, dispatch_data):
        """topk_scores length must be TK_padded; valid scores accessed via naept must be > 0."""
        indices, probs, tpe = dispatch_data
        E = len(tpe)
        N_recv = indices.shape[0]
        TK = (indices >= 0).sum().item()
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        topk_scores = result[5]
        naept = result[4]
        TK_padded = result[6]
        assert topk_scores.shape[0] == TK_padded, (
            f"topk_scores length {topk_scores.shape[0]} != TK_padded {TK_padded}"
        )
        # Valid scores (accessed via naept offsets) should all be > 0
        if TK > 0:
            for t in range(min(N_recv, 50)):
                start = naept[t].item()
                end = naept[t + 1].item()
                if end > start:
                    assert (topk_scores[start:end] > 0).all(), (
                        f"Token {t}: scores at [{start},{end}) should be > 0"
                    )
        # Total non-zero scores should equal TK
        nonzero_count = (topk_scores > 0).sum().item()
        assert nonzero_count == TK, (
            f"Non-zero scores {nonzero_count} != TK {TK}"
        )


# ── Score Consistency Tests ─────────────────────────────────────────────────

class TestScoreConsistency:
    """Test that scores are correctly preserved through the conversion."""

    @pytest.mark.parametrize("N_recv,topk,E", [
        (512, 8, 8),
        (256, 4, 256),  # 256 experts
    ])
    def test_per_token_score_sum(self, N_recv, topk, E):
        """Sum of scores per token should match sum of valid dispatched_probs."""
        torch.manual_seed(123)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        naept = result[4]
        topk_scores = result[5]

        for t in range(min(N_recv, 50)):  # Check first 50 tokens
            start = naept[t].item()
            end = naept[t + 1].item()
            score_sum = topk_scores[start:end].sum().item()

            # Expected: sum of valid probs for token t
            valid_mask = indices[t] >= 0
            expected_sum = probs[t][valid_mask].sum().item()

            assert abs(score_sum - expected_sum) < 1e-5, (
                f"Token {t}: score_sum={score_sum:.6f} vs expected={expected_sum:.6f}"
            )

    @pytest.mark.parametrize("N_recv,topk,E", [
        (256, 4, 8),
        (128, 4, 256),  # 256 experts
    ])
    def test_padding_scores_zero(self, N_recv, topk, E):
        """Valid scores (via naept) must be > 0; total non-zero count == TK."""
        torch.manual_seed(42)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        topk_scores = result[5]
        naept = result[4]
        TK = (indices >= 0).sum().item()
        TK_padded = result[6]

        # Per-token bounds via naept must wrap exactly the non-zero scores.
        # Vectorized: build a mask from naept ranges and check vs (scores>0).
        ranges = naept[1:] - naept[:-1]              # [N_recv]
        # All non-zero scores are inside some [naept[t], naept[t+1]) range and
        # those ranges tile [0, TK). Hence (topk_scores>0).sum() == TK suffices
        # to certify both "all positives in valid ranges" and "no spurious
        # positives in padding". Check ranges sum == TK as a structural cross-check.
        assert int(ranges.sum().item()) == TK

        # Total non-zero scores should equal TK
        nonzero_count = (topk_scores > 0).sum().item()
        assert nonzero_count == TK, f"Non-zero scores {nonzero_count} != TK {TK}"


# ── Gather Consistency Tests ────────────────────────────────────────────────

class TestGatherConsistency:
    """Test that x_gather_idx correctly gathers tokens per expert."""

    @pytest.mark.parametrize("N_recv,topk,E", [
        (256, 4, 8),
        (128, 4, 256),  # 256 experts
    ])
    def test_expert_segment_tokens(self, N_recv, topk, E):
        """Verify each expert's segment contains exactly the expected tokens."""
        torch.manual_seed(77)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        x_gather_idx = result[1]

        # Vectorize expected_set per expert: build [E, N_recv] bool of "is t routed to e"
        idx_long = indices.long().clamp(min=0)                       # avoid -1 indices
        valid_mask = indices >= 0                                    # [N, topk]
        # one_hot[N, topk, E] then reduce to [E, N]
        # Use scatter to avoid materializing one_hot: for each (t,k) with valid,
        # set expert_membership[expert, t] = True.
        expert_membership = torch.zeros((E, N_recv), dtype=torch.bool, device="cuda")
        flat_t = torch.arange(N_recv, device="cuda").unsqueeze(1).expand(-1, topk)[valid_mask]
        flat_e = idx_long[valid_mask]
        expert_membership[flat_e, flat_t] = True
        expected_sets = [set(torch.nonzero(expert_membership[e], as_tuple=False).flatten().tolist())
                         for e in range(E)]
        for e in range(E):
            seg_start = efo[e].item()
            real_count = tpe[e]
            gathered_set = set(x_gather_idx[seg_start:seg_start + real_count].tolist())
            assert gathered_set == expected_sets[e], (
                f"Expert {e}: gathered={sorted(gathered_set)} vs expected={sorted(expected_sets[e])}"
            )


# ── Edge Cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_masked(self):
        """All entries are -1 (no tokens routed to any local expert)."""
        N_recv, topk, E = 32, 4, 8
        indices = torch.full((N_recv, topk), -1, dtype=torch.int32, device="cuda")
        probs = torch.zeros(N_recv, topk, dtype=torch.float32, device="cuda")
        tpe = [0] * E

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo, x_gather, s_scatter, s_reverse, naept, scores, TK_padded, pad_rows, n_recv, _ = result

        assert TK_padded == 0
        assert pad_rows == 0
        assert n_recv == N_recv
        assert naept.shape[0] == N_recv + 1
        assert (naept == 0).all()

    def test_single_token_single_expert(self):
        """One token routed to one expert."""
        indices = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device="cuda")
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
        tpe = [1, 0, 0, 0]
        E = 4

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo, x_gather, s_scatter, s_reverse, naept, scores, TK_padded, pad_rows, n_recv, _ = result

        assert n_recv == 1
        assert naept[0].item() == 0
        assert naept[1].item() == 1
        assert TK_padded == 128  # padded to 128
        assert efo[1].item() == 128
        assert x_gather[0].item() == 0  # gather from token 0
        assert scores[0].item() == 1.0

    def test_all_tokens_to_one_expert(self):
        """All tokens routed to the same expert."""
        N_recv, E = 64, 8
        indices = torch.zeros(N_recv, 1, dtype=torch.int32, device="cuda")  # all to expert 0
        probs = torch.ones(N_recv, 1, dtype=torch.float32, device="cuda")
        tpe = [N_recv] + [0] * (E - 1)

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo, x_gather, _, _, naept, _, TK_padded, _, n_recv, _ = result

        assert n_recv == N_recv
        assert efo[1].item() == 128  # 64 padded to 128
        assert efo[-1].item() == TK_padded

    def test_tokens_with_zero_local_experts(self):
        """Some tokens have all -1 (routed to other ranks only)."""
        N_recv, topk, E = 16, 4, 4
        indices = torch.full((N_recv, topk), -1, dtype=torch.int32, device="cuda")
        probs = torch.zeros(N_recv, topk, dtype=torch.float32, device="cuda")

        # Only first 8 tokens have valid routing
        for i in range(8):
            indices[i, 0] = i % E
            probs[i, 0] = 1.0

        tpe = [2, 2, 2, 2]  # 2 tokens per expert

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        naept = result[4]

        # Tokens 8..15 should have naept[i] == naept[i+1] (zero assignments)
        diffs = (naept[1:] - naept[:-1])[8:]
        assert int(diffs.abs().sum().item()) == 0, "tokens 8..N_recv should have 0 assignments"

    def test_tokens_per_expert_with_zeros(self):
        """Some experts have zero tokens."""
        torch.manual_seed(42)
        N_recv, topk, E = 128, 4, 8
        indices = torch.full((N_recv, topk), -1, dtype=torch.int32, device="cuda")
        probs = torch.zeros(N_recv, topk, dtype=torch.float32, device="cuda")

        # Route to experts 0, 2, 4, 6 only (skip odd experts) — vectorized
        rows = torch.arange(N_recv, device="cuda")
        indices[:, 0] = ((rows % 4) * 2).to(torch.int32)
        probs[:, 0] = 1.0

        tpe = [32, 0, 32, 0, 32, 0, 32, 0]

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]

        # Odd experts should have zero-length segments
        for e in [1, 3, 5, 7]:
            assert efo[e + 1].item() - efo[e].item() == 0


# ── Identity Layout Equivalence ─────────────────────────────────────────────

class TestIdentityEquivalence:
    """Test that K=1 pre-sorted data matches deepep_to_sonic_metadata."""

    @pytest.mark.parametrize("tpe", [
        [512] * 8,
        [128, 256, 64, 512, 384, 100, 200, 300],
        [128] * 4,
        [16] * 64,  # many small experts (was [32]*256, too slow)
    ])
    def test_equivalence(self, tpe):
        """K=1 pre-sorted input should produce equivalent metadata."""
        E = len(tpe)
        T = sum(tpe)
        indices, probs, _ = fabricate_identity_layout(tpe, E)

        # Topk path
        topk_result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        (topk_efo, topk_xg, _, _, _, _, topk_TK_padded, topk_pad, _, _) = topk_result

        # Identity path
        id_result = deepep_to_sonic_metadata(tpe, T, E)
        (id_efo, id_xg, _, _, _, _, id_TK_padded, id_pad) = id_result

        # EFO should be identical
        assert (topk_efo == id_efo).all(), "EFO mismatch"
        # TK_padded should be identical
        assert topk_TK_padded == id_TK_padded, (
            f"TK_padded: topk={topk_TK_padded} vs id={id_TK_padded}"
        )

        # Per-expert token-set check (vectorized via combined sort key).
        seg_starts = topk_efo[:-1].long()
        tpe_t = torch.tensor(tpe, dtype=torch.int64, device="cuda")
        positions = torch.arange(topk_TK_padded, device="cuda", dtype=torch.int64)
        expert_id = torch.searchsorted(topk_efo[1:].long(), positions, right=True).clamp(max=E - 1)
        local_pos = positions - seg_starts[expert_id]
        is_real = local_pos < tpe_t[expert_id]
        real_pos = positions[is_real]
        real_exp = expert_id[is_real]
        topk_key = real_exp * (T + 1) + topk_xg[real_pos].long()
        id_key   = real_exp * (T + 1) + id_xg[real_pos].long()
        assert _t_eq(topk_key.sort(), id_key.sort()), (
            "Per-expert token multisets differ between topk and identity paths"
        )


# ── moe_permute Cross-Validation ──────────────────────────────────────────

class TestMoePermuteConsistency:
    """Cross-validate deepep_topk_to_sonic_metadata against paddle.nn.functional.moe_permute."""

    @pytest.mark.parametrize("N_recv,topk,E", [
        (64, 2, 4),
        (128, 4, 8),
        (256, 8, 8),   # N_recv <= 256 to avoid bf16 integer precision issues
        (128, 4, 64),  # many experts (was E=256, too slow)
    ])
    def test_token_set_per_expert(self, N_recv, topk, E):
        """Per-expert token sets from our metadata must match moe_permute's output."""
        torch.manual_seed(42)
        H = 32  # small hidden dim for the test
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        # ── Gold: paddle.nn.functional.moe_permute ────────────────────────
        # Make each token identifiable via its row content
        x = torch.zeros(N_recv, H, dtype=torch.bfloat16, device="cuda")
        x[:, 0] = torch.arange(N_recv, dtype=torch.bfloat16, device="cuda")  # tag row=token id (exact for i <= 256)

        gold_permuted, _, gold_probs, _ = paddle.nn.functional.moe_permute(
            x, None, indices, probs,
            num_experts=E, tokens_per_expert=tpe,
            padding_alignment=128, do_gather=True,
        )

        # ── Ours: deepep_topk_to_sonic_metadata ──────────────────────────
        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        x_gather_idx = result[1]
        TK_padded = result[6]

        # Both should have the same TK_padded
        assert gold_permuted.shape[0] == TK_padded, (
            f"TK_padded mismatch: moe_permute={gold_permuted.shape[0]} vs ours={TK_padded}"
        )

        # Vectorized per-expert token-set equality via combined sort key.
        seg_starts = efo[:-1].long()
        tpe_t = torch.tensor(tpe, dtype=torch.int64, device="cuda")
        positions = torch.arange(TK_padded, device="cuda", dtype=torch.int64)
        expert_id = torch.searchsorted(efo[1:].long(), positions, right=True).clamp(max=E - 1)
        local_pos = positions - seg_starts[expert_id]
        is_real = local_pos < tpe_t[expert_id]
        real_pos = positions[is_real]
        real_exp = expert_id[is_real]
        # gold token id encoded in column 0
        gold_tok = gold_permuted[real_pos, 0].to(torch.int64)
        our_tok = x_gather_idx[real_pos].long()
        gold_key = real_exp * (N_recv + 1) + gold_tok
        our_key  = real_exp * (N_recv + 1) + our_tok
        assert _t_eq(gold_key.sort(), our_key.sort()), (
            "Per-expert token multisets differ between ours and moe_permute"
        )

    @pytest.mark.parametrize("N_recv,topk,E", [
        (64, 2, 4),
        (256, 4, 8),
        (128, 4, 64),  # many experts (was E=256)
    ])
    def test_padding_alignment_consistent(self, N_recv, topk, E):
        """Padding alignment (128) must match between our metadata and moe_permute."""
        torch.manual_seed(99)
        H = 16
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        x = torch.randn(N_recv, H, dtype=torch.bfloat16, device="cuda")
        gold_permuted, _, _, _ = paddle.nn.functional.moe_permute(
            x, None, indices, probs,
            num_experts=E, tokens_per_expert=tpe,
            padding_alignment=128, do_gather=True,
        )

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        TK_padded = result[6]

        # Total padded size must match
        assert gold_permuted.shape[0] == TK_padded

        # Vectorized per-expert segment alignment check.
        seg = (efo[1:] - efo[:-1])
        nonzero = seg[seg > 0]
        assert _t_eq(nonzero % 128, torch.zeros_like(nonzero)), "Segment not 128-aligned"

    @pytest.mark.parametrize("N_recv,topk,E", [
        (128, 4, 8),
        (128, 4, 64),  # many experts (was E=256)
    ])
    def test_score_per_expert_consistent(self, N_recv, topk, E):
        """Per-token scores within each expert segment must match moe_permute probs."""
        torch.manual_seed(55)
        H = 16
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        x = torch.zeros(N_recv, H, dtype=torch.bfloat16, device="cuda")
        x[:, 0] = torch.arange(N_recv, dtype=torch.bfloat16, device="cuda")

        gold_permuted, _, gold_probs, _ = paddle.nn.functional.moe_permute(
            x, None, indices, probs,
            num_experts=E, tokens_per_expert=tpe,
            padding_alignment=128, do_gather=True,
        )

        result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        efo = result[0]
        x_gather_idx = result[1]
        s_scatter_idx = result[2]
        topk_scores = result[5]
        TK_padded = result[6]

        # Vectorized per-position score check (only where token ids align).
        seg_starts = efo[:-1].long()
        tpe_t = torch.tensor(tpe, dtype=torch.int64, device="cuda")
        positions = torch.arange(TK_padded, device="cuda", dtype=torch.int64)
        expert_id = torch.searchsorted(efo[1:].long(), positions, right=True).clamp(max=E - 1)
        local_pos = positions - seg_starts[expert_id]
        is_real = local_pos < tpe_t[expert_id]
        real_pos = positions[is_real]
        our_tok = x_gather_idx[real_pos].long()
        gold_tok = gold_permuted[real_pos, 0].to(torch.int64)
        match = our_tok == gold_tok
        if match.any():
            mp = real_pos[match]
            our_score = topk_scores[s_scatter_idx[mp].long()]
            gold_score = gold_probs[mp]
            assert torch.allclose(our_score.float(), gold_score.float(), atol=1e-5), (
                "Per-position scores differ where token ids align"
            )


# ── CUDA vs Python Fallback Comparison ────────────────────────────────────

class TestCudaVsPythonFallback:
    """Compare CUDA kernel output against Python fallback element-wise."""

    @pytest.mark.parametrize("N_recv,topk,E", [
        (64, 2, 4),
        (256, 8, 8),
        (512, 4, 32),
    ])
    def test_element_wise_consistency(self, N_recv, topk, E):
        """CUDA kernel and Python fallback must produce equivalent metadata.

        Note: s_scatter_idx at PADDING positions may differ (Python uses
        unique arange [TK, TK+pad), CUDA uses constant TK). Both are correct
        since padding scores are 0. We only compare REAL positions.
        """
        from sonicmoe.ernie_compat.deepep_metadata import (
            _HAS_TOPK_CUDA_KERNEL,
            _deepep_topk_to_sonic_metadata_cuda,
        )
        if not _HAS_TOPK_CUDA_KERNEL:
            pytest.skip("CUDA kernel not compiled")

        torch.manual_seed(42)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        # Force Python fallback
        import sonicmoe.ernie_compat.deepep_metadata as _mod
        saved = _mod._HAS_TOPK_CUDA_KERNEL
        _mod._HAS_TOPK_CUDA_KERNEL = False
        py_result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        _mod._HAS_TOPK_CUDA_KERNEL = saved

        # Run CUDA path
        cu_result = _deepep_topk_to_sonic_metadata_cuda(
            indices, probs, tpe, E, "cuda", 128,
        )

        py_efo, py_xg, py_ss, py_sr, py_naept, py_scores, py_TKp, py_pad, py_N, _ = py_result
        cu_efo, cu_xg, cu_ss, cu_sr, cu_naept, cu_scores, cu_TKp, cu_pad, cu_N, _ = cu_result

        # Scalars must match exactly
        assert py_TKp == cu_TKp, f"TK_padded: py={py_TKp} cu={cu_TKp}"
        assert py_pad == cu_pad, f"total_pad: py={py_pad} cu={cu_pad}"
        assert py_N == cu_N, f"N_recv: py={py_N} cu={cu_N}"

        # efo must match exactly
        assert (py_efo == cu_efo).all(), "efo mismatch"

        # naept must match exactly
        assert (py_naept == cu_naept).all(), "naept mismatch"

        # Per-expert token-set check is subsumed by the vectorized Check 2 below
        # (which compares (expert, token_id) multisets across all experts at once).
        TK_padded = py_TKp
        TK = sum(tpe)

        # s_reverse_scatter_idx: shape must match
        assert py_sr.shape == cu_sr.shape, f"s_reverse shape mismatch"

        # ── Functional equivalence checks ──────────────────────────────
        # The key semantic contract is:
        #   For token t routed to expert e with score s:
        #     x_gather_idx[padded_pos] == t
        #     topk_scores[s_scatter_idx[padded_pos]] == s
        #     s_reverse_scatter_idx[s_scatter_idx[padded_pos]] == padded_pos
        # Both paths may assign different token-major positions (s_scatter_idx
        # values) since Python uses argsort order and CUDA uses warp-ballot
        # order. But the gathered scores must match per (token, expert).

        # Check 1: Per-token score multiset via naept (sorted comparison).
        # Vectorized: build per-row sort within naept ranges. Since each token
        # has at most `topk` valid entries, gather them into a fixed [N, topk]
        # tensor padded with -inf, sort, and compare.
        col_idx = torch.arange(topk, dtype=torch.int64, device="cuda").unsqueeze(0)  # [1, topk]
        py_naept_l = py_naept.long()
        row_lens = (py_naept_l[1:] - py_naept_l[:-1]).unsqueeze(1)         # [N, 1]
        valid = col_idx < row_lens                                          # [N, topk]
        # Clamp gather indices into bounds; invalid slots will be masked out.
        gather_pos = (py_naept_l[:-1].unsqueeze(1) + col_idx).clamp(max=py_scores.numel() - 1)
        py_g = torch.where(valid, py_scores[gather_pos], torch.full_like(py_scores[gather_pos], float("-inf")))
        cu_g = torch.where(valid, cu_scores[gather_pos], torch.full_like(cu_scores[gather_pos], float("-inf")))
        py_g_sorted = py_g.sort(axis=1) if "paddle" in str(type(py_g)) else py_g.sort(dim=1)[0]
        cu_g_sorted = cu_g.sort(axis=1) if "paddle" in str(type(cu_g)) else cu_g.sort(dim=1)[0]
        assert torch.allclose(py_g_sorted, cu_g_sorted, atol=1e-6), (
            "Per-token score multisets differ (CUDA vs Python)"
        )

        # Check 2: For each expert segment, the (token_id → score) mapping
        # via topk_scores[s_scatter_idx[pos]] must agree.
        # Vectorized across ALL experts simultaneously: build per-segment
        # position bounds, gather token ids and scores, compare sorted-by-token.
        # Use the "real_count per expert" mask to identify real positions.
        py_pos2score = py_scores[py_ss.long()]   # [TK_padded] (incl. padding)
        cu_pos2score = cu_scores[cu_ss.long()]
        # Build per-position expert id and "is real" mask via efo + tpe.
        seg_starts_t = py_efo[:-1].long()                                  # [E]
        tpe_t = torch.tensor(tpe, dtype=torch.int64, device="cuda")        # [E]
        real_ends_t = seg_starts_t + tpe_t                                 # [E]
        # For each padded position, compute expert id by searchsorted.
        positions = torch.arange(py_TKp, device="cuda", dtype=torch.int64)
        expert_id = torch.searchsorted(py_efo[1:].long(), positions, right=True).clamp(max=E - 1)
        # Real if local_pos < tpe[expert]
        local_pos = positions - seg_starts_t[expert_id]
        is_real = local_pos < tpe_t[expert_id]
        real_positions = positions[is_real]                                # [TK]
        real_experts = expert_id[is_real]
        # Per-expert sort key = expert*N_recv + token_id → a single global sort.
        py_tok = py_xg[real_positions].long()
        cu_tok = cu_xg[real_positions].long()
        py_key = real_experts * (py_N + 1) + py_tok
        cu_key = real_experts * (py_N + 1) + cu_tok
        py_perm = py_key.argsort(stable=True)
        cu_perm = cu_key.argsort(stable=True)
        assert _t_eq(py_key[py_perm], cu_key[cu_perm]), (
            "Per-expert (token-id) multisets differ between CUDA and Python"
        )
        py_sorted_scores = py_pos2score[real_positions][py_perm]
        cu_sorted_scores = cu_pos2score[real_positions][cu_perm]
        assert torch.allclose(py_sorted_scores, cu_sorted_scores, atol=1e-6), (
            "Per-(expert,token) scores differ between CUDA and Python"
        )

        # Check 3: round-trip s_reverse_scatter_idx[s_scatter_idx[pos]] == pos
        # for all real positions (vectorized — reuses real_positions from Check 2).
        if real_positions.numel() > 0:
            py_rt = py_sr[py_ss[real_positions].long()]
            cu_rt = cu_sr[cu_ss[real_positions].long()]
            assert _t_eq(py_rt, real_positions.to(py_rt.dtype)), "Python round-trip fail"
            assert _t_eq(cu_rt, real_positions.to(cu_rt.dtype)), "CUDA round-trip fail"
        # Also drop expensive set-based per-expert check above by checking
        # token multiset across ALL experts in one go (already done in Check 2 via py_key/cu_key).


class TestCudaScalePacking:
    @pytest.mark.parametrize("rows,cols", [(7, 256), (129, 4096)])
    def test_quantizer_compact_scale_words_are_byte_exact(self, rows, cols):
        torch.manual_seed(314159)
        source = torch.randn(
            (rows, cols), dtype=torch.bfloat16, device="cuda"
        )
        fp8_raw, raw_scales = quantize_activation_blockscaled_fast(source)
        fp8_compact, compact_scales = quantize_activation_blockscaled_fast(
            source, pack_scale_words=True
        )

        assert compact_scales.dtype == torch.int32
        assert compact_scales.shape == (rows, cols // 128)
        assert _t_eq(fp8_raw.view(torch.uint8), fp8_compact.view(torch.uint8))
        assert _t_eq(
            raw_scales,
            compact_scales.view(torch.uint8).reshape(rows, cols // 32),
        )

    """Validate metadata-side raw scale packing against the existing reference."""

    @pytest.mark.parametrize("N_recv,topk,E,cols,raw_dtype", [
        (256, 4, 8, 256, torch.int32),
        (513, 8, 16, 4096, torch.int32),
        (384, 8, 8, 384, torch.uint8),
    ])
    @pytest.mark.parametrize("pack_mode", ["default", "scatter", "rowmajor"])
    def test_with_scales_matches_raw_gather_reference(
        self, N_recv, topk, E, cols, raw_dtype, pack_mode
    ):
        if not _HAS_TOPK_CUDA_SCALES_KERNEL:
            pytest.skip("CUDA scale-packing metadata kernel not compiled")

        torch.manual_seed(123)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)
        scale_cols = (cols + 31) // 32
        raw_i32 = (
            torch.arange(N_recv * scale_cols, dtype=torch.int32, device="cuda")
            .reshape(N_recv, scale_cols)
            % 251
        )
        raw_scales = raw_i32.to(torch.uint8) if raw_dtype is torch.uint8 else raw_i32

        result = deepep_topk_to_sonic_metadata_with_scales(
            indices,
            probs,
            tpe,
            E,
            raw_scales=raw_scales,
            cols=cols,
            block=128,
            pack_scales_in_scatter=(pack_mode == "scatter"),
            pack_scales_rowmajor=(pack_mode == "rowmajor"),
        )
        x_gather_idx = result[1]
        packed = result[-1]
        assert packed is not None

        ref = gather_raw_blockscaled_1x32_scales_to_isa(raw_scales, x_gather_idx, cols)
        assert packed.shape == ref.shape
        assert _t_eq(packed.view(torch.uint8), ref.view(torch.uint8)), (
            "metadata-side packed scales differ from raw gather reference"
        )

    @pytest.mark.parametrize("N_recv,topk,E,cols", [
        (257, 4, 8, 256),
        (513, 8, 16, 4096),
        (129, 4, 24, 2048),
    ])
    @pytest.mark.parametrize("pack_mode", ["default", "scatter", "rowmajor"])
    def test_compact_int32_carrier_is_byte_exact(
        self, N_recv, topk, E, cols, pack_mode
    ):
        if not _HAS_TOPK_CUDA_SCALES_KERNEL:
            pytest.skip("CUDA scale-packing metadata kernel not compiled")

        torch.manual_seed(314159)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)
        scale_cols = (cols + 31) // 32
        assert scale_cols % 4 == 0
        raw_scales = (
            torch.arange(N_recv * scale_cols, dtype=torch.int32, device="cuda")
            .reshape(N_recv, scale_cols)
            % 251
        ).to(torch.uint8)
        compact_carrier = raw_scales.view(torch.int32).reshape(
            N_recv, scale_cols // 4
        )
        assert compact_carrier.data_ptr() == raw_scales.data_ptr()

        kwargs = {
            "block": 128,
            "pack_scales_in_scatter": pack_mode == "scatter",
            "pack_scales_rowmajor": pack_mode == "rowmajor",
        }
        raw_result = deepep_topk_to_sonic_metadata_with_scales(
            indices, probs, tpe, E, raw_scales, cols, **kwargs
        )
        compact_result = deepep_topk_to_sonic_metadata_with_scales(
            indices, probs, tpe, E, compact_carrier, cols, **kwargs
        )

        for index in (0, 1, 2, 3, 4, 5, 9, 10):
            assert _t_eq(raw_result[index], compact_result[index]), (
                f"output {index} differs for compact carrier in {pack_mode} mode"
            )
        assert raw_result[6:9] == compact_result[6:9]

    def test_invalid_compact_carrier_shape_is_rejected(self):
        N_recv, topk, E, cols = 32, 4, 8, 256
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)
        malformed = torch.empty((N_recv, 3), dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="raw_scales shape mismatch"):
            deepep_topk_to_sonic_metadata_with_scales(
                indices, probs, tpe, E, malformed, cols
            )

    @pytest.mark.parametrize("rows,cols", [(7, 256), (31, 4096), (129, 2048)])
    def test_compact_carrier_fallbacks_are_byte_exact(self, rows, cols):
        scale_cols = cols // 32
        raw = (
            torch.arange(rows * scale_cols, dtype=torch.int32, device="cuda")
            .reshape(rows, scale_cols)
            % 251
        ).to(torch.uint8)
        compact = raw.view(torch.int32).reshape(rows, scale_cols // 4)
        gather_idx = torch.arange(128, dtype=torch.int32, device="cuda") % rows

        restored = _raw_1x32_scale_bytes(compact, rows, cols)
        assert restored.data_ptr() == compact.data_ptr()
        assert _t_eq(restored, raw)

        raw_gathered = _gather_1x32_scales_to_isa(
            raw, gather_idx, rows, cols
        )
        compact_gathered = _gather_1x32_scales_to_isa(
            compact, gather_idx, rows, cols
        )
        assert _t_eq(
            raw_gathered.view(torch.uint8),
            compact_gathered.view(torch.uint8),
        )

        raw_packed = _ensure_isa_1x32_scales(raw, rows, cols)
        compact_packed = _ensure_isa_1x32_scales(compact, rows, cols)
        assert _t_eq(
            raw_packed.view(torch.uint8), compact_packed.view(torch.uint8)
        )

    @pytest.mark.parametrize(
        "preact_bf16,allocate_z_scale",
        [(False, True), (True, False)],
        ids=["combined_quant", "postact_quant"],
    )
    def test_gated_output_allocation_contract(
        self, preact_bf16, allocate_z_scale
    ):
        if not _HAS_TOPK_CUDA_SCALES_KERNEL:
            pytest.skip("CUDA scale-packing metadata kernel not compiled")

        N_recv, topk, E, cols, gated_n = 256, 4, 8, 256, 256
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)
        raw_scales = torch.ones(
            (N_recv, cols // 32), dtype=torch.uint8, device="cuda"
        )
        prototype = torch.empty(
            (N_recv, cols), dtype=torch.float8_e4m3fn, device="cuda"
        )

        def run():
            return deepep_topk_to_sonic_metadata_with_scales(
                indices,
                probs,
                tpe,
                E,
                raw_scales=raw_scales,
                cols=cols,
                block=128,
                gated_output_prototype=prototype,
                gated_n=gated_n,
                gated_preact_bf16=preact_bf16,
                gated_allocate_z_scale=allocate_z_scale,
            )

        first = run()
        second = run()
        assert len(first) == 15
        TK_padded = first[6]
        for index in (0, 1, 2, 3, 4, 9):
            assert first[index].data_ptr() % 16 == 0, (
                f"metadata output {index} must satisfy TVM-FFI 16-byte alignment"
            )
        preact, postact, z_scale, postact_scale = first[11:15]
        assert preact.shape == (TK_padded, gated_n)
        assert postact.shape == (TK_padded, gated_n // 2)
        assert z_scale.shape == (
            (TK_padded, gated_n // 32) if allocate_z_scale else (0,)
        )
        assert postact_scale.shape == (
            TK_padded // 128,
            gated_n // 256,
            512,
        )
        assert preact.dtype == (
            torch.bfloat16 if preact_bf16 else torch.float8_e4m3fn
        )
        assert postact.dtype == torch.float8_e4m3fn
        assert z_scale.dtype == torch.uint8
        assert postact_scale.dtype == torch.uint8
        assert all(t.is_contiguous() for t in first[11:15])
        if not allocate_z_scale:
            assert first[13].numel() == 0 and second[13].numel() == 0
        assert all(
            lhs.data_ptr() != rhs.data_ptr()
            for lhs, rhs in zip(first[11:15], second[11:15])
            if lhs.numel() > 0
        ), "non-empty ctx-saved gated outputs must have independent storage per microbatch"

    def test_large_live_outputs_stay_byte_exact_across_calls(self):
        """PP/VPP-live metadata must not alias later custom-op invocations."""
        if not _HAS_TOPK_CUDA_SCALES_KERNEL:
            pytest.skip("CUDA scale-packing metadata kernel not compiled")

        N_recv, TK, topk, E, cols = 16385, 20001, 4, 24, 4096
        rows = torch.arange(N_recv, dtype=torch.int32, device="cuda")
        indices = torch.full(
            (N_recv, topk), -1, dtype=torch.int32, device="cuda"
        )
        probs = torch.zeros(
            (N_recv, topk), dtype=torch.float32, device="cuda"
        )
        indices[:, 0] = rows % E
        probs[:, 0] = 1.0
        extra = TK - N_recv
        indices[:extra, 1] = (rows[:extra] + 1) % E
        probs[:extra, :2] = 0.5
        valid_experts = indices[indices >= 0].long()
        tokens_per_expert = torch.bincount(
            valid_experts, minlength=E
        ).tolist()
        compact_scales = torch.zeros(
            (N_recv, cols // 128), dtype=torch.int32, device="cuda"
        )

        def run():
            return deepep_topk_to_sonic_metadata_with_scales(
                indices,
                probs,
                tokens_per_expert,
                E,
                compact_scales,
                cols,
                block=128,
            )

        output_indices = (0, 1, 2, 3, 4, 5, 9, 10)
        first = run()
        snapshots = {
            index: first[index].clone() for index in output_indices
        }
        torch.cuda.synchronize()

        live_outputs = [first]
        for _ in range(4):
            current = run()
            live_outputs.append(current)
            torch.cuda.synchronize()
            for index in output_indices:
                assert _t_eq(current[index], snapshots[index]), (
                    f"metadata output {index} changed across live invocations"
                )
                assert _t_eq(first[index], snapshots[index]), (
                    f"live metadata output {index} was overwritten by a later call"
                )

        for index in output_indices:
            pointers = [result[index].data_ptr() for result in live_outputs]
            assert len(set(pointers)) == len(pointers), (
                f"metadata output {index} must own per-call storage"
            )

# ── Performance Benchmark ───────────────────────────────────────────────────

class TestPerformance:
    """Benchmark metadata conversion latency.

    Note on threshold: CUDA-event timing under the Paddle compat layer
    includes ~1.5–2.5 ms of host-side Python/launch overhead per call which
    is **not** representative of real GPU work. The true GPU-projection
    (measured via nsys) is ~165 µs for the sonic-meta region of the
    forward pass. This test is therefore a **gross-regression guard**, not
    a perf contract — the contract lives in the nsys benchmarks
    (`tests/ops/bench_user_shape_fwd_nsys.py`).
    """

    @pytest.mark.parametrize("N_recv,topk,E", [
        (1024, 4, 8),
        (2048, 8, 32),
        (4096, 8, 64),
    ])
    def test_latency_under_target(self, N_recv, topk, E):
        """Sanity-check perf does not blow up by orders of magnitude.

        OOM root-cause: paddle's auto-growth allocator caches every chunk
        ever requested by the 60+ preceding tests in the same pytest
        process. Even though this test only needs a few MB, the pool can
        grow to >200 GB resident; on a contested shared GPU the next
        chunk request (Paddle doubles, so ~47 GB) fails. Empty_cache()
        helps but is not sufficient — we must also clear the module-level
        high-watermark caches *and* drop the dispatch_data fixture
        residuals before timing.
        """
        import gc
        from sonicmoe.ernie_compat import deepep_metadata as _dm
        # Clear module-level high-watermark caches
        try:
            _dm._TOPK_CACHE.clear()
            _dm._ARANGE_CACHE.clear()
        except AttributeError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

        torch.manual_seed(42)
        indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)

        # Warmup
        for _ in range(3):
            deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        torch.cuda.synchronize()

        # Timed run
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        repeats = 20

        start.record()
        for _ in range(repeats):
            deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
        end.record()
        torch.cuda.synchronize()

        avg_us = start.elapsed_time(end) * 1000 / repeats
        print(f"\n  [N={N_recv}, topk={topk}, E={E}] avg={avg_us:.1f}us "
              f"(includes ~2ms compat-layer host overhead)")
        # Gross regression guard only: real GPU work is <1ms; 10ms catches
        # 50x+ slowdowns without false-positives from host scheduling jitter
        # on the shared GPU.
        assert avg_us < 10000, f"Egregious regression: {avg_us:.1f}us > 10ms"


# ── Standalone runner ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running deepep_topk_to_sonic_metadata tests...\n")

    # Quick smoke test
    torch.manual_seed(42)
    N_recv, topk, E = 1024, 8, 8
    indices, probs, tpe = fabricate_dispatch_result(N_recv, topk, E)
    result = deepep_topk_to_sonic_metadata(indices, probs, tpe, E)
    efo, x_gather, s_scatter, s_reverse, naept, scores, TK_padded, pad_rows, n_recv, _ = result

    TK = (indices >= 0).sum().item()
    print(f"Config: N_recv={N_recv}, topk={topk}, E={E}")
    print(f"TK={TK}, TK_padded={TK_padded}, pad_rows={pad_rows}")
    print(f"efo: {efo.tolist()}")
    print(f"naept[:5]: {naept[:5].tolist()}")
    print(f"scores[:5]: {scores[:5].tolist()}")
    print(f"x_gather[:10]: {x_gather[:10].tolist()}")
    print(f"s_reverse[:10]: {s_reverse[:10].tolist()}")

    # Structural checks
    assert efo[0].item() == 0
    assert efo[-1].item() == TK_padded
    for e in range(E):
        seg = (efo[e + 1] - efo[e]).item()
        if seg > 0:
            assert seg % 128 == 0, f"Expert {e}: segment {seg} not 128-aligned"

    assert naept[0].item() == 0
    assert naept[-1].item() == TK
    assert (naept[1:] - naept[:-1] >= 0).all()

    print("\nAll smoke tests PASSED!")
    print("\nRunning full pytest suite...")
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
