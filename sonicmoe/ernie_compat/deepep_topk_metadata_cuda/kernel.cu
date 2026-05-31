// DeepEP topk -> SonicMoE metadata: high-performance 2-kernel design.
//
// Eliminates ALL spin-wait. Uses histogram-based prefix sum approach:
//   Kernel 1: histogram (per-block expert counts) + naept counts + prefix sums
//   Kernel 2: scatter + fixup (fully parallel, zero synchronization)
//
// Key optimizations vs v1 (single-kernel with inter-block chain):
//   - No spin-wait: 94.2% "No Eligible" stall eliminated entirely
//   - Single atomic flag barrier (1 global read) instead of 512-deep chain
//   - Fully coalesced reads of dispatched_indices/probs (row-major, 32 rows/block)
//   - Warp ballot + __popc for stable intra-block ordering (identical to moe_permute)
//   - Template-unrolled topk loop
//   - Vectorized int4 pad-fill in fixup
//   - Binary search for expert lookup in fixup (O(log E))
//
// Token ordering: STABLE ascending within each expert (identical to moe_permute).

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

static constexpr int ROWS_PER_BLOCK = 32;  // = warp size for ballot
static constexpr int BLOCK_DIM = 256;
static constexpr int MAX_TOPK = 16;

// ============================================================================
//  Kernel 1a: Histogram only (Phase A)
//
//  Each block independently counts per-expert tokens in its 32 rows using warp
//  ballot → writes block_hist[blockIdx.x * E + e]; also writes
//  naept[row+1] = per-token valid count.
//
//  No inter-block synchronization — safe for arbitrarily large grids.
//  Phase B (prefix sums) runs as a separate kernel launch on the same stream,
//  giving an implicit grid barrier.  This avoids the ad-hoc grid spin-wait
//  that deadlocks once `scatter_blocks` exceeds device occupancy (Target GPU hangs
//  at scatter_blocks ≈ 1358, observed via cuda-gdb on the legacy combined
//  kernel at kernel.cu:121).
// ============================================================================
template <typename IndexT, int TOPK>
__global__ __launch_bounds__(BLOCK_DIM)
void histogram_kernel(
    const IndexT* __restrict__ dispatched_indices,  // [N_recv, topk]
    int*         __restrict__ block_hist,            // [E * scatter_blocks] output
    int*         __restrict__ block_naept_sum,       // [scatter_blocks] output: sum of valid_counts in this block
    int          N_recv,
    int          num_experts,
    int          topk_param)
{
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    constexpr int warp_num = BLOCK_DIM >> 5;

    const int block_row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int global_row = block_row_base + lane_id;
    const bool row_valid = global_row < N_recv;

    extern __shared__ char smem[];
    uint32_t* expert_bitmask = reinterpret_cast<uint32_t*>(smem);

    // Initialize bitmask
    for (int i = threadIdx.x; i < num_experts; i += BLOCK_DIM) {
        expert_bitmask[i] = 0u;
    }
    __syncthreads();

    // Load topk entries + build per-expert bitmask
    int reg_valid_count = 0;

    #pragma unroll
    for (int col = 0; col < TOPK; col++) {
        if (col >= topk_param) break;
        int64_t expert_wide = -1;
        if (row_valid) {
            expert_wide = static_cast<int64_t>(dispatched_indices[global_row * topk_param + col]);
        }
        if (expert_wide >= 0 && expert_wide < static_cast<int64_t>(num_experts)) {
            int expert = static_cast<int>(expert_wide);
            // Use warp-distributed atomicOr to reduce contention
            if (col % warp_num == warp_id) {
                atomicOr(&expert_bitmask[expert], 1u << lane_id);
            }
            reg_valid_count++;
        }
    }
    __syncthreads();

    // ── Emit per-block sum of valid_counts (Fusion v2) ────────────────────
    // Each lane = one row in this block (32 rows/block); reg_valid_count is
    // identical across all 8 warps for a given lane (recomputed redundantly),
    // so we only reduce within warp 0. naept[i+1] is NOT written here — the
    // global exclusive prefix is materialized later by scatter_and_fixup.
    if (warp_id == 0) {
        int my_count = row_valid ? reg_valid_count : 0;
        int sum = my_count;
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, d);
        }
        if (lane_id == 0) {
            block_naept_sum[blockIdx.x] = sum;
        }
    }

    // Write per-expert counts for this block to block_hist
    // Layout: expert-major [E * scatter_blocks] — gives coalesced reads in
    // block_offset_scan_kernel where each block scans one expert's row.
    for (int e = warp_id; e < num_experts; e += warp_num) {
        int count = __popc(expert_bitmask[e]);
        if (lane_id == 0) {
            block_hist[e * gridDim.x + blockIdx.x] = count;
        }
    }
}

// ============================================================================
//  Kernel 1b: Per-expert exclusive prefix scan of block_hist row.
//
//  Replaces the serialized B.2 column-scan from the legacy single-block
//  prefix_sums_kernel.  Grid = E (one block per expert) gives full GPU
//  occupancy on E ≤ 256.  With expert-major block_hist layout the per-block
//  read is fully coalesced.
//
//  Algorithm: 3-phase block-wide scan
//    (a) Each thread serially sums its strided chunk → partial[tid].
//    (b) Block-wide exclusive scan of partial[] using two-level
//        (warp shuffle + cross-warp shared scan).
//    (c) Each thread writes per-element exclusive scan starting from its
//        partial-base into block_offset.
//
//  No spin-wait, no cross-block dependencies — fully deterministic.
// ============================================================================
static __device__ __forceinline__ int warp_exclusive_scan(int v) {
    // Hillis-Steele warp-level inclusive scan via __shfl_up_sync, then convert
    // to exclusive by shifting.
    const int lane = threadIdx.x & 31;
    int x = v;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int y = __shfl_up_sync(0xFFFFFFFF, x, d);
        if (lane >= d) x += y;
    }
    // inclusive → exclusive: subtract own value
    return x - v;
}

__global__ __launch_bounds__(BLOCK_DIM)
void block_offset_scan_kernel(
    const int* __restrict__ block_hist,          // [E * scatter_blocks]
    int*       __restrict__ block_offset,        // [E * scatter_blocks]
    int*       __restrict__ tokens_per_expert,   // [E] optional output, may be nullptr
    int scatter_blocks)
{
    const int expert = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int warp_num = BLOCK_DIM >> 5;

    const int* hist_row = block_hist  + expert * scatter_blocks;
    int*       off_row  = block_offset + expert * scatter_blocks;

    extern __shared__ char smem_so[];
    int* warp_totals = reinterpret_cast<int*>(smem_so);  // [warp_num]

    // Blocked partition: thread tid handles [chunk_lo, chunk_hi) of the row.
    int chunk_size = (scatter_blocks + BLOCK_DIM - 1) / BLOCK_DIM;
    int chunk_lo = tid * chunk_size;
    int chunk_hi = chunk_lo + chunk_size;
    if (chunk_hi > scatter_blocks) chunk_hi = scatter_blocks;
    if (chunk_lo > scatter_blocks) chunk_lo = scatter_blocks;

    // Phase (a): each thread sums its contiguous chunk of the row.
    int partial = 0;
    for (int i = chunk_lo; i < chunk_hi; i++) {
        partial += hist_row[i];
    }

    // Phase (b1): warp-level inclusive scan of partials, then collect totals.
    int warp_excl = warp_exclusive_scan(partial);
    int warp_incl = warp_excl + partial;
    if (lane == 31) warp_totals[warp] = warp_incl;
    __syncthreads();

    // Phase (b2): exclusive scan of warp totals (one warp).
    if (warp == 0) {
        int t = (lane < warp_num) ? warp_totals[lane] : 0;
        int e = warp_exclusive_scan(t);
        if (lane < warp_num) warp_totals[lane] = e;
        int last_e = __shfl_sync(0xFFFFFFFF, e, warp_num - 1);
        int last_t = __shfl_sync(0xFFFFFFFF, t, warp_num - 1);
        if (lane == 0 && tokens_per_expert != nullptr) {
            tokens_per_expert[expert] = last_e + last_t;
        }
    }
    __syncthreads();

    int my_base = warp_totals[warp] + warp_excl;

    // Phase (c): per-thread serial scan over its contiguous chunk.
    int running = my_base;
    for (int i = chunk_lo; i < chunk_hi; i++) {
        int v = hist_row[i];
        off_row[i] = running;
        running += v;
    }
}

// ============================================================================
//  Kernel 1c: Tail prefix sums (B.1 expert_offsets + block_naept_sum scan).
//
//  Fusion v2: instead of scanning naept[1..N_recv] (~116K elements), we scan
//  block_naept_sum[0..scatter_blocks] (~3625 elements, 32× smaller). The
//  per-row exclusive prefix WITHIN each block is then computed on-the-fly by
//  scatter_and_fixup_kernel from registers and offset by block_naept_base.
//
//  Single block, BLOCK_DIM threads.  3-phase block-wide scan
//  (per-thread serial sum → warp-shuffle exclusive scan of warp totals →
//   per-thread serial scatter).  Output array block_naept_base holds the
//   exclusive prefix of block_naept_sum.
// ============================================================================
__global__ __launch_bounds__(BLOCK_DIM)
void prefix_sums_kernel(
    const int*   __restrict__ tokens_per_expert,    // [E]
    const int*   __restrict__ block_naept_sum,       // [scatter_blocks] in
    int*         __restrict__ expert_offsets,        // [E+1] out
    int*         __restrict__ seg_starts,            // [E] out
    int*         __restrict__ real_bases,            // [E] out
    int*         __restrict__ block_naept_base,      // [scatter_blocks] out: exclusive prefix
    int*         __restrict__ naept,                 // [N_recv+1] (only naept[0] and naept[N_recv] written here)
    int          N_recv,
    int          scatter_blocks,
    int          num_experts,
    int          alignment)
{
    extern __shared__ char smem[];
    int* warp_totals = reinterpret_cast<int*>(smem);  // [warp_num]
    constexpr int warp_num = BLOCK_DIM >> 5;

    // --- B.1: Expert offsets (thread 0, O(E) — typically 8..128) ---
    if (threadIdx.x == 0) {
        int padded_cum = 0, real_cum = 0;
        expert_offsets[0] = 0;
        for (int e = 0; e < num_experts; e++) {
            int count = tokens_per_expert[e];
            int padded = (count > 0) ? ((count + alignment - 1) / alignment * alignment) : 0;
            seg_starts[e] = padded_cum;
            real_bases[e] = real_cum;
            padded_cum += padded;
            real_cum += count;
            expert_offsets[e + 1] = padded_cum;
        }
        naept[0] = 0;
    }

    // --- B.3 (v2): exclusive prefix scan of block_naept_sum ---
    //   block_naept_sum has scatter_blocks entries (~3625 at user shape).
    //   Use BLOCKED partition so each thread owns a contiguous chunk.
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int chunk_size = (scatter_blocks + BLOCK_DIM - 1) / BLOCK_DIM;
    int chunk_lo = tid * chunk_size;
    int chunk_hi = chunk_lo + chunk_size;
    if (chunk_hi > scatter_blocks) chunk_hi = scatter_blocks;
    if (chunk_lo > scatter_blocks) chunk_lo = scatter_blocks;

    // (a) per-thread serial sum over contiguous chunk (read-only — no race).
    int partial = 0;
    for (int i = chunk_lo; i < chunk_hi; i++) {
        partial += block_naept_sum[i];
    }

    // (b1) warp-level exclusive scan of partials.
    int warp_excl = warp_exclusive_scan(partial);
    int warp_incl = warp_excl + partial;
    if (lane == 31) warp_totals[warp] = warp_incl;
    __syncthreads();

    // (b2) cross-warp exclusive scan + grand total (single warp).
    __shared__ int grand_total;
    if (warp == 0) {
        int t = (lane < warp_num) ? warp_totals[lane] : 0;
        int e = warp_exclusive_scan(t);
        if (lane < warp_num) warp_totals[lane] = e;
        int last_e = __shfl_sync(0xFFFFFFFF, e, warp_num - 1);
        int last_t = __shfl_sync(0xFFFFFFFF, t, warp_num - 1);
        if (lane == 0) {
            grand_total = last_e + last_t;
        }
    }
    __syncthreads();

    int my_base = warp_totals[warp] + warp_excl;

    // (c) per-thread serial exclusive scatter into block_naept_base
    //     (separate output buffer → no races).
    int running = my_base;
    for (int i = chunk_lo; i < chunk_hi; i++) {
        int v = block_naept_sum[i];
        block_naept_base[i] = running;
        running += v;
    }

    // Total valid count goes into naept[N_recv] (read by downstream consumers).
    if (tid == 0) {
        naept[N_recv] = grand_total;
    }
}

// ============================================================================
//  Kernel 2: Scatter + Fixup (fully parallel, ZERO synchronization)
//
//  Each block processes ROWS_PER_BLOCK=32 token rows for scatter,
//  then processes a chunk of TK_padded for fixup.
//  All data needed (block_offset, expert_offsets, naept) is precomputed.
// ============================================================================
template <typename IndexT, int TOPK>
__global__ __launch_bounds__(BLOCK_DIM)
void scatter_and_fixup_kernel(
    const IndexT* __restrict__ dispatched_indices,  // [N_recv, topk]
    const float* __restrict__ dispatched_probs,     // [N_recv, topk]
    const int*   __restrict__ block_offset,          // [E * scatter_blocks]
    const int*   __restrict__ block_naept_base,      // [scatter_blocks] (Fusion v2)
    const int*   __restrict__ expert_offsets,        // [E+1] (padded cumsum)
    const int*   __restrict__ seg_starts,            // [E]
    const int*   __restrict__ tokens_per_expert,    // [E]
    int*         __restrict__ naept,                 // [N_recv+1] OUT (we materialize naept[global_row])
    int*         __restrict__ x_gather_idx,          // [TK_padded] output
    int*         __restrict__ s_scatter_idx,         // [TK_padded] output
    int*         __restrict__ s_reverse_scatter_idx, // [TK] output
    float*       __restrict__ topk_scores,           // [TK_padded] token-major output
    float*       __restrict__ expert_order_scores,   // [TK_padded] expert-order output
    int*         __restrict__ score_src_idx,         // [TK] output (int32: row*topk + col)
    int          N_recv,
    int          num_experts,
    int          topk_param,
    int          TK,
    int          TK_padded,
    int          scatter_blocks)
{
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    constexpr int warp_num = BLOCK_DIM >> 5;

    // ═══════════ Phase 1: Scatter (warp-ballot, deterministic ordering) ═══════
    if (blockIdx.x < scatter_blocks) {
        extern __shared__ char smem[];
        uint32_t* expert_bitmask = reinterpret_cast<uint32_t*>(smem);
        // Also cache block_offset for this block
        int* my_offset = reinterpret_cast<int*>(smem + num_experts * sizeof(uint32_t));

        // Load this block's precomputed offsets + init bitmask
        for (int i = threadIdx.x; i < num_experts; i += BLOCK_DIM) {
            expert_bitmask[i] = 0u;
            my_offset[i] = block_offset[i * scatter_blocks + blockIdx.x];
        }
        __syncthreads();

        const int block_row_base = blockIdx.x * ROWS_PER_BLOCK;
        const int global_row = block_row_base + lane_id;
        const bool row_valid = global_row < N_recv;

        // Load topk entries + build bitmask
        int reg_expert[MAX_TOPK];
        float reg_prob[MAX_TOPK];

        #pragma unroll
        for (int k = 0; k < TOPK; k++) {
            reg_expert[k] = -1;
            reg_prob[k] = 0.0f;
        }

        int reg_valid_count = 0;
        #pragma unroll
        for (int col = 0; col < TOPK; col++) {
            if (col >= topk_param) break;
            int64_t expert_wide = -1;
            float prob = 0.0f;
            if (row_valid) {
                expert_wide = static_cast<int64_t>(dispatched_indices[global_row * topk_param + col]);
                prob = dispatched_probs[global_row * topk_param + col];
            }
            if (expert_wide >= 0 && expert_wide < static_cast<int64_t>(num_experts)) {
                int expert = static_cast<int>(expert_wide);
                if (col % warp_num == warp_id) {
                    atomicOr(&expert_bitmask[expert], 1u << lane_id);
                }
                reg_expert[col] = expert;
                reg_prob[col] = prob;
                reg_valid_count++;
            }
        }
        __syncthreads();

        // ── Fusion v2: per-row exclusive prefix of valid_count within block ──
        //   Each lane = one row; warp_id 0 owns the canonical row→prefix mapping.
        //   We also publish naept[global_row] for downstream consumers (e.g.
        //   score_src_idx tests). Within a block: lane-disjoint writes (no race);
        //   across blocks: scatter_blocks blocks each write 32 disjoint cells of
        //   naept[0..N_recv) → no inter-block race. naept[N_recv] is written by
        //   prefix_sums_kernel and is stream-ordered before scatter.
        //   reg_valid_count is identical across all 8 warps (each warp recomputed
        //   from registers using the same data), so warp 0's scan is canonical.
        int my_count = row_valid ? reg_valid_count : 0;
        int local_excl = warp_exclusive_scan(my_count);  // exclusive prefix within warp
        const int naept_base = block_naept_base[blockIdx.x] + local_excl;
        if (warp_id == 0 && row_valid) {
            naept[global_row] = naept_base;
        }

        // Assign positions using ballot prefix count (deterministic, stable)
        int reg_padded_pos[MAX_TOPK];
        #pragma unroll
        for (int k = 0; k < TOPK; k++) reg_padded_pos[k] = -1;

        for (int expert_id = warp_id; expert_id < num_experts; expert_id += warp_num) {
            const uint32_t mask = expert_bitmask[expert_id];
            if (mask == 0u) continue;

            // This block's starting offset for this expert (precomputed, no spin!)
            const int base_offset = my_offset[expert_id];
            const bool lane_active = (mask & (1u << lane_id)) != 0;

            if (lane_active && row_valid) {
                // Intra-block position: stable via ballot prefix count
                int intra_pos = base_offset + __popc(mask & ((1u << lane_id) - 1));
                int padded_pos = seg_starts[expert_id] + intra_pos;

                // Find which topk slot matches this expert
                #pragma unroll
                for (int k = 0; k < TOPK; k++) {
                    if (reg_expert[k] == expert_id) {
                        reg_padded_pos[k] = padded_pos;
                        break;
                    }
                }
            }
        }

        // Write all outputs for this token
        // BUG FIX: within_token_rank must be computed globally across ALL warps.
        // Each warp only sets reg_padded_pos for its assigned experts, but all
        // threads have the complete reg_expert array (all topk cols were read).
        // Compute rank = count of valid expert slots with smaller expert_id
        // to give stable ascending-expert ordering within each token.
        if (row_valid) {
            #pragma unroll
            for (int k = 0; k < TOPK; k++) {
                if (k >= topk_param) break;
                if (reg_padded_pos[k] >= 0) {
                    const int padded_pos = reg_padded_pos[k];

                    // Compute global rank: how many of this token's valid
                    // expert assignments have expert_id < reg_expert[k]?
                    int rank = 0;
                    #pragma unroll
                    for (int j = 0; j < TOPK; j++) {
                        if (j >= topk_param) break;
                        if (reg_expert[j] >= 0 && reg_expert[j] < reg_expert[k]) {
                            rank++;
                        }
                    }

                    const int token_major_pos = naept_base + rank;

                    // Write ALL outputs (no conflicts — positions are unique)
                    x_gather_idx[padded_pos] = global_row;
                    s_scatter_idx[padded_pos] = token_major_pos;
                    s_reverse_scatter_idx[token_major_pos] = padded_pos;
                    if (topk_scores != nullptr) {
                        topk_scores[token_major_pos] = reg_prob[k];
                    }
                    if (expert_order_scores != nullptr) {
                        expert_order_scores[padded_pos] = reg_prob[k];
                    }
                    // Score-source flat index: token-major rank → original
                    // (row * topk + col) flat index into dispatched_probs.
                    // Bit-exact replacement for _build_score_src_idx_kernel.
                    score_src_idx[token_major_pos] = global_row * topk_param + k;
                }
            }
        }
    }

    // ═══════════ Phase 2: Pad-fill (vectorized, coalesced) ════════════════════
    // Fill expert-order padding positions: x_gather_idx=0, s_scatter_idx=TK.
    // Process all TK_padded positions, skip real ones.
    // Use grid-stride loop for full coverage.

    const int total_threads = gridDim.x * BLOCK_DIM;
    const int global_tid = blockIdx.x * BLOCK_DIM + threadIdx.x;
    const int actual_TK_padded = expert_offsets[num_experts];
    const int actual_TK = naept[N_recv];
    if (topk_scores != nullptr && global_tid == 0 && actual_TK < actual_TK_padded) {
        topk_scores[actual_TK] = 0.0f;
    }

    for (int pos = global_tid; pos < actual_TK_padded; pos += total_threads) {
        int lo = 0, hi = num_experts;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (expert_offsets[mid + 1] <= pos) lo = mid + 1;
            else hi = mid;
        }
        const int seg_start = expert_offsets[lo];
        const int real_count = tokens_per_expert[lo];
        const int local_pos = pos - seg_start;

        if (local_pos >= real_count) {
            x_gather_idx[pos] = 0;
            s_scatter_idx[pos] = actual_TK;
            if (expert_order_scores != nullptr) {
                expert_order_scores[pos] = 0.0f;
            }
        }
    }
}

// ============================================================================
//  C++ entry point
// ============================================================================
template <typename IndexT>
void launch_deepep_topk_metadata_cuda(
    torch::Tensor& dispatched_indices,
    torch::Tensor& dispatched_probs,
    torch::Tensor& tokens_per_expert,
    torch::Tensor& expert_offsets,
    torch::Tensor& seg_starts,
    torch::Tensor& real_bases,
    torch::Tensor& x_gather_idx,
    torch::Tensor& s_scatter_idx,
    torch::Tensor& s_reverse_scatter_idx,
    torch::Tensor& topk_scores,
    torch::Tensor& expert_order_scores,
    torch::Tensor& naept,
    torch::Tensor& global_block_cumsum,  // reused as workspace
    torch::Tensor& score_src_idx,
    int64_t N_recv,
    int64_t E,
    int64_t topk,
    int64_t TK,
    int64_t TK_padded,
    int64_t alignment,
    int64_t stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    float* topk_scores_ptr = topk_scores.numel() <= 1 ? nullptr : topk_scores.data_ptr<float>();
    float* expert_order_scores_ptr = expert_order_scores.numel() <= 1 ? nullptr : expert_order_scores.data_ptr<float>();
    const bool derive_counts = alignment < 0;
    const int actual_alignment = derive_counts ? -static_cast<int>(alignment) : static_cast<int>(alignment);

    if (N_recv == 0 || TK == 0) {
        if (derive_counts) {
            cudaMemsetAsync(tokens_per_expert.data_ptr<int>(), 0, E * sizeof(int), stream);
            cudaMemsetAsync(expert_offsets.data_ptr<int>(), 0, 3 * sizeof(int), stream);
        }
        cudaMemsetAsync(naept.data_ptr<int>(), 0, (N_recv + 1) * sizeof(int), stream);
        return;
    }

    const int scatter_blocks = (N_recv + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    // Workspace layout within global_block_cumsum:
    //   [0 .. scatter_blocks*E-1]:                  block_hist
    //   [scatter_blocks*E .. 2*scatter_blocks*E-1]: block_offset
    //   [2*scatter_blocks*E] (legacy):              previously completion_flag
    //                                               (no longer used; barrier
    //                                                is now a separate kernel
    //                                                launch).
    int* workspace = global_block_cumsum.data_ptr<int>();
    int* block_hist = workspace;
    int* block_offset = workspace + scatter_blocks * E;
    int* block_naept_sum = workspace + 2 * scatter_blocks * E;
    int* block_naept_base = workspace + 2 * scatter_blocks * E + scatter_blocks;

    // Shared memory for histogram_kernel: expert_bitmask[E]
    int smem_hist = static_cast<int>(E * sizeof(uint32_t));
    // Shared memory for block_offset_scan_kernel: warp_totals[warp_num]
    int smem_scan = static_cast<int>((BLOCK_DIM >> 5) * sizeof(int));
    // Shared memory for prefix_sums_kernel (tail): warp_totals[warp_num]
    int smem_prefix = static_cast<int>((BLOCK_DIM >> 5) * sizeof(int));

    dim3 grid1(scatter_blocks);
    dim3 block1(BLOCK_DIM);
    dim3 block_prefix(BLOCK_DIM);

    // ── Kernel 1a: Phase A — per-block histogram + per-block naept sum ───────
    #define LAUNCH_HIST(TV) \
        histogram_kernel<IndexT, TV><<<grid1, block1, smem_hist, stream>>>( \
            dispatched_indices.data_ptr<IndexT>(), \
            block_hist, \
            block_naept_sum, \
            static_cast<int>(N_recv), static_cast<int>(E), \
            static_cast<int>(topk));

    if (topk <= 4) { LAUNCH_HIST(4); }
    else if (topk <= 8) { LAUNCH_HIST(8); }
    else { LAUNCH_HIST(16); }
    #undef LAUNCH_HIST

    // ── Kernel 1b: Per-expert column scan of block_hist (parallel grid=E) ────
    block_offset_scan_kernel<<<dim3(static_cast<int>(E)), block1, smem_scan, stream>>>(
        block_hist, block_offset,
        derive_counts ? tokens_per_expert.data_ptr<int>() : nullptr,
        scatter_blocks);

    // ── Kernel 1c: Tail prefix sums (B.1 expert_offsets + scan over per-block sums) ──
    prefix_sums_kernel<<<dim3(1), block_prefix, smem_prefix, stream>>>(
        tokens_per_expert.data_ptr<int>(),
        block_naept_sum,
        expert_offsets.data_ptr<int>(),
        seg_starts.data_ptr<int>(),
        real_bases.data_ptr<int>(),
        block_naept_base,
        naept.data_ptr<int>(),
        static_cast<int>(N_recv), scatter_blocks, static_cast<int>(E),
        actual_alignment);

    // ── Kernel 2: Scatter + fixup ────────────────────────────────────────────
    // Grid must cover ALL scatter blocks: scatter phase relies on blockIdx.x
    // mapping 1:1 to a 32-row chunk of N_recv (block_row_base = blockIdx.x*32).
    // Capping the grid here would silently skip rows beyond cap*32 — which is
    // exactly what caused the SEQ_LEN ≥ 12K illegal-memory-access in
    // token_gather_sum_kernel: scatter_blocks=2712 capped at 2048 left rows
    // 65536..86768 unscattered, so x_gather_idx beyond TK_padded position
    // 2048*32 stayed at the pad-fill default and downstream gathers tried to
    // read 0-row data using s_scatter_idx pointing past valid topk_scores.
    // Pad-fill (Phase 2) uses a grid-stride loop, so any grid ≥ scatter_blocks
    // is correct and equally efficient.
    int grid2_blocks = scatter_blocks;
    int padfill_blocks = (int)(TK_padded + BLOCK_DIM - 1) / BLOCK_DIM;
    if (padfill_blocks > grid2_blocks) grid2_blocks = padfill_blocks;

    // Shared memory: expert_bitmask[E] + my_offset[E]
    int smem_k2 = (E * sizeof(uint32_t)) + (E * sizeof(int));

    dim3 grid2(grid2_blocks);
    dim3 block2(BLOCK_DIM);

    #define LAUNCH_K2(TV) \
        scatter_and_fixup_kernel<IndexT, TV><<<grid2, block2, smem_k2, stream>>>( \
            dispatched_indices.data_ptr<IndexT>(), \
            dispatched_probs.data_ptr<float>(), \
            block_offset, \
            block_naept_base, \
            expert_offsets.data_ptr<int>(), \
            seg_starts.data_ptr<int>(), \
            tokens_per_expert.data_ptr<int>(), \
            naept.data_ptr<int>(), \
            x_gather_idx.data_ptr<int>(), \
            s_scatter_idx.data_ptr<int>(), \
            s_reverse_scatter_idx.data_ptr<int>(), \
            topk_scores_ptr, \
            expert_order_scores_ptr, \
            score_src_idx.data_ptr<int>(), \
            static_cast<int>(N_recv), static_cast<int>(E), \
            static_cast<int>(topk), static_cast<int>(TK), \
            static_cast<int>(TK_padded), scatter_blocks);

    if (topk <= 4) { LAUNCH_K2(4); }
    else if (topk <= 8) { LAUNCH_K2(8); }
    else { LAUNCH_K2(16); }
    #undef LAUNCH_K2
}

void deepep_topk_metadata_cuda(
    torch::Tensor& dispatched_indices,
    torch::Tensor& dispatched_probs,
    torch::Tensor& tokens_per_expert,
    torch::Tensor& expert_offsets,
    torch::Tensor& seg_starts,
    torch::Tensor& real_bases,
    torch::Tensor& x_gather_idx,
    torch::Tensor& s_scatter_idx,
    torch::Tensor& s_reverse_scatter_idx,
    torch::Tensor& topk_scores,
    torch::Tensor& expert_order_scores,
    torch::Tensor& naept,
    torch::Tensor& global_block_cumsum,
    torch::Tensor& score_src_idx,
    int64_t N_recv,
    int64_t E,
    int64_t topk,
    int64_t TK,
    int64_t TK_padded,
    int64_t alignment,
    int64_t stream_ptr)
{
    if (dispatched_indices.scalar_type() == torch::kInt32) {
        launch_deepep_topk_metadata_cuda<int>(
            dispatched_indices, dispatched_probs, tokens_per_expert,
            expert_offsets, seg_starts, real_bases, x_gather_idx,
            s_scatter_idx, s_reverse_scatter_idx, topk_scores, expert_order_scores,
            naept, global_block_cumsum, score_src_idx, N_recv, E, topk, TK,
            TK_padded, alignment, stream_ptr);
    } else if (dispatched_indices.scalar_type() == torch::kInt64) {
        launch_deepep_topk_metadata_cuda<int64_t>(
            dispatched_indices, dispatched_probs, tokens_per_expert,
            expert_offsets, seg_starts, real_bases, x_gather_idx,
            s_scatter_idx, s_reverse_scatter_idx, topk_scores, expert_order_scores,
            naept, global_block_cumsum, score_src_idx, N_recv, E, topk, TK,
            TK_padded, alignment, stream_ptr);
    } else {
        TORCH_CHECK(false, "dispatched_indices must be int32 or int64");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deepep_topk_metadata_cuda",
          &deepep_topk_metadata_cuda,
          "DeepEP topk metadata: 2-kernel histogram + scatter (CUDA)",
          py::arg("dispatched_indices"),
          py::arg("dispatched_probs"),
          py::arg("tokens_per_expert"),
          py::arg("expert_offsets"),
          py::arg("seg_starts"),
          py::arg("real_bases"),
          py::arg("x_gather_idx"),
          py::arg("s_scatter_idx"),
          py::arg("s_reverse_scatter_idx"),
          py::arg("topk_scores"),
          py::arg("expert_order_scores"),
          py::arg("naept"),
          py::arg("global_block_cumsum"),
          py::arg("score_src_idx"),
          py::arg("N_recv"),
          py::arg("E"),
          py::arg("topk"),
          py::arg("TK"),
          py::arg("TK_padded"),
          py::arg("alignment"),
          py::arg("stream"));
}
