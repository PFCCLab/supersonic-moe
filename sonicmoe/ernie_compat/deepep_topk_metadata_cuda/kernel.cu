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
#include <limits>
#include <type_traits>
#include <vector>

static constexpr int ROWS_PER_BLOCK = 32;  // = warp size for ballot
static constexpr int BLOCK_DIM = 256;
static constexpr int MAX_TOPK = 16;
static constexpr int SF_TILE_M = 128;
static constexpr int SF_TILE_K = 128;
static constexpr int SF_VEC_SIZE = 32;
static constexpr int SF_TILE_STORAGE = SF_TILE_M * (SF_TILE_K / SF_VEC_SIZE);
static constexpr int SCALE_PACK_BLOCK_DIM = 256;
static constexpr int SCALE_PACK_WARPS_PER_CTA = SCALE_PACK_BLOCK_DIM / 32;

template <typename RawT>
__device__ __forceinline__ uint32_t pack_scale_row_4(
    const RawT* __restrict__ raw_scales,
    int64_t src_row,
    int64_t src_col_base,
    int64_t scale_cols,
    int64_t raw_stride_row,
    int64_t raw_stride_col)
{
    if (src_col_base + 3 < scale_cols) {
        if constexpr (std::is_same<RawT, int>::value) {
            if (raw_stride_col == 1) {
                const int4 r = *reinterpret_cast<const int4*>(
                    raw_scales + src_row * raw_stride_row + src_col_base);
                return
                    static_cast<uint32_t>(static_cast<uint8_t>(r.x)) |
                    (static_cast<uint32_t>(static_cast<uint8_t>(r.y)) << 8) |
                    (static_cast<uint32_t>(static_cast<uint8_t>(r.z)) << 16) |
                    (static_cast<uint32_t>(static_cast<uint8_t>(r.w)) << 24);
            }
        } else if constexpr (
            std::is_same<RawT, uint8_t>::value ||
            std::is_same<RawT, int8_t>::value) {
            if (raw_stride_col == 1) {
                return *reinterpret_cast<const uint32_t*>(
                    raw_scales + src_row * raw_stride_row + src_col_base);
            }
        }
    }

    uint32_t v = 0;
    #pragma unroll
    for (int k = 0; k < SF_TILE_K / SF_VEC_SIZE; ++k) {
        const int64_t src_col = src_col_base + k;
        uint8_t b = 1;
        if (src_col < scale_cols) {
            b = static_cast<uint8_t>(
                raw_scales[src_row * raw_stride_row + src_col * raw_stride_col]);
        }
        v |= static_cast<uint32_t>(b) << (8 * k);
    }
    return v;
}

static inline int64_t div_up_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static inline int64_t scale_storage_per_batch_i64(int64_t rows, int64_t cols) {
    return div_up_i64(rows, SF_TILE_M) * div_up_i64(cols, SF_TILE_K) * SF_TILE_STORAGE;
}

template <typename RawT>
__device__ __forceinline__ void pack_scale_row_to_sfa(
    const RawT* __restrict__ raw_scales,
    int64_t src_row,
    int64_t dst_row,
    uint8_t* __restrict__ packed_scales,
    int64_t scale_cols,
    int64_t raw_stride_row,
    int64_t raw_stride_col,
    int64_t k_tiles)
{
    const int64_t row_tile = dst_row / SF_TILE_M;
    const int64_t row_in_tile = dst_row - row_tile * SF_TILE_M;
    const int64_t lane_in_tile = row_in_tile & 31;
    const int64_t row_quarter = row_in_tile >> 5;

    for (int64_t k_tile_idx = 0; k_tile_idx < k_tiles; ++k_tile_idx) {
        const int64_t src_col_base = k_tile_idx * (SF_TILE_K / SF_VEC_SIZE);
        const uint32_t v = pack_scale_row_4(
            raw_scales, src_row, src_col_base, scale_cols,
            raw_stride_row, raw_stride_col);
        const int64_t dst_offset =
            (row_tile * k_tiles + k_tile_idx) * SF_TILE_STORAGE +
            lane_in_tile * 16 + row_quarter * 4;
        *reinterpret_cast<uint32_t*>(packed_scales + dst_offset) = v;
    }
}

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
template <int TOPK>
__global__ __launch_bounds__(BLOCK_DIM)
void histogram_kernel(
    const int*   __restrict__ dispatched_indices,   // [N_recv, topk]
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
        int expert = -1;
        if (row_valid) {
            expert = dispatched_indices[global_row * topk_param + col];
        }
        if (expert >= 0 && expert < num_experts) {
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
    const int* __restrict__ block_hist,    // [E * scatter_blocks]
    int*       __restrict__ block_offset,  // [E * scatter_blocks]
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
    const int*   __restrict__ block_hist,            // [E * scatter_blocks]
    const int*   __restrict__ block_offset,          // [E * scatter_blocks]
    const int*   __restrict__ block_naept_sum,       // [scatter_blocks] in
    int*         __restrict__ expert_offsets,        // [E+1] out
    int*         __restrict__ seg_starts,            // [E] out
    int*         __restrict__ expert_counts,         // [E] out
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
        int padded_cum = 0;
        expert_offsets[0] = 0;
        for (int e = 0; e < num_experts; e++) {
            const int last = scatter_blocks - 1;
            const int idx = e * scatter_blocks + last;
            const int count = block_offset[idx] + block_hist[idx];
            int padded = (count > 0) ? ((count + alignment - 1) / alignment * alignment) : 0;
            seg_starts[e] = padded_cum;
            expert_counts[e] = count;
            padded_cum += padded;
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
template <int TOPK, typename RawT = uint8_t, bool PACK_SCALES = false>
__global__ __launch_bounds__(BLOCK_DIM)
void scatter_and_fixup_kernel(
    const int*   __restrict__ dispatched_indices,   // [N_recv, topk]
    const float* __restrict__ dispatched_probs,     // [N_recv, topk]
    const int*   __restrict__ block_offset,          // [E * scatter_blocks]
    const int*   __restrict__ block_naept_base,      // [scatter_blocks] (Fusion v2)
    const int*   __restrict__ expert_offsets,        // [E+1] (padded cumsum)
    const int*   __restrict__ seg_starts,            // [E]
    const int*   __restrict__ expert_counts,         // [E]
    int*         __restrict__ naept,                 // [N_recv+1] OUT (we materialize naept[global_row])
    int*         __restrict__ x_gather_idx,          // [TK_padded] output
    int*         __restrict__ s_scatter_idx,         // [TK_padded] output
    int*         __restrict__ s_reverse_scatter_idx, // [TK] output
    float*       __restrict__ topk_scores,           // [TK_padded] output
    int*         __restrict__ score_src_idx,         // [TK] output (int32: row*topk + col)
    int          N_recv,
    int          num_experts,
    int          topk_param,
    int          TK,
    int          TK_padded,
    int          scatter_blocks,
    const RawT*  __restrict__ raw_scales = nullptr,
    uint8_t*     __restrict__ packed_scales = nullptr,
    int64_t      scale_cols = 0,
    int64_t      raw_stride_row = 0,
    int64_t      raw_stride_col = 0,
    int64_t      k_tiles = 0)
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
            int expert = -1;
            float prob = 0.0f;
            if (row_valid) {
                expert = dispatched_indices[global_row * topk_param + col];
                prob = dispatched_probs[global_row * topk_param + col];
            }
            if (expert >= 0 && expert < num_experts) {
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
                    topk_scores[token_major_pos] = reg_prob[k];
                    // Score-source flat index: token-major rank → original
                    // (row * topk + col) flat index into dispatched_probs.
                    // Bit-exact replacement for _build_score_src_idx_kernel.
                    score_src_idx[token_major_pos] = global_row * topk_param + k;
                    if constexpr (PACK_SCALES) {
                        pack_scale_row_to_sfa(
                            raw_scales, global_row, padded_pos, packed_scales,
                            scale_cols, raw_stride_row, raw_stride_col, k_tiles);
                    }
                }
            }
        }
    }

    // ═══════════ Phase 2: Pad-fill (vectorized, coalesced) ════════════════════
    // Fill padding positions: x_gather_idx=0, s_scatter_idx=TK, topk_scores=0
    // Process all TK_padded positions, skip real ones.
    // Use grid-stride loop for full coverage.

    const int total_threads = gridDim.x * BLOCK_DIM;
    const int global_tid = blockIdx.x * BLOCK_DIM + threadIdx.x;

    for (int pos = global_tid; pos < TK_padded; pos += total_threads) {
        // Binary search for expert
        int lo = 0, hi = num_experts;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (expert_offsets[mid + 1] <= pos) lo = mid + 1;
            else hi = mid;
        }
        const int seg_start = expert_offsets[lo];
        const int real_count = expert_counts[lo];
        const int local_pos = pos - seg_start;

        if (local_pos >= real_count) {
            // Padding position: fill defaults
            x_gather_idx[pos] = 0;
            s_scatter_idx[pos] = TK;  // points to topk_scores[TK]=0
            // topk_scores[pos] already 0 from zero-init
            if constexpr (PACK_SCALES) {
                pack_scale_row_to_sfa(
                    raw_scales, 0, pos, packed_scales,
                    scale_cols, raw_stride_row, raw_stride_col, k_tiles);
            }
        }
    }

    // Real scores densely cover [0, TK); clear only the padded tail here.
    for (int score_pos = TK + global_tid;
         score_pos < TK_padded;
         score_pos += total_threads) {
        topk_scores[score_pos] = 0.0f;
    }
}

// ============================================================================
//  Optional scale packing: raw DeepEP FP8 scales -> Sonic ISA/SFA layout.
//
//  This is intentionally kept as a separate stream-ordered CUDA kernel inside
//  the same C++ custom-op launcher.  The data dependency is x_gather_idx, which
//  is fully materialized by scatter_and_fixup_kernel.  Sinking this launch below
//  the metadata op removes the Python/Triton dispatch bubble while preserving
//  the safe grid barriers between metadata phases.
// ============================================================================
template <typename RawT>
__global__ __launch_bounds__(SCALE_PACK_BLOCK_DIM)
void pack_raw_scales_from_gather_kernel(
    const RawT*   __restrict__ raw_scales,   // [N_recv, ceil(cols/32)]
    const int*    __restrict__ x_gather_idx, // [TK_padded]
    uint8_t*      __restrict__ packed_scales,
    int64_t       TK_padded,
    int64_t       scale_cols,
    int64_t       raw_stride_row,
    int64_t       raw_stride_col,
    int64_t       k_tiles)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t row_tile =
        static_cast<int64_t>(blockIdx.x) * SCALE_PACK_WARPS_PER_CTA + warp;
    const int64_t row_base = row_tile * SF_TILE_M;
    if (row_base >= TK_padded) {
        return;
    }

    const int64_t k_tile_idx = blockIdx.y;
    const int64_t src_col_base = k_tile_idx * (SF_TILE_K / SF_VEC_SIZE);
    const int64_t dst_tile_base =
        (row_tile * k_tiles + k_tile_idx) * SF_TILE_STORAGE;
    uint4 packed16;

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int64_t row = row_base + lane + q * 32;
        uint32_t v = 0x01010101u;
        if (row < TK_padded) {
            const int64_t src_row = static_cast<int64_t>(x_gather_idx[row]);
            v = pack_scale_row_4(
                raw_scales, src_row, src_col_base, scale_cols,
                raw_stride_row, raw_stride_col);
        }
        if (q == 0) packed16.x = v;
        else if (q == 1) packed16.y = v;
        else if (q == 2) packed16.z = v;
        else packed16.w = v;
    }

    *reinterpret_cast<uint4*>(packed_scales + dst_tile_base + lane * 16) = packed16;
}

// Variant for int32/uint8 row-major raw scales.  It trades one shared-memory
// transpose for coalesced global loads while preserving coalesced ISA stores.
template <typename RawT>
__global__ __launch_bounds__(SCALE_PACK_BLOCK_DIM)
void pack_raw_scales_rowmajor_kernel(
    const RawT*   __restrict__ raw_scales,   // [N_recv, ceil(cols/32)]
    const int*    __restrict__ x_gather_idx, // [TK_padded]
    uint8_t*      __restrict__ packed_scales,
    int64_t       TK_padded,
    int64_t       scale_cols,
    int64_t       raw_stride_row,
    int64_t       raw_stride_col,
    int64_t       k_tiles)
{
    extern __shared__ uint8_t smem_scales[];
    const int tid = threadIdx.x;
    const int64_t row_tile = blockIdx.x;
    const int64_t row_base = row_tile * SF_TILE_M;
    const int64_t tile_elems = SF_TILE_M * scale_cols;

    for (int64_t idx = tid; idx < tile_elems; idx += blockDim.x) {
        const int64_t r = idx / scale_cols;
        const int64_t c = idx - r * scale_cols;
        const int64_t dst_row = row_base + r;
        uint8_t b = 1;
        if (dst_row < TK_padded) {
            const int64_t src_row = static_cast<int64_t>(x_gather_idx[dst_row]);
            b = static_cast<uint8_t>(
                raw_scales[src_row * raw_stride_row + c * raw_stride_col]);
        }
        smem_scales[idx] = b;
    }
    __syncthreads();

    const int64_t slots = k_tiles * 32;
    for (int64_t slot = tid; slot < slots; slot += blockDim.x) {
        const int64_t k_tile_idx = slot >> 5;
        const int64_t lane = slot & 31;
        const int64_t src_col_base = k_tile_idx * (SF_TILE_K / SF_VEC_SIZE);
        const int64_t dst_tile_base =
            (row_tile * k_tiles + k_tile_idx) * SF_TILE_STORAGE;
        uint4 packed16;

        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int64_t r = lane + q * 32;
            uint32_t v = 0;
            #pragma unroll
            for (int k = 0; k < SF_TILE_K / SF_VEC_SIZE; ++k) {
                const int64_t c = src_col_base + k;
                const uint8_t b = (c < scale_cols)
                    ? smem_scales[r * scale_cols + c]
                    : static_cast<uint8_t>(1);
                v |= static_cast<uint32_t>(b) << (8 * k);
            }
            if (q == 0) packed16.x = v;
            else if (q == 1) packed16.y = v;
            else if (q == 2) packed16.z = v;
            else packed16.w = v;
        }

        *reinterpret_cast<uint4*>(packed_scales + dst_tile_base + lane * 16) = packed16;
    }
}

// ============================================================================
//  C++ entry point
//
//  Allocation contract: this launcher now OWNS its output tensors. It allocates
//  them via the caching allocator (torch::empty/zeros) and returns them,
//  instead of receiving pre-allocated buffers through ``mutates_args``.
//  Motivation: the Python side previously issued ~9-11
//  ``torch.empty``/``torch.zeros`` dygraph dispatches per call. Sinking
//  allocation into C++ turns those into caching-allocator calls with zero
//  Python dispatch.
//
//  Correctness note (ctx-save / PP-1F1B safety): every torch::empty/zeros
//  below returns an INDEPENDENT storage per call — this is NOT cross-call
//  buffer reuse.  The returned tensors (expert_offsets / x_gather_idx /
//  s_scatter_idx / s_reverse_scatter_idx / naept / topk_scores /
//  score_src_idx) are saved on the autograd ctx by _UpProjection /
//  _DownProjection; giving each forward its own storage keeps interleaved
//  forward/backward (pipeline parallel, 1F1B) correct — identical semantics
//  to the previous per-call Python allocation.  ``seg_starts`` / ``real_bases``
//  / ``cumsum_workspace`` are pure scratch consumed entirely within this
//  launch and never escape.
//
//  The 4 kernels (histogram / block_offset_scan / prefix_sums /
//  scatter_and_fixup) are bit-identical to the previous revision; only the
//  ownership of the output buffers changed.
//
//  Returned vector order (must match the Python thin wrapper's unpacking):
//    [0] expert_offsets         int32 [E+1]
//    [1] x_gather_idx           int32 [TK_padded]  (zero-init)
//    [2] s_scatter_idx          int32 [TK_padded]
//    [3] s_reverse_scatter_idx  int32 [TK]
//    [4] naept                  int32 [N_recv+1]
//    [5] topk_scores            float32 [TK_padded] (zero-init)
//    [6] score_src_idx          int32 [TK]
//
//  Caller guarantees TK > 0 and N_recv > 0 (the TK==0 edge case is handled in
//  Python before dispatch, matching the previous behavior).
// ============================================================================
std::vector<torch::Tensor> deepep_topk_metadata_cuda_impl(
    torch::Tensor& dispatched_indices,
    torch::Tensor& dispatched_probs,
    int64_t N_recv,
    int64_t E,
    int64_t topk,
    int64_t TK,
    int64_t TK_padded,
    int64_t alignment,
    int64_t stream_ptr,
    torch::Tensor* raw_scales,
    int64_t cols,
    bool pack_scales_in_scatter,
    bool pack_scales_rowmajor,
    torch::Tensor* gated_output_prototype,
    int64_t gated_n,
    bool gated_preact_bf16,
    bool gated_allocate_z_scale)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    auto opt_i = torch::dtype(torch::kInt32).device(torch::kCUDA);
    auto opt_f = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    const int scatter_blocks = (N_recv + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int64_t workspace_elements =
        2 * static_cast<int64_t>(scatter_blocks) * (E + 1) + 2 * E;

    // Custom-op outputs must own independent storage. Undeclared output
    // aliasing can let live PP/VPP contexts observe later invocations.
    torch::Tensor expert_offsets        = torch::empty({E + 1}, opt_i);
    torch::Tensor x_gather_idx          = torch::empty({TK_padded}, opt_i);
    torch::Tensor s_scatter_idx         = torch::empty({TK_padded}, opt_i);
    torch::Tensor s_reverse_scatter_idx = torch::empty({TK}, opt_i);
    torch::Tensor naept                 = torch::empty({N_recv + 1}, opt_i);
    torch::Tensor score_src_idx         = torch::empty({TK}, opt_i);
    torch::Tensor cumsum_workspace =
        torch::empty({workspace_elements}, opt_i);
    torch::Tensor topk_scores           = torch::empty({TK_padded}, opt_f);

    // Workspace layout within cumsum_workspace:
    //   [0 .. scatter_blocks*E-1]:                  block_hist
    //   [scatter_blocks*E .. 2*scatter_blocks*E-1]: block_offset
    //   [2*scatter_blocks*E ..]:                    block_naept_sum / base
    //   [2*scatter_blocks*(E+1) ..]:                seg_starts / expert_counts
    int* workspace = cumsum_workspace.data_ptr<int>();
    int* block_hist = workspace;
    int* block_offset = workspace + scatter_blocks * E;
    int* block_naept_sum = workspace + 2 * scatter_blocks * E;
    int* block_naept_base = workspace + 2 * scatter_blocks * E + scatter_blocks;
    int* seg_starts = workspace + 2 * scatter_blocks * (E + 1);
    int* expert_counts = seg_starts + E;

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
        histogram_kernel<TV><<<grid1, block1, smem_hist, stream>>>( \
            dispatched_indices.data_ptr<int>(), \
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
        block_hist, block_offset, scatter_blocks);

    // ── Kernel 1c: Tail prefix sums (B.1 expert_offsets + scan over per-block sums) ──
    prefix_sums_kernel<<<dim3(1), block_prefix, smem_prefix, stream>>>(
        block_hist,
        block_offset,
        block_naept_sum,
        expert_offsets.data_ptr<int>(),
        seg_starts,
        expert_counts,
        block_naept_base,
        naept.data_ptr<int>(),
        static_cast<int>(N_recv), scatter_blocks, static_cast<int>(E),
        static_cast<int>(alignment));

    torch::Tensor packed_scales;
    uint8_t* packed_scale_output_bytes = nullptr;
    int64_t scale_cols = 0;
    int64_t stride_row = 0;
    int64_t stride_col = 0;
    int64_t k_tiles = 0;
    at::ScalarType raw_dtype = at::ScalarType::Byte;
    bool compact_scale_words = false;
    const uint8_t* compact_scale_bytes = nullptr;
    if (raw_scales != nullptr) {
        TORCH_CHECK(cols > 0, "cols must be positive when raw_scales is provided");
        TORCH_CHECK(raw_scales->is_cuda(), "raw_scales must be a CUDA tensor");
        TORCH_CHECK(raw_scales->dim() == 2,
                    "raw_scales must be a rank-2 raw scale matrix or compact int32 carrier");
        TORCH_CHECK(raw_scales->size(0) == N_recv,
                    "raw_scales row count mismatch: expected ", N_recv,
                    ", got ", raw_scales->size(0));

        const int64_t expected_scale_cols = div_up_i64(cols, SF_VEC_SIZE);
        const int64_t storage_scale_cols = raw_scales->size(1);
        raw_dtype = raw_scales->scalar_type();
        compact_scale_words =
            raw_dtype == at::ScalarType::Int &&
            expected_scale_cols % 4 == 0 &&
            storage_scale_cols == expected_scale_cols / 4;
        TORCH_CHECK(storage_scale_cols == expected_scale_cols || compact_scale_words,
                    "raw_scales column count mismatch: expected ", expected_scale_cols,
                    " raw scale elements or ", expected_scale_cols / 4,
                    " compact int32 words for cols=", cols,
                    ", got ", storage_scale_cols);

        auto opt_u8 = torch::dtype(torch::kUInt8).device(torch::kCUDA);
        const int64_t per_batch_storage = scale_storage_per_batch_i64(TK_padded, cols);
        packed_scales =
            (TK_padded % SF_TILE_M == 0 && cols % SF_TILE_K == 0)
                ? torch::empty({1, per_batch_storage}, opt_u8)
                : torch::full({1, per_batch_storage}, 1, opt_u8);
        packed_scale_output_bytes = packed_scales.data_ptr<uint8_t>();

        k_tiles = div_up_i64(cols, SF_TILE_K);
        TORCH_CHECK(raw_dtype == at::ScalarType::Int ||
                    raw_dtype == at::ScalarType::Byte ||
                    raw_dtype == at::ScalarType::Char,
                    "raw_scales dtype must be int32, uint8, or int8, got ",
                    raw_scales->scalar_type());
        scale_cols = expected_scale_cols;
        if (compact_scale_words) {
            // DeepEP transports four opaque E8M0 bytes in each int32 word.
            stride_row = raw_scales->stride(0) * sizeof(int);
            stride_col = 1;
            compact_scale_bytes = reinterpret_cast<const uint8_t*>(
                raw_scales->data_ptr<int>());
        } else {
            stride_row = raw_scales->stride(0);
            stride_col = raw_scales->stride(1);
        }
    }

    // Sink the four hot-path output allocations into this native bridge.
    // Each invocation still receives independent storage for autograd safety.
    torch::Tensor gated_preact;
    torch::Tensor gated_postact;
    torch::Tensor gated_z_scales;
    torch::Tensor gated_postact_scales;
    if (gated_output_prototype != nullptr) {
        TORCH_CHECK(raw_scales != nullptr,
                    "gated output allocation requires the with-scales path");
        TORCH_CHECK(gated_output_prototype->is_cuda(),
                    "gated_output_prototype must be a CUDA tensor");
        TORCH_CHECK(gated_output_prototype->element_size() == 1,
                    "gated_output_prototype must use a one-byte FP8 dtype");
        TORCH_CHECK(gated_n > 0 && gated_n % 256 == 0,
                    "gated_n must be positive and divisible by 256, got ", gated_n);
        TORCH_CHECK(TK_padded % SF_TILE_M == 0,
                    "TK_padded must be divisible by 128, got ", TK_padded);
        TORCH_CHECK(
            TK_padded <= std::numeric_limits<int64_t>::max() / gated_n,
            "gated output shape overflows int64: TK_padded=", TK_padded,
            ", gated_n=", gated_n);

        auto fp8_options = gated_output_prototype->options();
        auto preact_options = gated_preact_bf16
            ? torch::dtype(torch::kBFloat16).device(gated_output_prototype->device())
            : fp8_options;
        auto u8_options = torch::dtype(torch::kUInt8).device(
            gated_output_prototype->device());
        const int64_t postact_n = gated_n / 2;
        gated_preact = torch::empty({TK_padded, gated_n}, preact_options);
        gated_postact = torch::empty({TK_padded, postact_n}, fp8_options);
        gated_z_scales = gated_allocate_z_scale
            ? torch::empty({TK_padded, gated_n / SF_VEC_SIZE}, u8_options)
            : torch::empty({0}, u8_options);
        gated_postact_scales = torch::empty(
            {TK_padded / SF_TILE_M, postact_n / SF_TILE_K, SF_TILE_STORAGE},
            u8_options);
    }

    // ── Kernel 2: Scatter + fixup ────────────────────────────────────────────
    // Grid must cover ALL scatter blocks: scatter phase relies on blockIdx.x
    // mapping 1:1 to a 32-row chunk of N_recv (block_row_base = blockIdx.x*32).
    // Capping the grid here would silently skip rows beyond cap*32.
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
        scatter_and_fixup_kernel<TV><<<grid2, block2, smem_k2, stream>>>( \
            dispatched_indices.data_ptr<int>(), \
            dispatched_probs.data_ptr<float>(), \
            block_offset, \
            block_naept_base, \
            expert_offsets.data_ptr<int>(), \
            seg_starts, \
            expert_counts, \
            naept.data_ptr<int>(), \
            x_gather_idx.data_ptr<int>(), \
            s_scatter_idx.data_ptr<int>(), \
            s_reverse_scatter_idx.data_ptr<int>(), \
            topk_scores.data_ptr<float>(), \
            score_src_idx.data_ptr<int>(), \
            static_cast<int>(N_recv), static_cast<int>(E), \
            static_cast<int>(topk), static_cast<int>(TK), \
            static_cast<int>(TK_padded), scatter_blocks);

    #define LAUNCH_K2_PACK(TV, RAW_T, RAW_PTR) \
        scatter_and_fixup_kernel<TV, RAW_T, true><<<grid2, block2, smem_k2, stream>>>( \
            dispatched_indices.data_ptr<int>(), \
            dispatched_probs.data_ptr<float>(), \
            block_offset, \
            block_naept_base, \
            expert_offsets.data_ptr<int>(), \
            seg_starts, \
            expert_counts, \
            naept.data_ptr<int>(), \
            x_gather_idx.data_ptr<int>(), \
            s_scatter_idx.data_ptr<int>(), \
            s_reverse_scatter_idx.data_ptr<int>(), \
            topk_scores.data_ptr<float>(), \
            score_src_idx.data_ptr<int>(), \
            static_cast<int>(N_recv), static_cast<int>(E), \
            static_cast<int>(topk), static_cast<int>(TK), \
            static_cast<int>(TK_padded), scatter_blocks, \
            RAW_PTR, \
            packed_scale_output_bytes, \
            scale_cols, stride_row, stride_col, k_tiles);

    if (raw_scales != nullptr && pack_scales_in_scatter) {
        if (compact_scale_words) {
            if (topk <= 4) { LAUNCH_K2_PACK(4, uint8_t, compact_scale_bytes); }
            else if (topk <= 8) { LAUNCH_K2_PACK(8, uint8_t, compact_scale_bytes); }
            else { LAUNCH_K2_PACK(16, uint8_t, compact_scale_bytes); }
        } else if (raw_dtype == at::ScalarType::Int) {
            if (topk <= 4) { LAUNCH_K2_PACK(4, int, raw_scales->data_ptr<int>()); }
            else if (topk <= 8) { LAUNCH_K2_PACK(8, int, raw_scales->data_ptr<int>()); }
            else { LAUNCH_K2_PACK(16, int, raw_scales->data_ptr<int>()); }
        } else if (raw_dtype == at::ScalarType::Byte) {
            if (topk <= 4) { LAUNCH_K2_PACK(4, uint8_t, raw_scales->data_ptr<uint8_t>()); }
            else if (topk <= 8) { LAUNCH_K2_PACK(8, uint8_t, raw_scales->data_ptr<uint8_t>()); }
            else { LAUNCH_K2_PACK(16, uint8_t, raw_scales->data_ptr<uint8_t>()); }
        } else {
            if (topk <= 4) { LAUNCH_K2_PACK(4, int8_t, raw_scales->data_ptr<int8_t>()); }
            else if (topk <= 8) { LAUNCH_K2_PACK(8, int8_t, raw_scales->data_ptr<int8_t>()); }
            else { LAUNCH_K2_PACK(16, int8_t, raw_scales->data_ptr<int8_t>()); }
        }
    } else {
        if (topk <= 4) { LAUNCH_K2(4); }
        else if (topk <= 8) { LAUNCH_K2(8); }
        else { LAUNCH_K2(16); }
    }
    #undef LAUNCH_K2
    #undef LAUNCH_K2_PACK

    std::vector<torch::Tensor> out = {
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        naept,
        topk_scores,
        score_src_idx,
    };

    if (raw_scales != nullptr) {
        if (!pack_scales_in_scatter) {
            const int64_t row_tiles = div_up_i64(TK_padded, SF_TILE_M);
            dim3 block_scale(SCALE_PACK_BLOCK_DIM);
            const bool use_rowmajor_pack =
                pack_scales_rowmajor &&
                stride_col == 1 &&
                static_cast<size_t>(SF_TILE_M * scale_cols) <= 48 * 1024;

            if (use_rowmajor_pack) {
                dim3 grid_scale(static_cast<unsigned int>(row_tiles));
                size_t smem_scale = static_cast<size_t>(SF_TILE_M * scale_cols);
                if (compact_scale_words) {
                    pack_raw_scales_rowmajor_kernel<uint8_t><<<grid_scale, block_scale, smem_scale, stream>>>(
                        compact_scale_bytes,
                        x_gather_idx.data_ptr<int>(),
                        packed_scale_output_bytes,
                        TK_padded, scale_cols, stride_row, stride_col, k_tiles);
                } else if (raw_dtype == at::ScalarType::Int) {
                    pack_raw_scales_rowmajor_kernel<int><<<grid_scale, block_scale, smem_scale, stream>>>(
                        raw_scales->data_ptr<int>(),
                        x_gather_idx.data_ptr<int>(),
                        packed_scale_output_bytes,
                        TK_padded, scale_cols, stride_row, stride_col, k_tiles);
                } else if (raw_dtype == at::ScalarType::Byte) {
                    pack_raw_scales_rowmajor_kernel<uint8_t><<<grid_scale, block_scale, smem_scale, stream>>>(
                        raw_scales->data_ptr<uint8_t>(),
                        x_gather_idx.data_ptr<int>(),
                        packed_scale_output_bytes,
                        TK_padded, scale_cols, stride_row, stride_col, k_tiles);
                } else {
                    pack_raw_scales_rowmajor_kernel<int8_t><<<grid_scale, block_scale, smem_scale, stream>>>(
                        raw_scales->data_ptr<int8_t>(),
                        x_gather_idx.data_ptr<int>(),
                        packed_scale_output_bytes,
                        TK_padded, scale_cols, stride_row, stride_col, k_tiles);
                }
            } else {
                dim3 grid_scale(static_cast<unsigned int>(div_up_i64(row_tiles, SCALE_PACK_WARPS_PER_CTA)),
                                static_cast<unsigned int>(k_tiles));
                if (compact_scale_words) {
                    pack_raw_scales_from_gather_kernel<uint8_t><<<grid_scale, block_scale, 0, stream>>>(
                        compact_scale_bytes,
                        x_gather_idx.data_ptr<int>(),
                        packed_scale_output_bytes,
                        TK_padded, scale_cols, stride_row, stride_col, k_tiles);
                } else if (raw_dtype == at::ScalarType::Int) {
                    pack_raw_scales_from_gather_kernel<int><<<grid_scale, block_scale, 0, stream>>>(
                        raw_scales->data_ptr<int>(),
                        x_gather_idx.data_ptr<int>(),
                        packed_scale_output_bytes,
                        TK_padded, scale_cols, stride_row, stride_col, k_tiles);
                } else if (raw_dtype == at::ScalarType::Byte) {
                    pack_raw_scales_from_gather_kernel<uint8_t><<<grid_scale, block_scale, 0, stream>>>(
                        raw_scales->data_ptr<uint8_t>(),
                        x_gather_idx.data_ptr<int>(),
                        packed_scale_output_bytes,
                        TK_padded, scale_cols, stride_row, stride_col, k_tiles);
                } else {
                    pack_raw_scales_from_gather_kernel<int8_t><<<grid_scale, block_scale, 0, stream>>>(
                        raw_scales->data_ptr<int8_t>(),
                        x_gather_idx.data_ptr<int>(),
                        packed_scale_output_bytes,
                        TK_padded, scale_cols, stride_row, stride_col, k_tiles);
                }
            }
        }
        out.push_back(packed_scales);
    }
    if (gated_output_prototype != nullptr) {
        out.push_back(gated_preact);
        out.push_back(gated_postact);
        out.push_back(gated_z_scales);
        out.push_back(gated_postact_scales);
    }

    return out;
}

std::vector<torch::Tensor> deepep_topk_metadata_cuda(
    torch::Tensor& dispatched_indices,
    torch::Tensor& dispatched_probs,
    int64_t N_recv,
    int64_t E,
    int64_t topk,
    int64_t TK,
    int64_t TK_padded,
    int64_t alignment,
    int64_t stream_ptr)
{
    return deepep_topk_metadata_cuda_impl(
        dispatched_indices, dispatched_probs,
        N_recv, E, topk, TK, TK_padded, alignment, stream_ptr,
        nullptr, 0, false, false, nullptr, 0, false, false);
}

std::vector<torch::Tensor> deepep_topk_metadata_cuda_with_scales(
    torch::Tensor& dispatched_indices,
    torch::Tensor& dispatched_probs,
    int64_t N_recv,
    int64_t E,
    int64_t topk,
    int64_t TK,
    int64_t TK_padded,
    int64_t alignment,
    torch::Tensor& raw_scales,
    int64_t cols,
    int64_t stream_ptr)
{
    return deepep_topk_metadata_cuda_impl(
        dispatched_indices, dispatched_probs,
        N_recv, E, topk, TK, TK_padded, alignment, stream_ptr,
        &raw_scales, cols, false, false, nullptr, 0, false, false);
}

std::vector<torch::Tensor>
deepep_topk_metadata_cuda_with_scales_and_gated_outputs(
    torch::Tensor& dispatched_indices,
    torch::Tensor& dispatched_probs,
    int64_t N_recv,
    int64_t E,
    int64_t topk,
    int64_t TK,
    int64_t TK_padded,
    int64_t alignment,
    torch::Tensor& raw_scales,
    int64_t cols,
    torch::Tensor& gated_output_prototype,
    int64_t gated_n,
    bool gated_preact_bf16,
    bool gated_allocate_z_scale,
    int64_t stream_ptr)
{
    return deepep_topk_metadata_cuda_impl(
        dispatched_indices, dispatched_probs,
        N_recv, E, topk, TK, TK_padded, alignment, stream_ptr,
        &raw_scales, cols, false, false, &gated_output_prototype, gated_n,
        gated_preact_bf16, gated_allocate_z_scale);
}

std::vector<torch::Tensor> deepep_topk_metadata_cuda_with_scales_rowpack(
    torch::Tensor& dispatched_indices,
    torch::Tensor& dispatched_probs,
    int64_t N_recv,
    int64_t E,
    int64_t topk,
    int64_t TK,
    int64_t TK_padded,
    int64_t alignment,
    torch::Tensor& raw_scales,
    int64_t cols,
    int64_t stream_ptr)
{
    return deepep_topk_metadata_cuda_impl(
        dispatched_indices, dispatched_probs,
        N_recv, E, topk, TK, TK_padded, alignment, stream_ptr,
        &raw_scales, cols, false, true, nullptr, 0, false, false);
}

std::vector<torch::Tensor> deepep_topk_metadata_cuda_with_scales_scatterpack(
    torch::Tensor& dispatched_indices,
    torch::Tensor& dispatched_probs,
    int64_t N_recv,
    int64_t E,
    int64_t topk,
    int64_t TK,
    int64_t TK_padded,
    int64_t alignment,
    torch::Tensor& raw_scales,
    int64_t cols,
    int64_t stream_ptr)
{
    return deepep_topk_metadata_cuda_impl(
        dispatched_indices, dispatched_probs,
        N_recv, E, topk, TK, TK_padded, alignment, stream_ptr,
        &raw_scales, cols, true, false, nullptr, 0, false, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deepep_topk_metadata_cuda",
          &deepep_topk_metadata_cuda,
          "DeepEP topk metadata: allocates + returns metadata tensors (CUDA)",
          py::arg("dispatched_indices"),
          py::arg("dispatched_probs"),
          py::arg("N_recv"),
          py::arg("E"),
          py::arg("topk"),
          py::arg("TK"),
          py::arg("TK_padded"),
          py::arg("alignment"),
          py::arg("stream"));
    m.def("deepep_topk_metadata_cuda_with_scales",
          &deepep_topk_metadata_cuda_with_scales,
          "DeepEP topk metadata: returns metadata tensors plus packed Sonic FP8 scales (CUDA)",
          py::arg("dispatched_indices"),
          py::arg("dispatched_probs"),
          py::arg("N_recv"),
          py::arg("E"),
          py::arg("topk"),
          py::arg("TK"),
          py::arg("TK_padded"),
          py::arg("alignment"),
          py::arg("raw_scales"),
          py::arg("cols"),
          py::arg("stream"));
    m.def("deepep_topk_metadata_cuda_with_scales_and_gated_outputs",
          &deepep_topk_metadata_cuda_with_scales_and_gated_outputs,
          "DeepEP metadata + packed scales + preallocated FP8 gated outputs (CUDA)",
          py::arg("dispatched_indices"),
          py::arg("dispatched_probs"),
          py::arg("N_recv"),
          py::arg("E"),
          py::arg("topk"),
          py::arg("TK"),
          py::arg("TK_padded"),
          py::arg("alignment"),
          py::arg("raw_scales"),
          py::arg("cols"),
          py::arg("gated_output_prototype"),
          py::arg("gated_n"),
          py::arg("gated_preact_bf16"),
          py::arg("gated_allocate_z_scale"),
          py::arg("stream"));
    m.def("deepep_topk_metadata_cuda_with_scales_scatterpack",
          &deepep_topk_metadata_cuda_with_scales_scatterpack,
          "DeepEP topk metadata: packs Sonic FP8 scales inside scatter/fixup (CUDA)",
          py::arg("dispatched_indices"),
          py::arg("dispatched_probs"),
          py::arg("N_recv"),
          py::arg("E"),
          py::arg("topk"),
          py::arg("TK"),
          py::arg("TK_padded"),
          py::arg("alignment"),
          py::arg("raw_scales"),
          py::arg("cols"),
          py::arg("stream"));
    m.def("deepep_topk_metadata_cuda_with_scales_rowpack",
          &deepep_topk_metadata_cuda_with_scales_rowpack,
          "DeepEP topk metadata: row-major-load shared-transpose Sonic FP8 scale pack (CUDA)",
          py::arg("dispatched_indices"),
          py::arg("dispatched_probs"),
          py::arg("N_recv"),
          py::arg("E"),
          py::arg("topk"),
          py::arg("TK"),
          py::arg("TK_padded"),
          py::arg("alignment"),
          py::arg("raw_scales"),
          py::arg("cols"),
          py::arg("stream"));
}
