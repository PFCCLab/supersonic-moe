"""CPU planner for the grouped down projection (no GPU, no paddle).

Given one batch's routing (expert ids [T, K]) and a fixed list of expert
groups (pairs, triples, quads; list order = priority), choose for every token
a set of lane-disjoint groups contained in its top-k (greedy by priority) and
lay out the merged GEMM2 rows:

  segments  = groups ordered by (size desc, priority), then the E experts
              (residual rows, one expert each)
  rows      = global row space [0, nrow); segment s owns
              [seg_off[s], seg_off[s+1]); tokens are in ascending order inside
              every segment
  row_tok   = token of each row (int16 bit pattern of 0..65535)
  row_code  = sorted-rank lanes of the row, 4 bits each, low nibble first
              (int16 bit pattern; ranks refer to the token's experts sorted
              ascending)
  perm/moff = for the combine: token t's rows are perm[moff[t]:moff[t+1]]
  cu        = seg_off of every GEMM (size bucket, then the residual experts)
              as its own 16-byte aligned slice, for `cu_seqlens_m`:
              cu[at : at + s1 - s0 + 1] = seg_off[s0 : s1 + 1] for every
              (c, s0, s1, at) in index.gemms

Preconditions on `ids`: every row holds distinct experts in [0, E) or -1 (a
dropped lane: it has no row and takes no part in any group).
The result depends only on the routing and the list (deterministic for any
thread count or chunk count).

Cost per round: C(K, 2) probes of a dense E x E prefix table per token, plus
the triple/quad suffixes of the (a, b) prefixes that hit -- independent of the
list length.  Pair-only lists take a specialised pass without suffix lookups.
"""

import functools
import time
from dataclasses import dataclass

import numba as nb
import numpy as np
from numba.core import types as _nbt
from numba.extending import intrinsic as _intrinsic

MAX_C = 4  # largest group size
MAX_K = 16  # lanes are 4-bit ranks
MAX_T = 65536  # tokens are 16-bit


# --------------------------------------------------------------- list / index
def normalize_groups(groups, num_experts):
    """groups: [G, <=4] expert ids (-1 pad), priority order.  Returns int32
    [G, 4], each row sorted ascending, and the group sizes."""
    g = np.full((len(groups), MAX_C), -1, np.int32)
    for i, row in enumerate(groups):
        ex = sorted(int(e) for e in row if int(e) >= 0)
        if not 2 <= len(ex) <= MAX_C or len(set(ex)) != len(ex):
            raise ValueError(f"group {i} must have 2..{MAX_C} distinct experts, got {row}")
        if ex[-1] >= num_experts:
            raise ValueError(f"group {i} uses expert {ex[-1]} >= num_experts={num_experts}")
        g[i, : len(ex)] = ex
    sizes = (g >= 0).sum(1).astype(np.int32)
    if len({tuple(r[:s]) for r, s in zip(g, sizes)}) != len(g):
        raise ValueError("duplicate groups in the list")
    return g, sizes


@dataclass(frozen=True, eq=False)
class PlanIndex:
    """Static structures of one group list, built once by `build_index`."""

    G: int  # groups
    E: int  # experts
    groups: np.ndarray  # [G, 4] int32, sorted rows, -1 pad, priority order
    sizes: np.ndarray  # [G] int32
    seg_of: np.ndarray  # [G] segment of each group
    seg_groups: np.ndarray  # [G, 4] groups in segment order
    gemms: tuple  # ((c, s0, s1, at), ...): segments [s0, s1) of one GEMM, cu slice at `at`
    cu_len: int
    pairs_only: bool
    nbits: int  # hit-bitmap size: power of two > G
    pair_tab: np.ndarray  # [E * E] pair group of (a < b), or -1
    sfx_tab: np.ndarray  # [E * E] suffix node of prefix (a, b), or -1
    s_off: np.ndarray  # node n's suffixes: (s_e3, s_e4, s_g)[s_off[n]:s_off[n + 1]]
    s_e3: np.ndarray
    s_e4: np.ndarray  # -1 for a triple
    s_g: np.ndarray  # group id of the suffix


def build_index(groups, num_experts):
    """`PlanIndex` of a group list (priority order)."""
    g, sizes = normalize_groups(groups, num_experts)
    G, E = g.shape[0], num_experts
    order = sorted(range(G), key=lambda i: (-int(sizes[i]), i))  # segment order
    seg_of = np.empty(G, np.int32)
    seg_of[np.array(order, np.int64)] = np.arange(G, dtype=np.int32)
    pair_tab = np.full(E * E, -1, np.int32)
    sfx_tab = np.full(E * E, -1, np.int32)
    suffix = {}
    for i in range(G):
        a, b = int(g[i, 0]), int(g[i, 1])
        if sizes[i] == 2:
            pair_tab[a * E + b] = i
        else:
            suffix.setdefault((a, b), []).append((int(g[i, 2]), int(g[i, 3]), i))
    s_off, s_e3, s_e4, s_g = [0], [], [], []
    for nid, ab in enumerate(sorted(suffix)):
        sfx_tab[ab[0] * E + ab[1]] = nid
        for e3, e4, i in sorted(suffix[ab]):
            s_e3.append(e3)
            s_e4.append(e4)
            s_g.append(i)
        s_off.append(len(s_g))
    # one GEMM per size bucket (segments [s0, s1)), residual experts last;
    # `at` = start of its cu slice, 4-entry (16-byte) aligned
    spans = []
    for c in sorted({int(s) for s in sizes}, reverse=True):
        segs = seg_of[sizes == c]
        spans.append((c, int(segs.min()), int(segs.max()) + 1))
    gemms, at = [], 0
    for c, s0, s1 in spans + [(1, G, G + E)]:
        gemms.append((c, s0, s1, at))
        at += (s1 - s0 + 1 + 3) // 4 * 4
    seg_groups = g[np.array(order, np.int64)] if G else g
    nbits = 64
    while nbits <= G:  # one spare slot for table misses
        nbits *= 2

    def arr(x):
        return np.array(x if x else [0], np.int64)

    return PlanIndex(
        G=G,
        E=E,
        groups=g,
        sizes=sizes,
        seg_of=seg_of,
        seg_groups=seg_groups,
        gemms=tuple(gemms),
        cu_len=at,
        pairs_only=not suffix,
        nbits=nbits,
        pair_tab=pair_tab,
        sfx_tab=sfx_tab,
        s_off=np.array(s_off, np.int64),
        s_e3=arr(s_e3),
        s_e4=arr(s_e4),
        s_g=arr(s_g),
    )


# ---------------------------------------------------------------- scratch
class Scratch:
    """Work buffers for (T, K), allocated once and shareable by several lists:
    `index` is one `PlanIndex` or a sequence of them (sized for the largest).
    `nchunk` fixes the work split (best: the numba thread count); it does not
    change the result."""

    def __init__(self, T, K, index, nchunk):
        if not 0 < T <= MAX_T or not 2 <= K <= MAX_K:
            raise ValueError(f"need 0 < T <= {MAX_T} and 2 <= K <= {MAX_K}, got T={T} K={K}")
        indexes = index if isinstance(index, (list, tuple)) else [index]
        S = max(ix.G + ix.E for ix in indexes)
        R = K // 2  # disjoint groups per token
        self.T, self.K, self.C = T, K, nchunk
        self.sel_g = np.zeros((T, R), np.int32)
        self.sel_code = np.zeros((T, R), np.int32)
        self.used = np.zeros(T, np.int32)
        self.srt = np.zeros((T, K), np.int32)  # sorted valid expert ids
        self.nval = np.zeros(T, np.int32)  # valid lanes per token
        self._cnt = np.zeros(nchunk * S, np.int32)
        self._off = np.zeros(nchunk * S, np.int64)  # chunk offset inside a segment
        self._stg = np.zeros(nchunk * S, np.int64)  # chunk-private staging start
        self._seg_tot = np.zeros(S, np.int64)
        self.csum = np.zeros(nchunk + 1, np.int64)
        self.tmp = np.zeros(T * K, np.int32)  # staged rows: tok | code << 16

    def segment_arrays(self, S):
        """(cnt, off, stg, seg_tot) for S segments, contiguous views."""
        C = self.C
        return (
            self._cnt[: C * S].reshape(C, S),
            self._off[: C * S].reshape(C, S),
            self._stg[: C * S].reshape(C, S),
            self._seg_tot[:S],
        )


def alloc_out(T, K, index):
    """Per-round outputs (see the module docstring); row arrays sized T*K."""
    S = index.G + index.E
    return dict(
        seg_off=np.zeros(S + 1, np.int32),
        row_tok=np.zeros(T * K, np.int16),
        row_code=np.zeros(T * K, np.int16),
        perm=np.zeros(T * K, np.int32),
        moff=np.zeros(T + 1, np.int32),
        cu=np.zeros(index.cu_len, np.int32),
    )


# ---------------------------------------------------------------- pass 1
@_intrinsic
def _cttz64(typingctx, x):
    sig = _nbt.uint64(_nbt.uint64)

    def codegen(context, builder, signature, args):
        from llvmlite import ir

        return builder.cttz(args[0], ir.Constant(ir.IntType(1), 1))

    return sig, codegen


@nb.njit(inline="always")
def _sorted_lanes(ids, t, K, loc):
    """Valid (>= 0) expert ids of token t, ascending, into loc[:nv]; returns nv."""
    nv = 0
    for k in range(K):  # insertion sort, K <= 16
        e = np.int32(ids[t, k])
        if e < 0:
            continue
        j = nv
        while j > 0 and loc[j - 1] > e:
            loc[j] = loc[j - 1]
            j -= 1
        loc[j] = e
        nv += 1
    return nv


@nb.njit(inline="always")
def _greedy(hb, NW, code, sizes, pairs_only, sg, sc_):
    """Walk the hit bitmap in ascending group id (= priority) and accept every
    group whose lanes miss the used mask.  Branch-free accept; since groups
    are disjoint, at most K // 2 are accepted.  Clears the bitmap."""
    u = np.int32(0)
    r = 0
    for w in range(NW):
        x = hb[w]
        if x == np.uint64(0):
            continue
        hb[w] = np.uint64(0)
        while x != np.uint64(0):
            g = w * 64 + np.int32(_cttz64(x))
            x &= x - np.uint64(1)
            cd = code[g]
            m = (np.int32(1) << (cd & 15)) | (np.int32(1) << ((cd >> 4) & 15))
            if not pairs_only:
                if sizes[g] > 2:
                    m |= np.int32(1) << ((cd >> 8) & 15)
                    if sizes[g] > 3:
                        m |= np.int32(1) << ((cd >> 12) & 15)
            ok = np.int32((u & m) == 0)
            sg[r] = g
            sc_[r] = cd
            u |= m * ok
            r += ok
    return u, r


@nb.njit(inline="always")
def _emit(t, ch, u, r, nv, sg, sc_, loc, G, seg_of, sel_g, sel_code, used, srt, nval, cnt):
    """Store token t's selection and count its rows per segment."""
    for q in range(r):
        sel_g[t, q] = sg[q]
        sel_code[t, q] = sc_[q]
        cnt[ch, seg_of[sg[q]]] += 1
    for q in range(r, sel_g.shape[1]):
        sel_g[t, q] = -1
    n = r
    for k in range(nv):
        srt[t, k] = loc[k]
        if (u >> k) & 1 == 0:
            cnt[ch, G + loc[k]] += 1
            n += 1
    used[t] = u
    nval[t] = nv
    return n


@functools.lru_cache(maxsize=None)
def _select_kernels(nbits):
    """Pass-1 kernels for lists of G < nbits groups (nbits a power of two
    >= 64).  The hit bitmap and the code table have the compile-time size
    nbits; a table miss (-1) is masked onto the unused slot nbits - 1 with a
    zero bit, so the probes are branch-free."""
    NW = nbits // 64
    GM = nbits - 1

    @nb.njit(parallel=True, cache=False, boundscheck=False)
    def select_pairs(ids, pair_tab, sizes, seg_of, sel_g, sel_code, used, srt, nval, cnt, csum):
        """Pair-only lists: C(K, 2) probes of the pair table, greedy over the
        hit bitmap, per-chunk segment counts."""
        T, K = ids.shape
        C, S = cnt.shape
        G = sizes.size
        E = S - G
        R = sel_g.shape[1]
        csz = (T + C - 1) // C
        for ch in nb.prange(C):
            cnt[ch, :] = 0
            loc = np.empty(MAX_K, np.int32)
            hb = np.zeros(NW, np.uint64)
            code = np.zeros(nbits, np.int32)
            sg = np.empty(R + 1, np.int32)
            sc_ = np.empty(R + 1, np.int32)
            tot = 0
            for t in range(ch * csz, min(T, (ch + 1) * csz)):
                nv = _sorted_lanes(ids, t, K, loc)
                for i in range(nv - 1):
                    base = loc[i] * E
                    for j in range(i + 1, nv):
                        g = pair_tab[base + loc[j]]
                        gg = g & GM
                        hb[gg >> 6] |= np.uint64(g >= 0) << np.uint64(gg & 63)
                        code[gg] = np.int32(i) | (np.int32(j) << 4)
                u, r = _greedy(hb, NW, code, sizes, True, sg, sc_)
                tot += _emit(t, ch, u, r, nv, sg, sc_, loc, G, seg_of, sel_g, sel_code, used, srt, nval, cnt)
            csum[ch + 1] = tot

    @nb.njit(parallel=True, cache=False, boundscheck=False)
    def select_mixed(
        ids,
        pair_tab,
        sfx_tab,
        s_off,
        s_e3,
        s_e4,
        s_g,
        sizes,
        seg_of,
        sel_g,
        sel_code,
        used,
        srt,
        nval,
        cnt,
        csum,
    ):
        """Lists with triples / quads: as `select_pairs`, plus the suffixes of
        every (a, b) prefix of the token that starts a triple / quad, tested
        against the token's expert set."""
        T, K = ids.shape
        C, S = cnt.shape
        G = sizes.size
        E = S - G
        R = sel_g.shape[1]
        csz = (T + C - 1) // C
        for ch in nb.prange(C):
            cnt[ch, :] = 0
            loc = np.empty(MAX_K, np.int32)
            hb = np.zeros(NW, np.uint64)
            code = np.zeros(nbits, np.int32)
            sg = np.empty(R + 1, np.int32)
            sc_ = np.empty(R + 1, np.int32)
            member = np.zeros(E, np.int8)
            rank = np.zeros(E, np.int32)
            tot = 0
            for t in range(ch * csz, min(T, (ch + 1) * csz)):
                nv = _sorted_lanes(ids, t, K, loc)
                built = False  # member / rank of this token
                for i in range(nv - 1):
                    base = loc[i] * E
                    for j in range(i + 1, nv):
                        g = pair_tab[base + loc[j]]
                        gg = g & GM
                        hb[gg >> 6] |= np.uint64(g >= 0) << np.uint64(gg & 63)
                        cab = np.int32(i) | (np.int32(j) << 4)
                        code[gg] = cab
                        nid = sfx_tab[base + loc[j]]
                        if nid < 0:
                            continue
                        if not built:
                            for k in range(nv):
                                member[loc[k]] = 1
                                rank[loc[k]] = k
                            built = True
                        for q in range(s_off[nid], s_off[nid + 1]):
                            e3 = s_e3[q]
                            if member[e3] == 0:
                                continue
                            c3 = cab | (rank[e3] << 8)
                            e4 = s_e4[q]
                            if e4 >= 0:
                                if member[e4] == 0:
                                    continue
                                c3 |= rank[e4] << 12
                            g3 = s_g[q]
                            hb[g3 >> 6] |= np.uint64(1) << np.uint64(g3 & 63)
                            code[g3] = c3
                if built:
                    for k in range(nv):
                        member[loc[k]] = 0
                u, r = _greedy(hb, NW, code, sizes, False, sg, sc_)
                tot += _emit(t, ch, u, r, nv, sg, sc_, loc, G, seg_of, sel_g, sel_code, used, srt, nval, cnt)
            csum[ch + 1] = tot

    return select_pairs, select_mixed


# ---------------------------------------------------------------- layout
@nb.njit(parallel=True, cache=False, boundscheck=False)
def _prefix(cnt, off, csum, seg_tot, seg_off):
    """Chunk offsets inside every segment, segment offsets, row offset of
    every chunk."""
    C, S = cnt.shape
    for blk in nb.prange((S + 15) // 16):  # 16 segments per task: no false sharing
        for s in range(blk * 16, min(S, blk * 16 + 16)):
            acc = np.int64(0)
            for ch in range(C):
                off[ch, s] = acc
                acc += cnt[ch, s]
            seg_tot[s] = acc
    csum[0] = 0
    for ch in range(C):
        csum[ch + 1] += csum[ch]
    seg_off[0] = 0
    for s in range(S):
        seg_off[s + 1] = seg_off[s] + seg_tot[s]


@nb.njit(parallel=True, cache=False, boundscheck=False)
def _fill_a(srt, nval, sel_g, sel_code, used, seg_of, G, cnt, off, stg, csum, seg_off, tmp, perm, moff):
    """Pass 2a: per chunk, rows into its private staging area
    [csum[ch], csum[ch+1]) bucketed by segment (no false sharing); perm /
    moff are token-ordered, so written in place."""
    T, K = srt.shape
    C, S = off.shape
    R = sel_g.shape[1]
    csz = (T + C - 1) // C
    for ch in nb.prange(C):
        cur = np.empty(S, np.int64)  # next staging slot per segment
        dst = np.empty(S, np.int64)  # staging slot -> final row
        acc = csum[ch]
        for s in range(S):
            stg[ch, s] = acc
            cur[s] = acc
            dst[s] = seg_off[s] + off[ch, s] - acc
            acc += cnt[ch, s]
        o = csum[ch]
        for t in range(ch * csz, min(T, (ch + 1) * csz)):
            moff[t] = np.int32(o)
            tt = np.int32(t)
            for r in range(R):
                g = sel_g[t, r]
                if g < 0:
                    break
                s = seg_of[g]
                p = cur[s]
                cur[s] = p + 1
                tmp[p] = tt | (sel_code[t, r] << 16)
                perm[o] = np.int32(dst[s] + p)
                o += 1
            u = used[t]
            for k in range(nval[t]):
                if (u >> k) & 1 == 0:
                    s = G + srt[t, k]
                    p = cur[s]
                    cur[s] = p + 1
                    tmp[p] = tt | (np.int32(k) << 16)
                    perm[o] = np.int32(dst[s] + p)
                    o += 1
    moff[T] = np.int32(csum[C])


@nb.njit(parallel=True, cache=False, boundscheck=False)
def _fill_b(cnt, off, stg, seg_off, tmp, row_tok, row_code):
    """Pass 2b: per block of 16 segments, copy each chunk's run into place."""
    C, S = cnt.shape
    for blk in nb.prange((S + 15) // 16):
        for s in range(blk * 16, min(S, blk * 16 + 16)):
            for ch in range(C):
                src = stg[ch, s]
                dst = seg_off[s] + off[ch, s]
                for i in range(cnt[ch, s]):
                    x = tmp[src + i]
                    row_tok[dst + i] = np.int16(x)
                    row_code[dst + i] = np.int16(x >> 16)


def plan(ids, index, sc, out):
    """ids [T, K] int16/int32 expert ids of one round (see the module
    preconditions); out from `alloc_out`.  Returns nrow."""
    if ids.shape != (sc.T, sc.K):
        raise ValueError(f"ids shape {ids.shape} != scratch ({sc.T}, {sc.K})")
    ix = index
    cnt, off, stg, seg_tot = sc.segment_arrays(ix.G + ix.E)
    select_pairs, select_mixed = _select_kernels(ix.nbits)
    sel = (sc.sel_g, sc.sel_code, sc.used, sc.srt, sc.nval, cnt, sc.csum)
    if ix.pairs_only:
        select_pairs(ids, ix.pair_tab, ix.sizes, ix.seg_of, *sel)
    else:
        select_mixed(
            ids, ix.pair_tab, ix.sfx_tab, ix.s_off, ix.s_e3, ix.s_e4, ix.s_g, ix.sizes, ix.seg_of, *sel
        )
    seg_off = out["seg_off"]
    _prefix(cnt, off, sc.csum, seg_tot, seg_off)
    _fill_a(
        sc.srt,
        sc.nval,
        sc.sel_g,
        sc.sel_code,
        sc.used,
        ix.seg_of,
        ix.G,
        cnt,
        off,
        stg,
        sc.csum,
        seg_off,
        sc.tmp,
        out["perm"],
        out["moff"],
    )
    _fill_b(cnt, off, stg, seg_off, sc.tmp, out["row_tok"], out["row_code"])
    cu = out["cu"]
    for _, s0, s1, at in ix.gemms:
        cu[at : at + s1 - s0 + 1] = seg_off[s0 : s1 + 1]
    return int(seg_off[ix.G + ix.E])


# ---------------------------------------------------------------- list files
def save_list(path, groups):
    """Group list file: npz, key "groups" = int32 [G, 4] (-1 pad), priority order."""
    g = np.full((len(groups), MAX_C), -1, np.int32)
    for i, row in enumerate(groups):
        g[i, : len(row)] = row
    np.savez(path, groups=g)


def load_list(path):
    return [[int(e) for e in row if e >= 0] for row in np.load(path)["groups"]]


# ---------------------------------------------------------------- planner process
# One planner process per node serves every rank of the node (the repeated calls
# carry the same routing on all of them) and every MoE layer: a layer's list
# is a "table", and slot `table * nround + round` holds one round of it.
# Shared-memory block:
#   flags (int32 words), per slot s:
#     IN_SEQ   = q  local rank 0's GPU, after the ids of round q are in slot s
#     OUT_SEQ  = q  planner, after the plan of round q is in slot s
#     NROW, PLAN_US  of that plan
#     ACK + r  = q  rank r's GPU, after copying the plan of round q out
#   one ERROR word: set by any rank's GPU if a round's lanes do not match the
#     routing the plan was made from (never expected; raised by the client)
#   per slot: ids int16 [T*K] | the `alloc_out` arrays (sized for the largest list)
# The planner overwrites slot s only after every rank acked its previous plan;
# a negative IN_SEQ stops it.
IN_SEQ, OUT_SEQ, NROW, PLAN_US, ACK = range(5)
_ALIGN = 256


def flag_word(kind, slot, nslot, rank=0):
    """Index of a flag word (kind = IN_SEQ / OUT_SEQ / NROW / PLAN_US / ACK)."""
    return (kind + rank) * nslot + slot


def error_word(nslot, nrank):
    return (ACK + nrank) * nslot


def shm_layout(T, K, indexes, nround, nrank):
    """-> (flag words, {name: (offset, count, dtype)} of one slot, slot bytes,
    first slot offset, total bytes).  Offsets are relative to the slot; every
    slot is sized for the largest list."""
    nslot = len(indexes) * nround
    nflag = error_word(nslot, nrank) + 1
    S = max(ix.G + ix.E for ix in indexes)
    fields = [
        ("ids", T * K, np.int16),
        ("seg_off", S + 1, np.int32),
        ("row_tok", T * K, np.int16),
        ("row_code", T * K, np.int16),
        ("perm", T * K, np.int32),
        ("moff", T + 1, np.int32),
        ("cu", max(ix.cu_len for ix in indexes), np.int32),
    ]
    lay, o = {}, 0
    for name, cnt, dt in fields:
        lay[name] = (o, cnt, np.dtype(dt))
        o += (cnt * np.dtype(dt).itemsize + _ALIGN - 1) // _ALIGN * _ALIGN
    first = (nflag * 4 + 4095) // 4096 * 4096
    return nflag, lay, o, first, first + nslot * o


def slot_sizes(index, T, K):
    """Element counts of one slot's arrays for this list (its views)."""
    return dict(
        ids=T * K,
        seg_off=index.G + index.E + 1,
        row_tok=T * K,
        row_code=T * K,
        perm=T * K,
        moff=T + 1,
        cu=index.cu_len,
    )


def shm_views(buf, T, K, indexes, nround, nrank):
    """flags and, per slot, its arrays at the exact size of the slot's list."""
    nflag, lay, ssz, first, _ = shm_layout(T, K, indexes, nround, nrank)
    flags = np.ndarray((nflag,), np.int32, buffer=buf)
    slots = []
    for s in range(len(indexes) * nround):
        n = slot_sizes(indexes[s // nround], T, K)
        slots.append(
            {
                k: np.ndarray((n[k],), dt, buffer=buf, offset=first + s * ssz + o)
                for k, (o, _, dt) in lay.items()
            }
        )
    return flags, slots


def attach_shm(name):
    """Attach to an existing block without letting this process's resource
    tracker unlink it at exit (Python < 3.13 does that for every attacher);
    the creator owns the block."""
    from multiprocessing import resource_tracker, shared_memory

    shm = shared_memory.SharedMemory(name=name)
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm


def serve(shm_name, T, K, E, list_paths, threads, nround, nrank, round_order=None):
    """Planner loop (runs until a negative IN_SEQ).

    ``round_order`` is the priority in which rounds are served within every
    table (default ``0..nround-1``); a consumer that uses the rounds in a
    different order passes that order so the first-needed plan is produced
    first.
    """
    nb.set_num_threads(threads)
    indexes = [build_index(load_list(p), E) for p in list_paths]
    nslot = len(indexes) * nround
    shm = attach_shm(shm_name)
    flags, slots = shm_views(shm.buf, T, K, indexes, nround, nrank)
    sc = Scratch(T, K, indexes, threads)
    warm = np.random.RandomState(0).rand(T, E).argsort(1)[:, :K].astype(np.int16)
    for ix in indexes:  # JIT + first touch
        plan(warm, ix, sc, slots[0] if ix is indexes[0] else alloc_out(T, K, ix))
    if round_order is None:
        round_order = list(range(nround))
    order = [t * nround + r for r in round_order for t in range(len(indexes))]
    done = [0] * nslot
    ack = flags[flag_word(ACK, 0, nslot) : flag_word(ACK, 0, nslot) + nrank * nslot].reshape(nrank, nslot)
    out = None
    print("ready", flush=True)
    try:
        while True:  # polls without sleeping: a sleep costs ~70 us of timer slack
            for s in order:
                q = int(flags[flag_word(IN_SEQ, s, nslot)])
                if q < 0:
                    return
                if q <= done[s] or (ack[:, s] < done[s]).any():
                    continue
                t0 = time.perf_counter()
                out = slots[s]
                n = plan(out["ids"].reshape(T, K), indexes[s // nround], sc, out)
                flags[flag_word(NROW, s, nslot)] = n
                flags[flag_word(PLAN_US, s, nslot)] = int((time.perf_counter() - t0) * 1e6)
                done[s] = q
                flags[flag_word(OUT_SEQ, s, nslot)] = q  # publish last (x86-TSO store order)
    finally:
        flags = slots = ack = out = None  # drop the views before closing
        shm.close()


def _main():
    import argparse

    ap = argparse.ArgumentParser(description="row-group node planner process")
    ap.add_argument("--shm", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--E", type=int, required=True)
    ap.add_argument("--list", action="append", required=True, help="one per table, in table order")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--nround", type=int, required=True)
    ap.add_argument("--nrank", type=int, required=True)
    ap.add_argument("--round-order", default="", help="comma round indices in serve priority")
    a = ap.parse_args()
    ro = [int(x) for x in a.round_order.split(",") if x != ""] or None
    serve(a.shm, a.T, a.K, a.E, a.list, a.threads, a.nround, a.nrank, round_order=ro)


if __name__ == "__main__":
    _main()
