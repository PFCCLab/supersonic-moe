"""CPU planner: exact against a direct Python implementation of the spec, edge
coverage, and determinism across thread / chunk counts."""

import itertools

import numpy as np
import pytest

from sonicmoe import row_group_planner as P


def skewed_routing(seed, T, E=256, K=8):
    """Top-k ids with a Zipf-like expert popularity (frequent group hits)."""
    rng = np.random.RandomState(seed)
    p = 1.0 / np.arange(1, E + 1) ** 0.8
    p = p[rng.permutation(E)]
    p /= p.sum()
    return np.stack([rng.choice(E, K, replace=False, p=p) for _ in range(T)])


def reference(ids, groups, E):
    """Spec, written plainly: greedy by list order over lane-disjoint groups of
    the token's sorted experts; segments by (size desc, priority) then experts."""
    g, sizes = P.normalize_groups(groups, E)
    G = len(g)
    lookup = {tuple(r[:s]): i for i, (r, s) in enumerate(zip(g, sizes))}
    order = sorted(range(G), key=lambda i: (-int(sizes[i]), i))
    seg_of = {gi: si for si, gi in enumerate(order)}
    T, K = ids.shape
    rows = [[] for _ in range(G + E)]  # (token, code)
    tok_rows = []
    for t in range(T):
        srt = sorted(int(x) for x in ids[t] if x >= 0)  # -1 = dropped lane
        nv = len(srt)
        cand = []
        for c in (2, 3, 4):
            for lanes in itertools.combinations(range(nv), c):
                key = tuple(srt[l] for l in lanes)
                if key in lookup:
                    cand.append((lookup[key], lanes))
        cand.sort()
        used, mine = 0, []
        for gi, lanes in cand:
            m = sum(1 << l for l in lanes)
            if used & m == 0 and len(mine) < K // 2:
                used |= m
                code = sum(l << (4 * j) for j, l in enumerate(lanes))
                mine.append((seg_of[gi], code))
        for k in range(nv):
            if not used >> k & 1:
                mine.append((G + srt[k], k))
        tok_rows.append(mine)
        for s, code in mine:
            rows[s].append((t, code))
    seg_off = np.zeros(G + E + 1, np.int64)
    for s in range(G + E):
        seg_off[s + 1] = seg_off[s] + len(rows[s])
    row_tok = np.concatenate([[t for t, _ in r] for r in rows if r]).astype(np.int64)
    row_code = np.concatenate([[c for _, c in r] for r in rows if r]).astype(np.int64)
    # perm: token order, rows in the order the token produced them
    pos = {}
    for s in range(G + E):
        for j, (t, code) in enumerate(rows[s]):
            pos.setdefault((t, s), []).append(seg_off[s] + j)
    perm, moff = [], [0]
    for t in range(T):
        for s, code in tok_rows[t]:
            perm.append(pos[(t, s)].pop(0))
        moff.append(len(perm))
    return seg_off, row_tok, row_code, np.array(perm), np.array(moff)


def run(ids, groups, E, nchunk, threads=8):
    import numba

    numba.set_num_threads(threads)
    idx = P.build_index(groups, E)
    T, K = ids.shape
    sc = P.Scratch(T, K, idx, nchunk)
    out = P.alloc_out(T, K, idx)
    n = P.plan(ids.astype(np.int16), idx, sc, out)
    return n, out, idx


def random_groups(rng, E, n, sizes=(2, 3, 4)):
    seen, out = set(), []
    while len(out) < n:
        c = int(rng.choice(sizes))
        g = tuple(sorted(rng.choice(E, c, replace=False).tolist()))
        if g not in seen:
            seen.add(g)
            out.append(list(g))
    return out


@pytest.mark.parametrize("source", ["random", "random_k10", "pairs", "skewed"])
def test_planner_matches_reference(source):
    rng = np.random.RandomState(1)
    if source != "skewed":
        # random_k10: quads on lanes up to 9 -> codes above the int16 range;
        # pairs: the pair-only pass
        E, K, T = {"random": (64, 8, 3000), "random_k10": (48, 10, 3000), "pairs": (64, 8, 3000)}[source]
        ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
        groups = random_groups(rng, E, 300, sizes=(2,) if source == "pairs" else (2, 3, 4))
    else:
        ids = skewed_routing(0, 2000)
        E, K = 256, ids.shape[1]
        T = ids.shape[0]
        # the most frequent subsets of the batch, mixed sizes, as a realistic list
        from collections import Counter

        cnt = Counter()
        for r in np.sort(ids, 1):
            for c in (2, 3, 4):
                for comb in itertools.combinations(r.tolist(), c):
                    cnt[comb] += 1
        groups = [list(k) for k, v in cnt.most_common(400)]
        rng.shuffle(groups)
    n, out, idx = run(ids, groups, E, nchunk=7)
    seg_off, row_tok, row_code, perm, moff = reference(ids, groups, E)
    assert n == seg_off[-1]
    assert idx.pairs_only == (source == "pairs")
    if source == "random_k10":
        assert row_code.max() >= 1 << 15  # the case is exercised
    _check(n, out, (seg_off, row_tok, row_code, perm, moff))
    _check_cu(out, idx)


def _check_cu(out, idx):
    """Per-GEMM cu slices: aligned, consecutive, covering all segments once."""
    nxt = 0
    for c, s0, s1, at in idx.gemms:
        assert at % 4 == 0 and s0 == nxt
        assert np.array_equal(out["cu"][at : at + s1 - s0 + 1], out["seg_off"][s0 : s1 + 1])
        nxt = s1
    assert nxt == idx.G + idx.E


def _check(n, out, ref):
    seg_off, row_tok, row_code, perm, moff = ref
    assert n == seg_off[-1]
    assert np.array_equal(out["seg_off"], seg_off)
    assert np.array_equal(out["row_tok"][:n].view(np.uint16).astype(np.int64), row_tok)
    assert np.array_equal(out["row_code"][:n].view(np.uint16).astype(np.int64), row_code)
    assert np.array_equal(out["perm"][:n], perm)
    assert np.array_equal(out["moff"], moff)


@pytest.mark.parametrize("G", [1, 63, 64, 65, 127, 128])
@pytest.mark.parametrize("sizes", [(2,), (2, 3, 4)])
def test_list_sizes_at_bitmap_boundaries(G, sizes):
    """The hit bitmap is sized to the next power of two above G (spare slot
    for table misses): lists right at the boundaries."""
    rng = np.random.RandomState(G)
    E, K, T = 24, 8, 1500
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
    groups = random_groups(rng, E, G, sizes=sizes)
    n, out, idx = run(ids, groups, E, nchunk=5)
    assert idx.nbits > G
    _check(n, out, reference(ids, groups, E))


def test_every_edge_covered_once():
    ids = skewed_routing(1, 8192)
    T, K = ids.shape
    E = 256
    rng = np.random.RandomState(2)
    groups = random_groups(rng, E, 2000)
    n, out, idx = run(ids, groups, E, nchunk=32)
    srt = np.sort(ids, 1)
    G = idx.G
    seg_of_row = np.repeat(np.arange(G + E), np.diff(out["seg_off"]))
    covered = np.zeros((T, K), np.int32)
    tok = out["row_tok"][:n].view(np.uint16).astype(np.int64)
    code = out["row_code"][:n].view(np.uint16).astype(np.int64)
    sizes = np.concatenate([(idx.seg_groups >= 0).sum(1), np.ones(E, np.int64)])
    for r in range(n):
        s = seg_of_row[r]
        c = sizes[s]
        lanes = [(code[r] >> (4 * j)) & 15 for j in range(c)]
        exp = idx.seg_groups[s, :c] if s < G else np.array([s - G])
        assert np.array_equal(srt[tok[r], lanes], exp), (r, s)
        covered[tok[r], lanes] += 1
    assert (covered == 1).all()
    # combine view: every row referenced exactly once, rows of token t belong to t
    assert np.array_equal(np.sort(out["perm"][:n]), np.arange(n))
    for t in range(0, T, 97):
        assert (tok[out["perm"][out["moff"][t] : out["moff"][t + 1]]] == t).all()


def test_deterministic_across_chunks_and_threads():
    ids = skewed_routing(2, 16384)
    rng = np.random.RandomState(3)
    groups = random_groups(rng, 256, 1500)
    ref = None
    for nchunk, th in ((1, 1), (16, 4), (256, 32), (97, 8)):
        n, out, _ = run(ids, groups, 256, nchunk=nchunk, threads=th)
        cur = (n,) + tuple(
            out[k][: n if k in ("row_tok", "row_code", "perm") else None].tobytes()
            for k in ("seg_off", "row_tok", "row_code", "perm", "moff")
        )
        if ref is None:
            ref = cur
        assert cur == ref, (nchunk, th)


def test_bad_lists_rejected():
    with pytest.raises(ValueError):
        P.build_index([[1, 1]], 8)
    with pytest.raises(ValueError):
        P.build_index([[1, 2], [2, 1]], 8)
    with pytest.raises(ValueError):
        P.build_index([[1, 9]], 8)
    with pytest.raises(ValueError):
        P.build_index([[1, 2, 3, 4, 5]], 8)


def test_bad_shapes_rejected():
    idx = P.build_index([[1, 2]], 8)
    with pytest.raises(ValueError):
        P.Scratch(65537, 8, idx, 4)
    with pytest.raises(ValueError):
        P.Scratch(16, 17, idx, 4)
    sc = P.Scratch(16, 4, idx, 4)
    with pytest.raises(ValueError):
        P.plan(np.zeros((16, 5), np.int16), idx, sc, P.alloc_out(16, 4, idx))


@pytest.mark.parametrize("seed", range(12))
def test_fuzz_small(seed):
    """Small random cases incl. empty lists, T = 1, more chunks than tokens."""
    rng = np.random.RandomState(100 + seed)
    E = int(rng.choice([4, 9, 16, 40]))
    K = int(rng.randint(2, min(E, 16) + 1))
    T = int(rng.choice([1, 2, 17, 300]))
    G = int(rng.randint(0, min(60, E * (E - 1) // 2) + 1))
    sizes = tuple(c for c in (2, 3, 4) if c <= E)
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
    groups = random_groups(rng, E, G, sizes=sizes)
    nchunk = int(rng.choice([1, 3, T + 5]))
    n, out, idx = run(ids, groups, E, nchunk=nchunk, threads=4)
    _check(n, out, reference(ids, groups, E))
    _check_cu(out, idx)


@pytest.mark.parametrize("sizes", [(2,), (2, 3, 4)])
def test_dropped_lanes(sizes):
    """-1 lanes (dropped) take no part: same result as the reference, every
    valid lane covered once, nothing emitted for dropped lanes."""
    rng = np.random.RandomState(11)
    E, K, T = 32, 8, 2000
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
    drop = rng.rand(T, K) < 0.15
    drop[:50] = True  # whole tokens dropped
    ids[drop] = -1
    groups = random_groups(rng, E, 200, sizes=sizes)
    n, out, idx = run(ids, groups, E, nchunk=6)
    _check(n, out, reference(ids, groups, E))
    assert n <= int((ids >= 0).sum())
    assert (np.diff(out["moff"])[:50] == 0).all()


def test_scratch_shared_by_lists():
    """One scratch sized for the largest list serves smaller ones."""
    rng = np.random.RandomState(12)
    E, K, T = 40, 8, 1500
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
    lists = [random_groups(rng, E, g, sizes=(2, 3)) for g in (300, 20, 0)]
    indexes = [P.build_index(g, E) for g in lists]
    sc = P.Scratch(T, K, indexes, 5)
    for g, ix in zip(lists, indexes):
        out = P.alloc_out(T, K, ix)
        n = P.plan(ids.astype(np.int16), ix, sc, out)
        _check(n, out, reference(ids, g, E))
        _check_cu(out, ix)
