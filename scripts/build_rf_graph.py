"""Random-forest proximity graph: each user's k most proximate users under an unsupervised forest.

Breiman's unsupervised forest separates the real rows from a copy in which every column is
permuted on its own, which keeps the marginals and breaks the dependence between features. Two
users are proximate when they fall in the same leaf in many trees. Unlike Euclidean k-NN, the
proximity is invariant to monotone transformations of a feature and gives little weight to
features that do not separate real rows from permuted ones.

The forest (scikit-learn) is fit on a random sample of users and as many permuted rows, every
user is then passed down every tree, and each user keeps the k users it shares the most leaves
with. Exact top-k proximity is quadratic, so it is found in two steps: each user draws
--per-leaf random members of its own leaf in every tree, the 4k candidates drawn most often are
kept, and those are ranked by their exact proximity over all trees.

Saves the ranked k=10 table (rf_table.npy, rf_prox.npy) and edge lists rfprox{k}.npy in the
format of build_graphs.py, so graph_features.py (rfprox2_agg), train_gnn.py (--graph rfprox2) and
train_embed.py (--add-knn rfprox2) read them as they read knn{k}.

Run: uv run python scripts/build_rf_graph.py --sample 200000 --trees 100
"""

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import paths

GRAPHS = paths.GRAPHS
K_MAX = 10


# -------------------------------------------------------------------------------
# Forest
# -------------------------------------------------------------------------------
def fit_forest(x: np.ndarray, trees: int, min_leaf: int, seed: int) -> RandomForestClassifier:
    """Fit real rows (label 1) against a column-wise permuted copy (label 0)."""
    rng = np.random.default_rng(seed)
    permuted = np.column_stack([rng.permutation(col) for col in x.T])
    rf = RandomForestClassifier(
        n_estimators=trees, max_features="sqrt", min_samples_leaf=min_leaf, n_jobs=-1, random_state=seed
    )
    rf.fit(np.vstack([x, permuted]), np.r_[np.ones(len(x)), np.zeros(len(x))])
    return rf


def leaf_matrix(rf: RandomForestClassifier, x: np.ndarray, block: int = 500_000) -> np.ndarray:
    """Leaf of every row in every tree, (n, trees) int32."""
    return np.vstack([rf.apply(x[i : i + block]).astype(np.int32) for i in range(0, len(x), block)])


# -------------------------------------------------------------------------------
# Top-k proximity
# -------------------------------------------------------------------------------
def shuffled_leaves(leaves: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per tree, the rows sorted by leaf in random order within each leaf.

    Returns members (trees, n), each row's position in members (n, trees), and the start and
    size of its leaf there, packed as (n, trees, 2).
    """
    n, trees = leaves.shape
    members = np.empty((trees, n), dtype=np.int32)
    pos = np.empty((n, trees), dtype=np.int32)
    span = np.empty((n, trees, 2), dtype=np.int32)
    for t in range(trees):
        col = leaves[:, t]
        order = np.lexsort((rng.permutation(n), col)).astype(np.int32)
        members[t] = order
        pos[order, t] = np.arange(n, dtype=np.int32)
        _, first, size = np.unique(col[order], return_index=True, return_counts=True)
        leaf_id = np.repeat(np.arange(len(first)), size)
        span[order, t, 0] = first[leaf_id]
        span[order, t, 1] = size[leaf_id]
    return members, pos, span


def topk(leaves: np.ndarray, k: int, per_leaf: int, seed: int = 0, block: int = 10_000) -> tuple:
    """Each row's k most proximate rows, (n, k) indices (-1 when missing) and proximities."""
    n, trees = leaves.shape
    rng = np.random.default_rng(seed)
    members, pos, span = shuffled_leaves(leaves, rng)
    shortlist = 4 * k
    idx = np.full((n, k), -1, dtype=np.int64)
    prox = np.zeros((n, k), dtype=np.float32)
    steps = np.arange(1, per_leaf + 1, dtype=np.int64)
    tree_ids = np.arange(trees)
    for lo in range(0, n, block):
        rows = np.arange(lo, min(lo + block, n))
        m = len(rows)
        # the per_leaf members that follow the row in its shuffled leaf, wrapping within the leaf
        start, size = span[rows, :, 0].astype(np.int64), span[rows, :, 1].astype(np.int64)
        offset = (pos[rows].astype(np.int64) - start)[:, :, None] + steps
        q = start[:, :, None] + offset % size[:, :, None]
        cand = members[tree_ids[None, :, None], q]
        # a leaf smaller than per_leaf + 1 would wrap around to the row itself and repeat members
        cand = np.where(steps < size[:, :, None], cand, -1).reshape(m, -1)
        # count how often each candidate was drawn: sort each row and measure its runs
        cand.sort(axis=1)
        c = cand.shape[1]
        cols = np.arange(c)
        new_run = np.ones_like(cand, dtype=bool)
        new_run[:, 1:] = cand[:, 1:] != cand[:, :-1]
        # position of the next run start after each column, so a run's length is next - start
        nxt = np.full(cand.shape, c)
        nxt[:, :-1] = np.where(new_run[:, 1:], cols[1:], c)
        nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
        head = new_run & (cand >= 0)
        first = np.where(head, cand, -1)
        drawn = np.where(head, nxt - cols, -1)
        sel = np.argpartition(-drawn, min(shortlist, drawn.shape[1] - 1), axis=1)[:, :shortlist]
        short = np.take_along_axis(first, sel, axis=1)
        # exact proximity of the shortlist over all trees; ties broken by a random draw
        exact = (leaves[rows][:, None, :] == leaves[np.maximum(short, 0)]).mean(2)
        exact = np.where(short >= 0, exact + rng.random(exact.shape) * 1e-6, -1.0)
        best = np.argsort(-exact, axis=1)[:, :k]
        chosen, score = np.take_along_axis(short, best, axis=1), np.take_along_axis(exact, best, axis=1)
        idx[rows] = np.where(score >= 0, chosen, -1)
        prox[rows] = np.where(score >= 0, np.floor(score * trees + 1e-3) / trees, 0.0)
    return idx, prox


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=200_000, help="real users the forest is fit on")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--min-leaf", type=int, default=5)
    ap.add_argument("--per-leaf", type=int, default=64, help="leaf members each user draws per tree")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 6])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # the same preprocessed features as the k-NN graph
    x = np.load(GRAPHS / "x_knn.npy")
    n = x.shape[0]
    rows = np.sort(np.random.default_rng(args.seed).choice(n, size=min(args.sample, n), replace=False))
    t0 = time.time()
    rf = fit_forest(x[rows], args.trees, args.min_leaf, args.seed)
    t_fit = time.time() - t0
    print(f"fit on {len(rows):,} users + {len(rows):,} permuted rows: {t_fit:.0f}s", flush=True)

    leaves = leaf_matrix(rf, x)
    t_apply = time.time() - t0 - t_fit
    sizes = np.concatenate([np.unique(leaves[:, t], return_counts=True)[1] for t in range(leaves.shape[1])])
    # the size of the leaf a typical user sits in weights each leaf by its own size
    weighted = np.repeat(sizes, sizes)
    print(
        f"apply to {n:,} users: {t_apply:.0f}s; leaf sizes median {np.median(sizes):.0f}, "
        f"user-weighted median {np.median(weighted):.0f}, largest {sizes.max():,}",
        flush=True,
    )

    idx, prox = topk(leaves, K_MAX, args.per_leaf, args.seed)
    t_topk = time.time() - t0 - t_fit - t_apply
    print(
        f"top-{K_MAX}: {t_topk:.0f}s; proximity of 1st/2nd/6th neighbor (median) "
        f"{np.median(prox[:, 0]):.2f}/{np.median(prox[:, 1]):.2f}/{np.median(prox[:, 5]):.2f}, "
        f"users with fewer than {K_MAX} neighbors {(idx[:, -1] < 0).mean():.2%}",
        flush=True,
    )
    np.save(GRAPHS / "rf_table.npy", idx)
    np.save(GRAPHS / "rf_prox.npy", prox)
    for k in args.k:
        src, dst = np.repeat(np.arange(n), k), idx[:, :k].reshape(-1)
        keep = dst >= 0
        ei = np.stack([src[keep], dst[keep]])
        np.save(GRAPHS / f"rfprox{k}.npy", ei)
        print(f"rfprox{k}: {ei.shape[1]:,} directed edges")
    meta = {**vars(args), "fit_s": round(t_fit), "apply_s": round(t_apply), "topk_s": round(t_topk)}
    (GRAPHS / "rf_graph.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
