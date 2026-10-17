"""Build similarity graphs over all DGraph-Fin nodes.

knn: exact k nearest neighbors (Euclidean, FAISS IndexFlatL2 on CPU, ~1 h) in
the preprocessed feature space. A tiny jitter (sd 1e-3) breaks ties, since many
users share identical feature vectors (most features are missing for ~55% of
nodes). Saves the ranked k=10 neighbor table and edge lists knn{k}.npy.

FAISS and torch clash over OpenMP in one process on macOS, so this script does
not import torch; the jittered input is written by `--prepare` in a separate run.

Run: uv run python scripts/build_graphs.py --prepare
     KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/build_graphs.py --k 2 6
"""

import argparse
import time

import numpy as np

import paths

GRAPHS = paths.GRAPHS
X_KNN = GRAPHS / "x_knn.npy"
K_MAX = 10


def prepare() -> None:
    import torch

    import dgraph_data

    g = dgraph_data.load()
    gen = torch.Generator().manual_seed(0)
    x = g.x + 1e-3 * torch.randn(g.x.shape, generator=gen)
    GRAPHS.mkdir(parents=True, exist_ok=True)
    np.save(X_KNN, x.numpy().astype(np.float32))


# -------------------------------------------------------------------------------
# Exact k-NN with FAISS
# -------------------------------------------------------------------------------
def knn_table(x: np.ndarray, k: int, block: int = 100_000) -> np.ndarray:
    import faiss

    faiss.omp_set_num_threads(10)
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    n = x.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    t0 = time.time()
    for s in range(0, n, block):
        _, i = index.search(x[s : s + block], k + 1)
        rows = np.arange(s, min(s + block, n))
        # drop self (normally column 0; fall back to the last column if self is not returned)
        not_self = i != rows[:, None]
        keep = np.where(not_self.all(1, keepdims=True), np.arange(k + 1) < k, not_self)
        out[s : s + block] = i[keep].reshape(-1, k)
        done = min(s + block, n) / n
        print(f"knn {done:.1%} {time.time() - t0:.0f}s (eta {(time.time() - t0) / done * (1 - done):.0f}s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 6])
    args = ap.parse_args()
    if args.prepare:
        prepare()
        return

    table_path = GRAPHS / "knn_table.npy"
    if table_path.exists():
        table = np.load(table_path)
    else:
        table = knn_table(np.load(X_KNN), K_MAX)
        np.save(table_path, table)
    n = table.shape[0]
    for k in args.k:
        ei = np.stack([np.repeat(np.arange(n), k), table[:, :k].reshape(-1)])
        np.save(GRAPHS / f"knn{k}.npy", ei)
        print(f"knn{k}: {ei.shape[1]:,} directed edges")


if __name__ == "__main__":
    main()
