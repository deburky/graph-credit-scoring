"""DGraph-Fin loading, feature preprocessing and splits.

Source: Hugging Face mirror cryoushiwo/DGraph (DGraph/raw/dgraph.npz,
sha256 038f1b0a...ff43cb). This is the competition release: official test
labels are hidden (y == -100) and three uniform-noise columns (17-19, AUC 0.50)
were appended, which we drop to recover the 17 published features. Columns 0-4
also carry a small per-row perturbation (about +0.0003, so the missing code -1
appears as -0.999), so missing values are detected within MISSING_TOL of -1.

Nodes are Finvolution users; a directed edge a -> b means a listed b as an
emergency contact. Labels: 0 normal, 1 fraud/default (overdue), 2-3 background
users with no target but present in the graph.

Splits: the official train mask is kept as train; the official valid mask is
split 50/50 (stratified, seeded) into our val and test, since the official
test labels are not public.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import paths

RAW = paths.RAW
CACHE = paths.DATA / "processed.pt"
N_FEATURES = 17
MISSING_TOL = 0.01


@dataclass
class DGraph:
    x: torch.Tensor  # (n, d) float32, preprocessed node features
    x_raw: np.ndarray  # (n, 17) float32, original features (-1 = missing)
    y: torch.Tensor  # (n,) int64, labels in {0, 1}; -1 for unlabeled nodes
    edge_index: torch.Tensor  # (2, m) int64, directed contact edges
    edge_type: torch.Tensor  # (m,) int64
    edge_time: torch.Tensor  # (m,) int64
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]


# -------------------------------------------------------------------------------
# Feature preprocessing
# -------------------------------------------------------------------------------
def preprocess(x_raw: np.ndarray) -> np.ndarray:
    """Missing indicators for -1 entries, then clip at 0.1/99.9 pct and standardize."""
    missing = np.abs(x_raw + 1) < MISSING_TOL
    miss_cols = np.where(missing.any(0))[0]
    x = x_raw.astype(np.float64).copy()
    for j in range(x.shape[1]):
        obs = ~missing[:, j]
        col = x[obs, j]
        lo, hi = np.quantile(col, [0.001, 0.999])
        col = np.clip(col, lo, hi)
        mu, sd = col.mean(), col.std() + 1e-8
        x[:, j] = 0.0
        x[obs, j] = (col - mu) / sd
    return np.hstack([x, missing[:, miss_cols]]).astype(np.float32)


# -------------------------------------------------------------------------------
# Loading
# -------------------------------------------------------------------------------
def load(seed: int = 0) -> DGraph:
    if CACHE.exists():
        return DGraph(**torch.load(CACHE, weights_only=False))
    d = np.load(RAW, allow_pickle=False)
    x_raw = d["x"][:, :N_FEATURES].astype(np.float32)
    y_full = d["y"]
    y = np.where((y_full == 0) | (y_full == 1), y_full, -1)
    valid = d["valid_mask"]
    val, test = train_test_split(valid, test_size=0.5, stratify=y[valid], random_state=seed)
    g = DGraph(
        x=torch.from_numpy(preprocess(x_raw)),
        x_raw=x_raw,
        y=torch.from_numpy(y).long(),
        edge_index=torch.from_numpy(d["edge_index"].T.copy()).long(),
        edge_type=torch.from_numpy(d["edge_type"]).long(),
        edge_time=torch.from_numpy(d["edge_timestamp"]).long(),
        train_idx=torch.from_numpy(np.sort(d["train_mask"])).long(),
        val_idx=torch.from_numpy(np.sort(val)).long(),
        test_idx=torch.from_numpy(np.sort(test)).long(),
    )
    torch.save(vars(g), CACHE)
    return g


if __name__ == "__main__":
    g = load()
    deg = torch.bincount(g.edge_index.flatten(), minlength=g.num_nodes)
    print(f"nodes={g.num_nodes:,} edges={g.edge_index.shape[1]:,} features={g.x.shape[1]}")
    for name, idx in [("train", g.train_idx), ("val", g.val_idx), ("test", g.test_idx)]:
        print(f"{name}: n={len(idx):,} fraud_rate={g.y[idx].float().mean():.4f}")
    labeled = g.y >= 0
    print(f"isolated nodes: all={(deg == 0).float().mean():.3f} labeled={(deg[labeled] == 0).float().mean():.3f}")
    print(f"mean degree: {deg.float().mean():.2f}, max={deg.max()}")
