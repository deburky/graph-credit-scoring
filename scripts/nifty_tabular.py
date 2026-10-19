"""Data loading and graph-free baselines for NIFTY's German credit and Taiwan credit-default graphs.

Loads both datasets exactly as NIFTY's utils.py does (features, label coding, the similarity edges
shipped with the repo, the fixed split from random.seed(20), and its feature scaling) and scores,
with several seeds:

    logit, lgbm          logistic regression and LightGBM on the node features
    logit_nbr, lgbm_nbr  the same on the node features plus the mean of each borrower's graph
                         neighbors' features (one explicit smoothing step, no GNN)

nifty_baseline.py imports the loading code and runs the GNNs. The two stay in separate processes
because LightGBM segfaults when torch was imported first (each bundles an OpenMP runtime), and
this module never imports torch.

Run: uv run python scripts/nifty_tabular.py --dataset german --seeds 0 1 2 3 4
"""

import argparse
import csv
import json
import os
import random
import zipfile

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import paths

NIFTY = paths.ROOT / os.environ.get("NIFTY_DIR", "nifty/dataset")
RESULTS = paths.OUTPUTS / "results_nifty.jsonl"
# dataset: (label column, columns NIFTY drops, label_number, scale features to [-1, 1], sensitive column)
SPECS = {
    "german": ("GoodCustomer", ["OtherLoansAtStore", "PurposeOfLoan"], 100, False, 0),
    "credit": ("NoDefaultNextMonth", ["Single"], 6000, True, 1),
}


# -------------------------------------------------------------------------------
# Data, as in NIFTY's utils.py and nifty_sota_gnn.py
# -------------------------------------------------------------------------------
def read_edges(name: str) -> np.ndarray:
    path = NIFTY / name / f"{name}_edges.txt"
    if path.exists():
        return np.loadtxt(path).astype(np.int64)
    with (
        zipfile.ZipFile(path.with_suffix(".txt.zip")) as z,
        z.open(f"{name}_edges.txt") as f,
    ):
        return np.loadtxt(f).astype(np.int64)


def load(name: str, labels: int | None = None) -> dict:
    """labels: the cap on training labels (NIFTY's label_number); None keeps NIFTY's value."""
    label, dropped, label_number, scale, sens_idx = SPECS[name]
    label_number = labels or label_number
    with (NIFTY / name / f"{name}.csv").open() as f:
        records = list(csv.DictReader(f))
    if name == "german":
        for r in records:
            r["Gender"] = "1" if r["Gender"] == "Female" else "0"
    header = [c for c in records[0] if c != label and c not in dropped]
    x = np.array([[float(r[c]) for c in header] for r in records], dtype=np.float32)
    y = np.array([int(float(r[label])) for r in records])
    y[y == -1] = 0
    if scale:
        # feature_norm, keeping the sensitive column as it is
        lo, hi = x.min(0), x.max(0)
        scaled = 2 * (x - lo) / (hi - lo) - 1
        scaled[:, sens_idx] = x[:, sens_idx]
        x = scaled

    # symmetric adjacency with self-loops, as NIFTY builds it (the same nonzero pattern)
    e = read_edges(name)
    n = len(y)
    loops = np.tile(np.arange(n), (2, 1))
    edge_index = np.unique(np.hstack([e.T, e[:, ::-1].T, loops]), axis=1)

    # the fixed split: random.seed(20), half of each class for training (capped), a quarter each for val and test
    random.seed(20)
    # shuffled as lists, which gives the same order as NIFTY's in-place shuffle of the arrays
    order0, order1 = list(np.where(y == 0)[0]), list(np.where(y == 1)[0])
    random.shuffle(order0)
    random.shuffle(order1)
    idx0, idx1 = np.array(order0), np.array(order1)
    n0, n1 = len(idx0), len(idx1)
    train = np.append(
        idx0[: min(int(0.5 * n0), label_number // 2)],
        idx1[: min(int(0.5 * n1), label_number // 2)],
    )
    val = np.append(idx0[int(0.5 * n0) : int(0.75 * n0)], idx1[int(0.5 * n1) : int(0.75 * n1)])
    test = np.append(idx0[int(0.75 * n0) :], idx1[int(0.75 * n1) :])
    return {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "train": train,
        "val": val,
        "test": test,
    }


def neighbor_mean(x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Mean of each borrower's neighbors' features, self-loops excluded."""
    src, dst = edge_index[:, edge_index[0] != edge_index[1]]
    total = np.zeros_like(x)
    np.add.at(total, dst, x[src])
    deg = np.bincount(dst, minlength=len(x))
    return total / np.maximum(deg, 1)[:, None]


# -------------------------------------------------------------------------------
# Graph-free baselines
# -------------------------------------------------------------------------------
def logit(x: np.ndarray, d: dict) -> np.ndarray:
    """Logistic regression on standardized features, C chosen by validation AUC."""
    xs = StandardScaler().fit(x[d["train"]]).transform(x)
    best_auc, best_p = -1.0, np.zeros(len(x))
    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=c, max_iter=5000).fit(xs[d["train"]], d["y"][d["train"]])
        p = clf.predict_proba(xs)[:, 1]
        auc = roc_auc_score(d["y"][d["val"]], p[d["val"]])
        if auc > best_auc:
            best_auc, best_p = auc, p
    return best_p


def lightgbm(x: np.ndarray, d: dict, seed: int) -> np.ndarray:
    """LightGBM with early stopping on validation AUC; small leaves for German's 100 training rows."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_data_in_leaf": 5 if len(d["train"]) < 500 else 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": seed,
    }
    booster = lgb.train(
        params,
        lgb.Dataset(x[d["train"]], d["y"][d["train"]]),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(x[d["val"]], d["y"][d["val"]])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return np.asarray(booster.predict(x, num_iteration=booster.best_iteration))


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def write(rows: list[dict]) -> None:
    with RESULTS.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


def row(dataset: str, model: str, seed: int, p: np.ndarray, d: dict) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "labels": len(d["train"]),
        "test_auc": float(roc_auc_score(d["y"][d["test"]], p[d["test"]])),
        "val_auc": float(roc_auc_score(d["y"][d["val"]], p[d["val"]])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(SPECS), default="german")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--labels", type=int, default=None, help="cap on training labels (default: NIFTY's)")
    args = ap.parse_args()

    d = load(args.dataset, args.labels)
    x = d["x"]
    x_nbr = np.hstack([x, neighbor_mean(x, d["edge_index"])])
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        write(
            [
                row(args.dataset, "logit", seed, logit(x, d), d),
                row(args.dataset, "logit_nbr", seed, logit(x_nbr, d), d),
                row(args.dataset, "lgbm", seed, lightgbm(x, d, seed), d),
                row(args.dataset, "lgbm_nbr", seed, lightgbm(x_nbr, d, seed), d),
            ]
        )


if __name__ == "__main__":
    main()
