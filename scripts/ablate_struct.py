"""Which parts of the real-graph structure block carry the lift (LightGBM).

real_struct columns: 0 out-degree, 1 in-degree, 2-23 out/in counts for contact
types 1-11, 24-27 edge-time mean/min/max/span.

Run: uv run python scripts/ablate_struct.py
"""

import json

import lightgbm as lgb  # load before torch (OpenMP), see train_tabular.py
import numpy as np
from sklearn.metrics import roc_auc_score

import dgraph_data
import graph_features
import paths

RESULTS = paths.OUTPUTS / "results_ablation.jsonl"
GROUPS = {"degree": [0, 1], "types": list(range(2, 24)), "time": list(range(24, 28))}


def fit(x: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> float:
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
        seed=0,
        num_threads=10,
    )
    booster = lgb.train(
        params,
        lgb.Dataset(x[tr], y[tr]),
        3000,
        valid_sets=[lgb.Dataset(x[va], y[va])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return roc_auc_score(y[te], booster.predict(x[te], num_iteration=booster.best_iteration))


def main() -> None:
    g = dgraph_data.load()
    node, struct = graph_features.build("self", g), graph_features.build("real_struct", g)
    y = g.y.numpy()
    tr, va, te = (i.numpy() for i in (g.train_idx, g.val_idx, g.test_idx))
    runs = {f"node+{k}": np.hstack([node, struct[:, cols]]) for k, cols in GROUPS.items()}
    runs["node+degree+types"] = np.hstack([node, struct[:, GROUPS["degree"] + GROUPS["types"]]])
    runs["time only"] = struct[:, GROUPS["time"]]
    with RESULTS.open("a") as f:
        for name, x in runs.items():
            auc = fit(x, y, tr, va, te)
            row = dict(blocks=name, test_auc=auc, test_gini=2 * auc - 1)
            print(json.dumps(row), flush=True)
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
