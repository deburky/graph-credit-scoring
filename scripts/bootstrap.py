"""Paired bootstrap of test-AUC differences between saved predictions.

Resamples the test users with replacement (500 draws, seed 0) and reports the mean difference
and a 95% percentile interval for each pair. The interval covers test-set sampling only, not
training randomness. Writes outputs/results_bootstrap.json.

Run: uv run python scripts/bootstrap.py
"""

import json

import numpy as np
from sklearn.metrics import roc_auc_score

import dgraph_data
import paths

MODAL = paths.OUTPUTS / "modal"
PREDS = {
    "relsage": MODAL / "relsage_real" / "preds" / "relsage_real.npy",
    "relsage_knn": MODAL / "relsage_knn2" / "preds" / "relsage_real+knn2.npy",
    "transformer": MODAL / "transformer_real" / "preds" / "transformer_real.npy",
    "transformer_knn": MODAL / "transformer_knn2" / "preds" / "transformer_real+knn2.npy",
    "transformer_h64": MODAL / "transformer_h64" / "preds" / "transformer_real.npy",
    "global": MODAL / "global" / "preds" / "global_none.npy",
    "lgbm_node": paths.PREDS / "lgbm_self_all.npy",
    "lgbm_real": paths.PREDS / "lgbm_self+real_struct+real_agg_all.npy",
    "lgbm_real_knn": paths.PREDS / "lgbm_self+real_struct+real_agg+knn6_agg_all.npy",
    "lgbm_node_rf": paths.PREDS / "lgbm_self+rfprox6_agg_all.npy",
    "lgbm_real_rf": paths.PREDS / "lgbm_self+real_struct+real_agg+rfprox6_agg_all.npy",
}
PAIRS = [
    ("relsage_knn", "relsage"),
    ("transformer_knn", "transformer"),
    ("relsage", "transformer"),
    ("relsage_knn", "transformer_knn"),
    ("relsage", "transformer_h64"),
    ("global", "lgbm_node"),
    ("lgbm_real", "relsage"),
    ("lgbm_real_knn", "lgbm_real"),
    ("lgbm_node_rf", "lgbm_node"),
    ("lgbm_real_rf", "lgbm_real"),
]


def main() -> None:
    g = dgraph_data.load()
    te = g.test_idx.numpy()
    y = g.y.numpy()[te]
    p = {k: np.load(f)[te] for k, f in PREDS.items() if f.exists()}
    rng = np.random.default_rng(0)
    pairs = [(a, b) for a, b in PAIRS if a in p and b in p]
    draws: dict[str, list[float]] = {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(500):
        i = rng.integers(0, len(y), len(y))
        auc = {k: roc_auc_score(y[i], v[i]) for k, v in p.items()}
        for a, b in pairs:
            draws[f"{a}-{b}"].append(auc[a] - auc[b])
    diff: dict[str, dict[str, float]] = {}
    for key, v in draws.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        diff[key] = {"mean": float(np.mean(v)), "lo": float(lo), "hi": float(hi)}
        print(f"{key}: {np.mean(v):+.4f} [{lo:+.4f}, {hi:+.4f}]")
    out = {"auc": {k: float(roc_auc_score(y, v)) for k, v in p.items()}, "diff": diff}
    (paths.OUTPUTS / "results_bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
