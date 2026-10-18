"""Score the saved test predictions on discrimination and calibration.

Reads every all-node prediction file in outputs/preds/ (written by --save-preds and by
train_gnn.py / train_embed.py), restricts it to the test users and writes AUC, Gini, Brier score,
Brier Index and adjusted Brier Index to outputs/results_calibration.jsonl, with the constant
forecast at the training fraud rate as the first row.

Run: uv run python scripts/calibration.py
"""

import json

import numpy as np

import dgraph_data
import metrics
import paths


def main() -> None:
    g = dgraph_data.load()
    te = g.test_idx.numpy()
    y = g.y.numpy()[te]
    train_rate = float(g.y[g.train_idx].float().mean())
    files = sorted(f for f in paths.PREDS.glob("*.npy") if not f.stem.endswith("_attention"))
    # the constant forecast at the training fraud rate is the reference the adjusted index scores 50
    preds = {"constant": np.full(len(y), train_rate)}
    preds |= {f.stem.removesuffix("_all"): np.load(f).astype(np.float64)[te] for f in files}
    with open(paths.OUTPUTS / "results_calibration.jsonl", "w") as out:
        for name, p in preds.items():
            row = dict(name=name, **metrics.summary(y, p))
            row["adjusted_brier_index_train_rate"] = metrics.adjusted_brier_index(y, p, train_rate)
            out.write(json.dumps(row) + "\n")
            print(
                f"{name} auc={row['auc']:.4f} brier={row['brier']:.5f} "
                f"BI={row['brier_index']:.1f} ABI={row['adjusted_brier_index']:.1f} mean_p={row['mean_p']:.4f}"
            )


if __name__ == "__main__":
    main()
