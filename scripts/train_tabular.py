"""LightGBM / CatBoost / MLP on node features plus graph-derived feature blocks.

Each experiment is a list of feature blocks from graph_features.py, so the
contribution of each graph is read off as the change in test AUC / Gini
against the "self" baseline.

Run: uv run python scripts/train_tabular.py --learner lgbm --blocks self real_struct real_agg
"""

import argparse
import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import paths

# torch (via dgraph_data) and LightGBM each bundle an OpenMP runtime; LightGBM
# must load first or it segfaults, and an MLP run must not load LightGBM at all.
# Hence the imports inside main() and the learners.

RESULTS = paths.OUTPUTS / "results_tabular.jsonl"
PREDS = paths.PREDS


# -------------------------------------------------------------------------------
# Learners
# -------------------------------------------------------------------------------
def fit_lgbm(x: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    import lightgbm as lgb

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
        seed=seed,
        num_threads=10,
    )
    dtr = lgb.Dataset(x[tr], y[tr])
    dva = lgb.Dataset(x[va], y[va], reference=dtr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    p = np.asarray(booster.predict(x, num_iteration=booster.best_iteration), dtype=np.float64)
    return p, int(booster.best_iteration)


def fit_catboost(x: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=100,
        random_seed=seed,
        thread_count=10,
        verbose=False,
    )
    model.fit(x[tr], y[tr], eval_set=(x[va], y[va]), use_best_model=True)
    p = np.asarray(model.predict_proba(x), dtype=np.float64)[:, 1]
    return p, int(model.get_best_iteration())


def fit_mlp(x: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    xt = torch.from_numpy((x - mu) / sd).float()
    yt = torch.from_numpy(y).long()
    model = nn.Sequential(
        nn.Linear(x.shape[1], 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    tr_t = torch.from_numpy(tr)

    @torch.no_grad()
    def predict(idx: np.ndarray) -> np.ndarray:
        model.eval()
        out = [model(xt[idx[i : i + 65536]].to(dev)).softmax(-1)[:, 1].cpu() for i in range(0, len(idx), 65536)]
        return torch.cat(out).numpy()

    best, best_state, best_epoch = 0.0, None, 0
    for epoch in range(1, 61):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for i in range(0, len(perm), 4096):
            b = perm[i : i + 4096]
            loss = F.cross_entropy(model(xt[b].to(dev)), yt[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_auc = roc_auc_score(y[va], predict(va))
        if val_auc > best:
            best, best_epoch = val_auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= 8:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return predict(np.arange(len(x))), best_epoch


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", default="lgbm", choices=["lgbm", "catboost", "mlp"])
    ap.add_argument("--blocks", nargs="+", default=["self"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--save-preds", action="store_true", help="save test scores (.npz) and scores for all nodes (_all.npy)"
    )
    args = ap.parse_args()
    if args.learner == "lgbm":
        import lightgbm  # noqa: F401
    elif args.learner == "catboost":
        import catboost  # noqa: F401

    import dgraph_data
    import graph_features

    g = dgraph_data.load()
    x = graph_features.matrix(args.blocks, g)
    y = g.y.numpy()
    tr, va, te = (idx.numpy() for idx in (g.train_idx, g.val_idx, g.test_idx))

    # fit on labeled rows only, but score every node (background users included)
    t0 = time.time()
    fit = {"lgbm": fit_lgbm, "catboost": fit_catboost, "mlp": fit_mlp}[args.learner]
    p, best_iter = fit(x, y, tr, va, args.seed)
    aucs = {name: roc_auc_score(y[i], p[i]) for name, i in [("train", tr), ("val", va), ("test", te)]}
    row = dict(
        learner=args.learner,
        blocks="+".join(args.blocks),
        n_features=x.shape[1],
        seed=args.seed,
        best_iter=best_iter,
        train_auc=aucs["train"],
        val_auc=aucs["val"],
        test_auc=aucs["test"],
        test_gini=2 * aucs["test"] - 1,
        seconds=round(time.time() - t0, 1),
    )
    print(json.dumps(row))
    if args.save_preds:
        PREDS.mkdir(parents=True, exist_ok=True)
        np.savez(PREDS / f"{args.learner}_{row['blocks']}.npz", node=te, y=y[te], score=p[te])
        np.save(PREDS / f"{args.learner}_{row['blocks']}_all.npy", p.astype(np.float32))
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
