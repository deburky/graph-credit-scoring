"""LightGBM / CatBoost / MLP on node features plus graph-derived feature blocks.

Each experiment is a list of feature blocks from graph_features.py, so the
contribution of each graph is read off as the change in test AUC / Gini
against the "self" baseline.

Run: uv run python scripts/train_tabular.py --learner lgbm --blocks self real_struct real_agg
"""

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

import paths

# torch (via dgraph_data) and LightGBM each bundle an OpenMP runtime; LightGBM
# must load first or it segfaults, and an MLP run must not load LightGBM at all.
# Hence the imports inside main() and the learners.

RESULTS = paths.OUTPUTS / "results_tabular.jsonl"
PREDS = paths.PREDS
TUNED = paths.OUTPUTS / "tuning"
THREADS = 6
# progress callback for the tuner: (step, validation AUC so far); None outside tuning
OnStep = Callable[[int, float], None] | None

# starting values; tune_tabular.py searches around them and --tuned loads its best values
DEFAULTS: dict[str, dict[str, float]] = {
    "lgbm": dict(
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        lambda_l2=1.0,
    ),
    "catboost": dict(learning_rate=0.05, depth=8, l2_leaf_reg=3.0, bagging_temperature=1.0),
    "mlp": dict(lr=1e-3, width=256, dropout=0.3, weight_decay=1e-5),
}


# -------------------------------------------------------------------------------
# Learners
# -------------------------------------------------------------------------------
def fit_lgbm(
    x: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    seed: int,
    hp: dict[str, float],
    on_step: OnStep = None,
) -> tuple[np.ndarray, int]:
    import lightgbm as lgb

    def report(env: lgb.callback.CallbackEnv) -> None:
        # one step = 50 boosting rounds
        results = env.evaluation_result_list
        if on_step is not None and results and (env.iteration + 1) % 50 == 0:
            on_step((env.iteration + 1) // 50, float(results[0][2]))

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=hp["learning_rate"],
        num_leaves=int(hp["num_leaves"]),
        min_data_in_leaf=int(hp["min_data_in_leaf"]),
        feature_fraction=hp["feature_fraction"],
        bagging_fraction=hp["bagging_fraction"],
        bagging_freq=1,
        lambda_l2=hp["lambda_l2"],
        verbose=-1,
        seed=seed,
        num_threads=THREADS,
    )
    dtr = lgb.Dataset(x[tr], y[tr])
    dva = lgb.Dataset(x[va], y[va], reference=dtr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False), report],
    )
    p = np.asarray(booster.predict(x, num_iteration=booster.best_iteration), dtype=np.float64)
    return p, int(booster.best_iteration)


def fit_catboost(
    x: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    seed: int,
    hp: dict[str, float],
    on_step: OnStep = None,
) -> tuple[np.ndarray, int]:
    from catboost import CatBoostClassifier

    class Report:
        # one step = 100 iterations; returning True lets training continue
        def after_iteration(self, info: Any) -> bool:
            it = info.iteration + 1
            if on_step is not None and it % 100 == 0:
                on_step(it // 100, float(info.metrics["validation"]["AUC"][-1]))
            return True

    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=hp["learning_rate"],
        depth=int(hp["depth"]),
        l2_leaf_reg=hp["l2_leaf_reg"],
        bagging_temperature=hp["bagging_temperature"],
        eval_metric="AUC",
        od_type="Iter",
        od_wait=100,
        random_seed=seed,
        thread_count=THREADS,
        verbose=False,
    )
    model.fit(x[tr], y[tr], eval_set=(x[va], y[va]), use_best_model=True, callbacks=[Report()])
    p = np.asarray(model.predict_proba(x), dtype=np.float64)[:, 1]
    return p, int(model.get_best_iteration())


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    seed: int,
    hp: dict[str, float],
    on_step: OnStep = None,
) -> tuple[np.ndarray, int]:
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    xt = torch.from_numpy((x - mu) / sd).float()
    yt = torch.from_numpy(y).long()
    w, drop = int(hp["width"]), hp["dropout"]
    model = nn.Sequential(
        nn.Linear(x.shape[1], w),
        nn.BatchNorm1d(w),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Linear(w, w // 2),
        nn.BatchNorm1d(w // 2),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Linear(w // 2, 2),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
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
        if on_step is not None:
            on_step(epoch, float(val_auc))
        if val_auc > best:
            best, best_epoch = val_auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= 8:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return predict(np.arange(len(x))), best_epoch


FITS = {"lgbm": fit_lgbm, "catboost": fit_catboost, "mlp": fit_mlp}


def tuned_params(learner: str, blocks: str) -> dict[str, float]:
    """Best values found by tune_tabular.py for this learner and feature set."""
    return json.loads((TUNED / f"{learner}_{blocks}.json").read_text())["best_params"]


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", default="lgbm", choices=["lgbm", "catboost", "mlp"])
    ap.add_argument("--blocks", nargs="+", default=["self"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--save-preds",
        action="store_true",
        help="save test scores (.npz) and scores for all nodes (_all.npy)",
    )
    ap.add_argument("--tuned", action="store_true", help="use the values found by tune_tabular.py")
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
    blocks = "+".join(args.blocks)
    hp = tuned_params(args.learner, blocks) if args.tuned else DEFAULTS[args.learner]
    t0 = time.time()
    p, best_iter = FITS[args.learner](x, y, tr, va, args.seed, hp)
    aucs = {name: roc_auc_score(y[i], p[i]) for name, i in [("train", tr), ("val", va), ("test", te)]}
    row = dict(
        learner=args.learner,
        blocks=blocks,
        n_features=x.shape[1],
        tuned=args.tuned,
        params=hp,
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
