"""One Syne Tune trial: fit a tabular learner with the given hyperparameters and report
validation AUC as it trains (see tune_tabular.py, which launches this script).

Syne Tune passes every entry of the config space as a command-line argument.
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

import train_tabular
import tune_tabular


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_t", type=int, required=True)
    args, _ = ap.parse_known_args()
    # the learner's hyperparameters; Syne Tune also passes arguments of its own, which are ignored
    names = list(tune_tabular.config_space(args.learner))
    hp_ap = argparse.ArgumentParser()
    for name in names:
        hp_ap.add_argument(f"--{name}", type=float, required=True)
    hp = vars(hp_ap.parse_known_args()[0])
    # OpenMP: the learner loads before torch (dgraph_data), as in train_tabular.py
    if args.learner == "lgbm":
        import lightgbm  # noqa: F401
    elif args.learner == "catboost":
        import catboost  # noqa: F401
    from syne_tune import Reporter

    import dgraph_data
    import graph_features

    g = dgraph_data.load()
    x = graph_features.matrix(args.blocks.split("+"), g)
    y = g.y.numpy()
    tr, va = g.train_idx.numpy(), g.val_idx.numpy()
    report = Reporter()
    last = 0

    def on_step(step: int, val_auc: float) -> None:
        nonlocal last
        last = step
        report(step=step, val_auc=val_auc, final=0, steps_run=step)

    p, _ = train_tabular.FITS[args.learner](x, y, tr, va, args.seed, hp, on_step)
    # the early-stopped model counts as having used the full budget, so it reaches the top rung
    final_auc = float(roc_auc_score(y[va], np.asarray(p)[va]))
    report(step=args.max_t, val_auc=final_auc, final=1, steps_run=last)


if __name__ == "__main__":
    main()
