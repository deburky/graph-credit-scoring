"""Tune a tabular learner on one feature set with Syne Tune (BOHB), by validation AUC.

BOHB (Falkner et al., 2018) pairs a TPE-style Bayesian searcher with asynchronous successive
halving: each trial (tune_trial.py) reports validation AUC as it trains, every 50 LightGBM rounds,
100 CatBoost iterations or one MLP epoch, and trials that trail at a rung are stopped early. Each
trial fits on the training users with early stopping on the validation users; the test users are
never touched. The first trial is the hand-set starting point. The best values go to
outputs/tuning/{learner}_{blocks}.json, which train_tabular.py --tuned reads.

Run: uv run python scripts/tune_tabular.py --learner lgbm --blocks self real_struct real_agg --trials 25
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import paths
import train_tabular

HERE = Path(__file__).resolve().parent
# steps per trial: 3000 rounds / 50 for LightGBM, 3000 iterations / 100 for CatBoost, 60 MLP epochs
MAX_STEPS = {"lgbm": 60, "catboost": 30, "mlp": 60}


def config_space(learner: str) -> dict:
    from syne_tune.config_space import choice, lograndint, loguniform, randint, uniform

    if learner == "lgbm":
        return {
            "learning_rate": loguniform(0.01, 0.2),
            "num_leaves": lograndint(15, 255),
            "min_data_in_leaf": lograndint(20, 2000),
            "feature_fraction": uniform(0.3, 1.0),
            "bagging_fraction": uniform(0.5, 1.0),
            "lambda_l2": loguniform(1e-3, 100.0),
        }
    if learner == "catboost":
        return {
            "learning_rate": loguniform(0.02, 0.2),
            "depth": randint(4, 10),
            "l2_leaf_reg": loguniform(1.0, 30.0),
            "bagging_temperature": uniform(0.0, 1.0),
        }
    return {
        "lr": loguniform(1e-4, 3e-3),
        "width": choice([128, 256, 512]),
        "dropout": uniform(0.0, 0.5),
        "weight_decay": loguniform(1e-6, 1e-3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", default="lgbm", choices=["lgbm", "catboost", "mlp"])
    ap.add_argument("--blocks", nargs="+", default=["self"])
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    blocks = "+".join(args.blocks)
    folder = paths.OUTPUTS / "syne-tune"
    os.environ["SYNETUNE_FOLDER"] = str(folder)

    from syne_tune import StoppingCriterion, Tuner
    from syne_tune.backend import LocalBackend
    from syne_tune.experiments import load_experiment
    from syne_tune.optimizer.baselines import BOHB

    # the final report of a trial goes one step past its progress reports
    learner, max_t = args.learner, MAX_STEPS[args.learner] + 1
    # constants are passed to every trial as command-line arguments
    space = {**config_space(learner), "learner": learner, "blocks": blocks, "seed": args.seed, "max_t": max_t}
    start = {k: v for k, v in train_tabular.DEFAULTS[learner].items()}
    scheduler = BOHB(
        space,
        metric="val_auc",
        do_minimize=False,
        time_attr="step",
        max_t=max_t,
        grace_period=2,
        reduction_factor=3,
        random_seed=args.seed,
        points_to_evaluate=[start],
    )
    name = f"{learner}-{blocks.replace('+', '-').replace('_', '')}"
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=str(HERE / "tune_trial.py"), binary=sys.executable),
        scheduler=scheduler,
        # the criterion is checked after the next trial has started, hence the - 1
        stop_criterion=StoppingCriterion(max_num_trials_started=args.trials - 1),
        n_workers=1,
        tuner_name=name,
        wait_trial_completion_when_stopping=True,
        print_update_interval=300,
    )
    t0 = time.time()
    tuner.run()

    # a trial's last report (final=1) is the AUC of its early-stopped model; trials stopped by
    # successive halving have no final report and cannot be the best
    df = load_experiment(tuner.name, local_path=str(folder)).results
    final = df[df["final"] == 1].sort_values("trial_id")
    best = final.loc[final["val_auc"].idxmax()]
    keys = list(config_space(learner))
    first = final[final["trial_id"] == 0]
    out = {
        "learner": learner,
        "blocks": blocks,
        "tuner": "syne-tune BOHB",
        "trials": int(df["trial_id"].nunique()),
        "trials_completed": int(len(final)),
        "best_val_auc": float(best["val_auc"]),
        "default_val_auc": float(first["val_auc"].iloc[0]) if len(first) else None,
        "best_params": {k: best[f"config_{k}"].item() for k in keys},
        "history": [
            {"trial_id": int(r["trial_id"]), "val_auc": float(r["val_auc"]), "steps": int(r["steps_run"])}
            for _, r in final.iterrows()
        ],
        "syne_tune_name": tuner.name,
        "seconds": round(time.time() - t0, 1),
    }
    train_tabular.TUNED.mkdir(parents=True, exist_ok=True)
    (train_tabular.TUNED / f"{learner}_{blocks}.json").write_text(json.dumps(out, indent=1))
    print(
        f"{learner} {blocks}: start val AUC {out['default_val_auc']}, best {out['best_val_auc']:.4f} "
        f"{out['best_params']} ({out['trials_completed']}/{out['trials']} trials ran to the end, "
        f"{out['seconds']:.0f}s)"
    )


if __name__ == "__main__":
    main()
