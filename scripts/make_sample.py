"""Write a small synthetic dataset in the DGraph-Fin format (data/sample/raw/dgraph.npz).

DGraph-Fin itself is distributed under a user license that requires registration,
so this repository ships synthetic data instead. Every array has the same key,
dtype and coding as the real file, so all scripts run on it unchanged:

    x               (n, 17) float64   node features, -1 = missing
    y               (n,)    int64     0 normal, 1 credit fraud, 2/3 background users
    edge_index      (m, 2)  int64     (source, target): source named target as an emergency contact
    edge_type       (m,)    int64     contact type, 1-11
    edge_timestamp  (m,)    int64     desensitized time, 1-821
    train_mask, valid_mask, test_mask  index arrays, a 70/15/15 split of labeled users

The structure is planted, loosely following what the DGraph authors report:
fraudsters name fewer contacts (mean out-degree 0.75 against 1.73 for normal
users), tend to name other fraudsters, and add contacts later; features carry
a weak signal. Fraud is made more common than in the real data (5% of borrowers
against about 1.3%) so that a small sample has enough positives to score.

Run: uv run python scripts/make_sample.py
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_FEATURES = 17
N_TYPES = 11
T_MAX = 821


def make(n: int, fraud_rate: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    # users: about a third are borrowers with a label, the rest background users
    borrower = rng.random(n) < 0.33
    fraud = borrower & (rng.random(n) < fraud_rate)
    y = np.where(borrower, fraud.astype(np.int64), rng.choice([2, 3], size=n, p=[0.65, 0.35]))

    # features: five always-observed columns, twelve that are often missing
    x = np.empty((n, N_FEATURES))
    shift = np.where(fraud, 0.5, 0.0)
    x[:, :5] = np.abs(rng.normal(1.0 + shift[:, None], 1.0, size=(n, 5)))
    x[:, 2] = rng.integers(0, 9, size=n)
    x[:, 5:] = np.clip(rng.normal(0.4 - 0.3 * shift[:, None], 0.25, size=(n, N_FEATURES - 5)), 0, 1)
    p_missing = np.where(borrower, 0.45, 0.65)
    x[:, 5:][rng.random((n, N_FEATURES - 5)) < p_missing[:, None]] = -1.0

    # edges: who names whom as an emergency contact
    out_degree = rng.poisson(np.where(fraud, 0.75, np.where(borrower, 1.73, 0.9)))
    fraud_ids = np.flatnonzero(fraud)
    src_list: list[int] = []
    dst_list: list[int] = []
    type_list: list[int] = []
    time_list: list[int] = []
    for i in np.flatnonzero(out_degree):
        for _ in range(out_degree[i]):
            if fraud[i] and len(fraud_ids) > 1 and rng.random() < 0.25:
                j = int(rng.choice(fraud_ids))
            else:
                j = int(rng.integers(n))
            if j == i:
                continue
            src_list.append(int(i))
            dst_list.append(j)
            type_list.append(int(rng.choice(N_TYPES, p=_type_probs(bool(fraud[i])))) + 1)
            late = 300 if fraud[i] else 0
            time_list.append(int(np.clip(rng.integers(1, T_MAX + 1) + late, 1, T_MAX)))

    # every user appears in at least one edge, as in DGraph-Fin
    touched = np.zeros(n, dtype=bool)
    touched[src_list] = True
    touched[dst_list] = True
    for i in np.flatnonzero(~touched):
        j = int(rng.integers(n))
        if j == i:
            j = (j + 1) % n
        src_list.append(j)
        dst_list.append(int(i))
        type_list.append(1)
        time_list.append(int(rng.integers(1, T_MAX + 1)))

    labeled = rng.permutation(np.flatnonzero(borrower))
    n_train, n_valid = int(0.70 * len(labeled)), int(0.15 * len(labeled))
    return dict(
        x=x,
        y=y,
        edge_index=np.stack([src_list, dst_list], 1).astype(np.int64),
        edge_type=np.array(type_list, dtype=np.int64),
        edge_timestamp=np.array(time_list, dtype=np.int64),
        train_mask=np.sort(labeled[:n_train]),
        valid_mask=np.sort(labeled[n_train : n_train + n_valid]),
        test_mask=np.sort(labeled[n_train + n_valid :]),
    )


def _type_probs(is_fraud: bool) -> np.ndarray:
    """Contact-type mix; fraudsters favor a few types more than normal users do."""
    w = np.ones(N_TYPES)
    w[0] = 4.0
    if is_fraud:
        w[[3, 7, 10]] += 2.0
    return w / w.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="number of users")
    ap.add_argument("--fraud-rate", type=float, default=0.05, help="share of borrowers with credit fraud")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sample" / "raw" / "dgraph.npz")
    args = ap.parse_args()
    d = make(args.n, args.fraud_rate, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=d["x"],
        y=d["y"],
        edge_index=d["edge_index"],
        edge_type=d["edge_type"],
        edge_timestamp=d["edge_timestamp"],
        train_mask=d["train_mask"],
        valid_mask=d["valid_mask"],
        test_mask=d["test_mask"],
    )
    y = d["y"]
    print(
        f"wrote {args.out}: users={len(y):,} edges={len(d['edge_index']):,} "
        f"borrowers={(y <= 1).sum():,} fraud={(y == 1).sum():,} background={(y >= 2).sum():,}"
    )


if __name__ == "__main__":
    main()
