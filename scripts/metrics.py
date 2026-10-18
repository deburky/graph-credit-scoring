"""Discrimination and calibration metrics for binary scores.

The Brier Index of the Forecasting Research Institute (Kucinskas, Bastani and Reynolds, 2026)
rescales the Brier score to a percentage:

    Brier Index = (1 - sqrt(Brier)) x 100%

with 100% for perfect foresight, 50% for always predicting 0.5 and 0% for maximally wrong
forecasts. Its 50% reference assumes a balanced outcome. For a rare event such as credit fraud,
always predicting 0.5 is a poor reference; the constant forecast at the base rate p already
has Brier p(1 - p). The adjusted index measures against that forecast instead:

    Adjusted Brier Index = 100% - 50% x sqrt(Brier / (p (1 - p)))

which keeps 100% for perfect foresight and gives 50% for always predicting the base rate.
Equivalently, it is 100% - 50% x sqrt(1 - BSS), with BSS the Brier skill score.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - y) ** 2))


def brier_index(y: np.ndarray, p: np.ndarray) -> float:
    """FRI Brier Index in percent: 50 for always predicting 0.5."""
    return 100.0 * (1.0 - np.sqrt(brier(y, p)))


def adjusted_brier_index(y: np.ndarray, p: np.ndarray, base_rate: float | None = None) -> float:
    """Brier Index against the base-rate forecast, in percent: 50 for always predicting the base rate.

    base_rate defaults to the event rate of y, the reference a constant forecast would need to
    know in advance; pass the training rate to measure against a forecast that is available.
    """
    rate = float(np.mean(y)) if base_rate is None else base_rate
    return 100.0 - 50.0 * np.sqrt(brier(y, p) / (rate * (1.0 - rate)))


def brier_skill(y: np.ndarray, p: np.ndarray, base_rate: float | None = None) -> float:
    rate = float(np.mean(y)) if base_rate is None else base_rate
    return 1.0 - brier(y, p) / (rate * (1.0 - rate))


def summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(y, p))
    return dict(
        auc=auc,
        gini=2 * auc - 1,
        brier=brier(y, p),
        brier_index=brier_index(y, p),
        adjusted_brier_index=adjusted_brier_index(y, p),
        brier_skill=brier_skill(y, p),
        mean_p=float(np.mean(p)),
        base_rate=float(np.mean(y)),
    )
