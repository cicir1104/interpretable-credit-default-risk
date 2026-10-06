"""Metrics and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> dict[str, float | int]:
    """Return probability and threshold metrics for a binary classifier."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    total_cost = false_negative_cost * fn + false_positive_cost * fp
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "illustrative_cost": float(total_cost),
        "cost_per_1000": float(total_cost / len(y_true) * 1000),
    }


def threshold_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> list[dict[str, float | int]]:
    """Evaluate a fixed threshold grid without looking at the test set."""
    return [
        classification_metrics(
            y_true,
            probabilities,
            threshold=float(threshold),
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2)
    ]


def select_cost_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> tuple[float, list[dict[str, float | int]]]:
    """Select the lowest-cost threshold on validation data only."""
    rows = threshold_table(
        y_true,
        probabilities,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    best = min(rows, key=lambda row: (row["illustrative_cost"], -row["recall"]))
    return float(best["threshold"]), rows
