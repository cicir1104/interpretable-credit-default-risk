"""Unit tests for metrics and threshold selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.evaluation import (  # noqa: E402
    classification_metrics,
    select_cost_threshold,
)


class EvaluationTests(unittest.TestCase):
    def test_confusion_counts_and_cost(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.4, 0.9])
        metrics = classification_metrics(y_true, probabilities, threshold=0.5)
        self.assertEqual((metrics["tn"], metrics["fp"]), (1, 1))
        self.assertEqual((metrics["fn"], metrics["tp"]), (1, 1))
        self.assertEqual(metrics["illustrative_cost"], 6.0)

    def test_high_false_negative_cost_lowers_threshold(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1])
        probabilities = np.array([0.05, 0.2, 0.4, 0.35, 0.8])
        threshold, _ = select_cost_threshold(
            y_true,
            probabilities,
            false_negative_cost=10.0,
            false_positive_cost=1.0,
        )
        self.assertLessEqual(threshold, 0.35)


if __name__ == "__main__":
    unittest.main()
