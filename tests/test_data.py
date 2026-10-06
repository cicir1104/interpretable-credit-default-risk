"""Unit tests for field separation and transparent recoding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.data import (  # noqa: E402
    DEMOGRAPHIC_COLUMNS,
    FINANCIAL_FEATURES,
    IDENTIFIER,
    TARGET,
    build_project_tables,
)


class ProjectTableTests(unittest.TestCase):
    def setUp(self) -> None:
        row = {column: 0 for column in FINANCIAL_FEATURES}
        row.update(
            {
                IDENTIFIER: 7,
                TARGET: 1,
                "SEX": 2,
                "EDUCATION": 6,
                "MARRIAGE": 0,
                "AGE": 35,
            }
        )
        self.raw = pd.DataFrame([row])

    def test_main_model_contains_only_financial_features(self) -> None:
        tables = build_project_tables(self.raw)
        self.assertEqual(tables["predictors"].columns.tolist(), FINANCIAL_FEATURES)
        self.assertNotIn(IDENTIFIER, tables["predictors"].columns)
        for column in DEMOGRAPHIC_COLUMNS:
            self.assertNotIn(column, tables["predictors"].columns)

    def test_undocumented_codes_are_grouped_transparently(self) -> None:
        audit = build_project_tables(self.raw)["demographic_audit"].iloc[0]
        self.assertEqual(audit["education_group"], "other_or_unknown")
        self.assertEqual(audit["marriage_group"], "other_or_unknown")
        self.assertEqual(audit["age_group"], "30-39")
        self.assertEqual(audit["EDUCATION"], 6)
        self.assertEqual(audit["MARRIAGE"], 0)


if __name__ == "__main__":
    unittest.main()
