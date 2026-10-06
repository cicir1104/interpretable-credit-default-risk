"""Data loading, field roles and transparent demographic recoding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


IDENTIFIER = "ID"
TARGET = "default payment next month"

FINANCIAL_FEATURES = [
    "LIMIT_BAL",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

DEMOGRAPHIC_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]


def load_raw_data(path: Path) -> pd.DataFrame:
    """Read the untouched workbook using its second row as the header."""
    return pd.read_excel(path, sheet_name="Data", header=1, engine="xlrd")


def recode_demographics(data: pd.DataFrame) -> pd.DataFrame:
    """Create readable audit groups while preserving each original code."""
    audit = data[[IDENTIFIER, *DEMOGRAPHIC_COLUMNS]].copy()
    audit = audit.rename(columns={IDENTIFIER: "customer_id"})

    audit["sex_group"] = data["SEX"].map({1: "male", 2: "female"}).fillna(
        "other_or_unknown"
    )
    audit["education_group"] = data["EDUCATION"].map(
        {
            1: "graduate_school",
            2: "university",
            3: "high_school",
            4: "other",
            0: "other_or_unknown",
            5: "other_or_unknown",
            6: "other_or_unknown",
        }
    ).fillna("other_or_unknown")
    audit["marriage_group"] = data["MARRIAGE"].map(
        {1: "married", 2: "single", 3: "other", 0: "other_or_unknown"}
    ).fillna("other_or_unknown")
    audit["age_group"] = pd.cut(
        data["AGE"],
        bins=[20, 29, 39, 49, 59, float("inf")],
        labels=["21-29", "30-39", "40-49", "50-59", "60+"],
        right=True,
    ).astype("string")
    return audit


def build_project_tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Separate modelling fields from identifiers and demographic audit data."""
    required = {IDENTIFIER, TARGET, *FINANCIAL_FEATURES, *DEMOGRAPHIC_COLUMNS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    predictors = data[FINANCIAL_FEATURES].copy()
    target = data[[TARGET]].rename(columns={TARGET: "default_next_month"}).copy()
    identifiers = data[[IDENTIFIER]].rename(columns={IDENTIFIER: "customer_id"}).copy()
    demographic_audit = recode_demographics(data)
    model_table = pd.concat([identifiers, predictors, target], axis=1)

    return {
        "model_table": model_table,
        "predictors": predictors,
        "target": target,
        "identifiers": identifiers,
        "demographic_audit": demographic_audit,
    }
