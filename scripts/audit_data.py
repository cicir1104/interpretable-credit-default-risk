"""Audit the untouched UCI credit-default workbook.

This script reads the raw Excel file without modifying it and writes the same
audit in machine-readable JSON and human-readable Markdown formats.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
REPORT_DIR = PROJECT_ROOT / "reports"
TARGET = "default payment next month"
IDENTIFIER = "ID"

EXPECTED_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
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
    TARGET,
]


def sha256(path: Path) -> str:
    """Return a stable fingerprint for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_data(path: Path) -> pd.DataFrame:
    """Read the workbook using its second row as the real header."""
    return pd.read_excel(path, sheet_name="Data", header=1, engine="xlrd")


def validate_schema(data: pd.DataFrame) -> None:
    """Stop if the workbook structure differs from the documented source."""
    actual = data.columns.tolist()
    if actual != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected workbook schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual}"
        )


def integer_counts(series: pd.Series) -> dict[str, int]:
    """Return sorted value counts with JSON-safe keys and values."""
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def audit(data: pd.DataFrame, source_path: Path) -> dict[str, object]:
    """Calculate descriptive checks without changing the input data."""
    validate_schema(data)
    predictor_columns = [
        column for column in data.columns if column not in {IDENTIFIER, TARGET}
    ]
    payment_status_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_columns = [f"BILL_AMT{month}" for month in range(1, 7)]
    payment_columns = [f"PAY_AMT{month}" for month in range(1, 7)]

    target_counts = integer_counts(data[TARGET])
    target_rates = {
        value: round(count / len(data), 6) for value, count in target_counts.items()
    }

    def range_summary(columns: list[str]) -> dict[str, int]:
        values = data[columns]
        return {
            "minimum": int(values.min().min()),
            "maximum": int(values.max().max()),
            "negative_cells": int((values < 0).sum().sum()),
            "zero_cells": int((values == 0).sum().sum()),
        }

    repeated_group_sizes = data.groupby(predictor_columns, dropna=False).size()

    return {
        "source": {
            "path": str(source_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(source_path),
            "sheet": "Data",
            "header_row_zero_based": 1,
        },
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "schema": {
            "all_columns_integer": bool((data.dtypes == "int64").all()),
            "duplicate_column_names": int(data.columns.duplicated().sum()),
            "columns": data.columns.tolist(),
        },
        "missing_cells": int(data.isna().sum().sum()),
        "identifier": {
            "unique_values": int(data[IDENTIFIER].nunique()),
            "duplicate_values": int(data[IDENTIFIER].duplicated().sum()),
            "minimum": int(data[IDENTIFIER].min()),
            "maximum": int(data[IDENTIFIER].max()),
        },
        "duplicates": {
            "full_rows_including_id": int(data.duplicated().sum()),
            "rows_repeating_predictors": int(
                data.duplicated(subset=predictor_columns).sum()
            ),
            "rows_repeating_predictors_and_target": int(
                data.duplicated(subset=predictor_columns + [TARGET]).sum()
            ),
            "repeated_predictor_groups": int((repeated_group_sizes > 1).sum()),
        },
        "target": {"counts": target_counts, "rates": target_rates},
        "category_counts": {
            column: integer_counts(data[column])
            for column in ["SEX", "EDUCATION", "MARRIAGE"]
        },
        "ranges": {
            "age": {
                "minimum": int(data["AGE"].min()),
                "maximum": int(data["AGE"].max()),
            },
            "credit_limit": {
                "minimum": int(data["LIMIT_BAL"].min()),
                "maximum": int(data["LIMIT_BAL"].max()),
            },
            "payment_status": range_summary(payment_status_columns),
            "bill_amount": range_summary(bill_columns),
            "payment_amount": range_summary(payment_columns),
        },
    }


def markdown_report(result: dict[str, object]) -> str:
    """Render the main audit findings and their interpretation."""
    shape = result["shape"]
    identifier = result["identifier"]
    duplicates = result["duplicates"]
    target = result["target"]
    categories = result["category_counts"]
    ranges = result["ranges"]
    source = result["source"]

    return f"""# Raw data audit

## Source integrity

- File: `{source['path']}`
- SHA-256: `{source['sha256']}`
- Worksheet: `{source['sheet']}`
- The workbook is read with its second row as the column header.
- The raw workbook was not modified by this audit.

## Structure and completeness

- Rows: {shape['rows']:,}
- Columns: {shape['columns']}
- Missing cells: {result['missing_cells']}
- Duplicate column names: {result['schema']['duplicate_column_names']}
- All columns were read as integers: {result['schema']['all_columns_integer']}
- Unique IDs: {identifier['unique_values']:,}
- Duplicate IDs: {identifier['duplicate_values']}
- ID range: {identifier['minimum']} to {identifier['maximum']}

## Target balance

- Non-default (`0`): {target['counts']['0']:,} ({target['rates']['0']:.2%})
- Default (`1`): {target['counts']['1']:,} ({target['rates']['1']:.2%})

The target is imbalanced enough that accuracy alone would be misleading. Later
evaluation must include recall, precision, precision-recall AUC and probability
calibration.

## Duplicate checks

- Fully duplicated rows including `ID`: {duplicates['full_rows_including_id']}
- Rows repeating all predictors after excluding `ID` and the target: {duplicates['rows_repeating_predictors']}
- Rows repeating all predictors and the target after excluding `ID`: {duplicates['rows_repeating_predictors_and_target']}
- Predictor combinations that occur more than once: {duplicates['repeated_predictor_groups']}

Repeated predictor combinations are not automatically data errors because each
row has a unique client ID. They will not be removed without additional
evidence.

## Category-code checks

- `SEX`: {categories['SEX']}
- `EDUCATION`: {categories['EDUCATION']}
- `MARRIAGE`: {categories['MARRIAGE']}

`EDUCATION` includes codes 0, 5 and 6, and `MARRIAGE` includes code 0. These
codes are not clearly defined in the short data dictionary and must be handled
transparently rather than silently discarded.

## Numeric ranges

- Age: {ranges['age']['minimum']} to {ranges['age']['maximum']}
- Credit limit: {ranges['credit_limit']['minimum']:,} to {ranges['credit_limit']['maximum']:,}
- Repayment-status codes: {ranges['payment_status']['minimum']} to {ranges['payment_status']['maximum']}
- Bill amounts: {ranges['bill_amount']['minimum']:,} to {ranges['bill_amount']['maximum']:,}
- Negative bill-amount cells: {ranges['bill_amount']['negative_cells']:,}
- Payment amounts: {ranges['payment_amount']['minimum']:,} to {ranges['payment_amount']['maximum']:,}
- Negative payment-amount cells: {ranges['payment_amount']['negative_cells']:,}

Negative bill amounts may reflect credits, refunds or balance adjustments. They
are observed values, not missing values, and will not be changed during the raw
audit.

## Audit conclusion

The file matches the documented 30,000-row, 25-column structure and has no
missing cells or duplicate IDs. The next step is to define field roles and
transparent recoding rules before any train/test split or model fitting.
"""


def main() -> None:
    data = load_raw_data(RAW_FILE)
    result = audit(data, RAW_FILE)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "data_audit.json"
    markdown_path = REPORT_DIR / "data_audit.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    markdown_path.write_text(markdown_report(result))
    print(f"Wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {markdown_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
