"""Create reproducible modelling and audit tables from the raw workbook."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.data import (  # noqa: E402
    DEMOGRAPHIC_COLUMNS,
    FINANCIAL_FEATURES,
    IDENTIFIER,
    TARGET,
    build_project_tables,
    load_raw_data,
)


RAW_FILE = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "field_roles.md"
ROLE_PATH = PROJECT_ROOT / "reports" / "field_roles.json"


def main() -> None:
    raw = load_raw_data(RAW_FILE)
    tables = build_project_tables(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        table.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    roles = {
        "identifier": [IDENTIFIER],
        "target": [TARGET],
        "main_model_predictors": FINANCIAL_FEATURES,
        "demographic_audit_fields": DEMOGRAPHIC_COLUMNS,
        "main_model_predictor_count": len(FINANCIAL_FEATURES),
        "policy": {
            "identifier": "retained only for row alignment",
            "demographics": "excluded from the main model and retained for subgroup audit",
            "education_codes_0_5_6": "other_or_unknown",
            "marriage_code_0": "other_or_unknown",
            "negative_bill_amounts": "preserved as observed values",
        },
    }
    ROLE_PATH.write_text(json.dumps(roles, indent=2) + "\n")

    REPORT_PATH.write_text(
        f"""# Field roles and modelling-table rules

## Main model

The main model uses {len(FINANCIAL_FEATURES)} financial and repayment-history
features. It excludes the client identifier and demographic fields.

### Predictors

{chr(10).join(f'- `{column}`' for column in FINANCIAL_FEATURES)}

### Target

- Raw name: `{TARGET}`
- Modelling name: `default_next_month`
- `1` means default in the following month; `0` means no default.

## Fields excluded from the main model

- `ID` becomes `customer_id` and is retained only for row alignment.
- `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` are retained in a separate audit
  table for subgroup performance checks.

## Transparent recoding for subgroup audit

- `EDUCATION` codes 0, 5 and 6 become `other_or_unknown`.
- `MARRIAGE` code 0 becomes `other_or_unknown`.
- Original numeric demographic codes remain alongside the readable groups.
- Age is grouped as 21-29, 30-39, 40-49, 50-59 and 60+.

## Preserved values

Negative bill amounts are not replaced or deleted. They may represent credits,
refunds or balance adjustments and are valid observed values unless stronger
evidence shows otherwise.

## Output tables

- `data/processed/model_table.csv`: ID, 19 predictors and target
- `data/processed/predictors.csv`: 19 predictors
- `data/processed/target.csv`: renamed binary target
- `data/processed/identifiers.csv`: customer ID only
- `data/processed/demographic_audit.csv`: original demographic codes and groups

Processed CSV files are reproducible local outputs and are excluded from Git.
"""
    )

    for name, table in tables.items():
        print(f"{name}: {table.shape}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {ROLE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
