# Field roles and modelling-table rules

## Main model

The main model uses 19 financial and repayment-history
features. It excludes the client identifier and demographic fields.

### Predictors

- `LIMIT_BAL`
- `PAY_0`
- `PAY_2`
- `PAY_3`
- `PAY_4`
- `PAY_5`
- `PAY_6`
- `BILL_AMT1`
- `BILL_AMT2`
- `BILL_AMT3`
- `BILL_AMT4`
- `BILL_AMT5`
- `BILL_AMT6`
- `PAY_AMT1`
- `PAY_AMT2`
- `PAY_AMT3`
- `PAY_AMT4`
- `PAY_AMT5`
- `PAY_AMT6`

### Target

- Raw name: `default payment next month`
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
