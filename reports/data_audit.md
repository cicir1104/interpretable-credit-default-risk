# Raw data audit

## Source integrity

- File: `data/raw/default of credit card clients.xls`
- SHA-256: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`
- Worksheet: `Data`
- The workbook is read with its second row as the column header.
- The raw workbook was not modified by this audit.

## Structure and completeness

- Rows: 30,000
- Columns: 25
- Missing cells: 0
- Duplicate column names: 0
- All columns were read as integers: True
- Unique IDs: 30,000
- Duplicate IDs: 0
- ID range: 1 to 30000

## Target balance

- Non-default (`0`): 23,364 (77.88%)
- Default (`1`): 6,636 (22.12%)

The target is imbalanced enough that accuracy alone would be misleading. Later
evaluation must include recall, precision, precision-recall AUC and probability
calibration.

## Duplicate checks

- Fully duplicated rows including `ID`: 0
- Rows repeating all predictors after excluding `ID` and the target: 56
- Rows repeating all predictors and the target after excluding `ID`: 35
- Predictor combinations that occur more than once: 52

Repeated predictor combinations are not automatically data errors because each
row has a unique client ID. They will not be removed without additional
evidence.

## Category-code checks

- `SEX`: {'1': 11888, '2': 18112}
- `EDUCATION`: {'0': 14, '1': 10585, '2': 14030, '3': 4917, '4': 123, '5': 280, '6': 51}
- `MARRIAGE`: {'0': 54, '1': 13659, '2': 15964, '3': 323}

`EDUCATION` includes codes 0, 5 and 6, and `MARRIAGE` includes code 0. These
codes are not clearly defined in the short data dictionary and must be handled
transparently rather than silently discarded.

## Numeric ranges

- Age: 21 to 79
- Credit limit: 10,000 to 1,000,000
- Repayment-status codes: -2 to 8
- Bill amounts: -339,603 to 1,664,089
- Negative bill-amount cells: 3,932
- Payment amounts: 0 to 1,684,259
- Negative payment-amount cells: 0

Negative bill amounts may reflect credits, refunds or balance adjustments. They
are observed values, not missing values, and will not be changed during the raw
audit.

## Audit conclusion

The file matches the documented 30,000-row, 25-column structure and has no
missing cells or duplicate IDs. The next step is to define field roles and
transparent recoding rules before any train/test split or model fitting.
