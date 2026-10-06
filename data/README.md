# Data provenance and initial rules

## Source

Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C55S3H

Dataset page:
https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Target definition

The original target is `default payment next month`:

- `1`: the client defaults in the following month
- `0`: the client does not default in the following month

The cleaned modelling table will rename this field to `default_next_month`.

## Leakage and fairness rules

- `ID` is an identifier and must not be used as a predictor.
- Features must be limited to information available before the target month.
- `SEX`, `AGE`, `EDUCATION`, and `MARRIAGE` require explicit treatment as
  demographic or potentially sensitive attributes.
- The main model will be compared with and without demographic attributes.
- Demographic fields will be retained for subgroup error and calibration checks.
- Raw files must remain unchanged; cleaning outputs will be written to a separate
  processed-data directory in a later step.

## Original variable groups

- `LIMIT_BAL`: credit limit in New Taiwan dollars
- `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`: demographic variables
- `PAY_0`, `PAY_2` to `PAY_6`: repayment status history
- `BILL_AMT1` to `BILL_AMT6`: monthly bill statement amounts
- `PAY_AMT1` to `PAY_AMT6`: monthly payment amounts
- `default payment next month`: prediction target

