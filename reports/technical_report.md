# Technical report: interpretable credit-default risk modelling

## Objective

Estimate next-month default probability from information available at the
prediction date, compare interpretable and non-linear models, and study how an
illustrative asymmetric error cost changes the classification threshold.

## Data and scope

- 30,000 Taiwan credit-card clients from 2005
- 19 financial and repayment-history predictors in the main model
- Demographic fields excluded from the main model and retained for audit
- Historical observational data; results are predictive associations, not
  causal effects or a present-day lending policy

## Experimental design

- Training: 18,000 rows
- Validation: 6,000 rows
- Test: 6,000 rows
- Random seed: 42
- Hyperparameters and thresholds selected without using the test set
- Scenario cost: missed default = 5, unnecessary high-risk
  flag = 1

## Models

Dummy prior, scaled logistic regression, decision tree, random forest and
histogram gradient boosting were compared. The validation scenario selected
**random_forest** with a threshold of **0.14**.

## Final test result

- ROC AUC: 0.776
- Precision-recall AUC: 0.552
- Brier score: 0.135
- Precision: 0.338
- Recall: 0.808
- F1: 0.477
- Illustrative cost per 1,000 clients: 561.8

These are held-out test estimates for this historical dataset, not guaranteed
future business outcomes.

## Explanation

The leading permutation-importance features were: PAY_0, LIMIT_BAL, PAY_2, BILL_AMT1, PAY_3.
The leading mean-absolute SHAP features were: PAY_0, PAY_2, PAY_3, LIMIT_BAL, PAY_4. Importance means the
model relied on a feature for prediction; it does not show that changing the
feature would cause default risk to change.

## Subgroup audit

Error rates and mean predicted probabilities were calculated by sex, education,
marital-status and age groups. The audit is descriptive. Small groups,
historical selection effects and unobserved confounding prevent a fairness or
discrimination claim from these numbers alone.

## Limitations

1. The data come from one market and one historical period.
2. The illustrative 5:1 cost ratio is not an empirically estimated bank cost.
3. Repeated feature combinations and undocumented category codes were retained
   or transparently grouped rather than silently removed.
4. Calibration and subgroup behaviour may drift in another population.
5. The project supports learning and model comparison, not automated lending.
