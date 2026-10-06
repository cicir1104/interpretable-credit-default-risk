# Learning Map

This document separates prior course knowledge from new machine-learning ideas.
Each modelling stage should be explainable in plain language before the next
stage begins.

## 1 Knowledge already supported by completed courses

| Course foundation | How it is used in this project |
| --- | --- |
| Introduction to Probability and Statistics | Random samples, conditional probability, uncertainty and sampling variation |
| Statistical Modelling I | Response and predictor variables, regression structure, assumptions and model comparison |
| Statistical Distribution Theory | Bernoulli outcomes, likelihood-based reasoning and probability distributions |
| Linear Algebra I and II | Matrix representation of predictors and numerical model fitting |
| Operational Research and Mathematical Computing | Reproducible computation and decisions under constraints or unequal costs |
| Stochastic Processes | Thinking about uncertainty and dependence across monthly repayment records |
| Financial Mathematics | Interpreting default risk and asymmetric financial losses |
| Microeconomics and Accounting and Finance | Connecting a statistical prediction to incentives and lending decisions |

## 2 New concepts to learn during the project

### 2.1 Statistical classification

1. Binary targets and class probabilities
2. Logistic function, log-odds and odds ratios
3. Train, validation and test roles
4. Stratified cross-validation
5. Confusion matrix, precision, recall and F1 score
6. ROC-AUC and precision-recall curves
7. Probability calibration
8. Class imbalance and why accuracy alone is insufficient

### 2.2 Non-linear machine learning

1. Decision-tree splits and impurity
2. Overfitting, tree depth and regularisation
3. Bagging and random forests
4. Boosting and sequential error correction
5. Hyperparameter tuning inside cross-validation

### 2.3 Interpretation and decisions

1. Decision thresholds and unequal error costs
2. Coefficient interpretation versus feature importance
3. Permutation importance and SHAP values
4. Subgroup error rates and calibration
5. Limits of fairness claims from historical observational data

## 3 Explanation checkpoints

Before accepting a result, be able to answer:

- What information was available at the prediction date?
- Why is the target binary, and what does a predicted probability mean?
- Why was the data split before transformations were learned?
- What error does each metric emphasise?
- What changes when the classification threshold moves?
- Why might a more flexible model overfit?
- Does the model estimate association, prediction or causation?
- Which conclusions are limited by the dataset's country and 2005 time period?

## 4 Concepts intentionally postponed

Neural networks, deep learning and time-series forecasting are not required for
this first project. They add complexity without improving the main learning
goal: understanding a complete, defensible supervised-learning workflow.
