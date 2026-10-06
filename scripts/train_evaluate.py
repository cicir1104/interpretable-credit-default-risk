"""Train, compare, explain and audit credit-default models.

The test set is isolated before model selection. Hyperparameters and the
illustrative cost-sensitive threshold are chosen on validation data only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.data import build_project_tables, load_raw_data  # noqa: E402
from credit_risk.evaluation import (  # noqa: E402
    classification_metrics,
    select_cost_threshold,
)


RANDOM_STATE = 42
FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def model_candidates() -> dict[str, list[tuple[str, object, dict[str, object]]]]:
    """Return a deliberately small, reproducible candidate grid."""
    return {
        "dummy": [
            ("prior", DummyClassifier(strategy="prior"), {"strategy": "prior"})
        ],
        "logistic_regression": [
            (
                "scaled_logistic",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=3000,
                                solver="lbfgs",
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
                {"scaling": "standard", "class_weight": None},
            )
        ],
        "decision_tree": [
            (
                f"depth_{depth}_leaf_{leaf}",
                DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=RANDOM_STATE,
                ),
                {"max_depth": depth, "min_samples_leaf": leaf},
            )
            for depth in [3, 5, 7]
            for leaf in [20, 100]
        ],
        "random_forest": [
            (
                f"depth_{depth}_leaf_{leaf}",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                {
                    "n_estimators": 300,
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                },
            )
            for depth, leaf in [(6, 20), (10, 10), (None, 5)]
        ],
        "gradient_boosting": [
            (
                f"leaves_{leaves}_rate_{str(rate).replace('.', '_')}",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    max_leaf_nodes=leaves,
                    learning_rate=rate,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
                {
                    "max_iter": 300,
                    "max_leaf_nodes": leaves,
                    "learning_rate": rate,
                    "min_samples_leaf": 30,
                    "l2_regularization": 1.0,
                },
            )
            for leaves, rate in [(7, 0.03), (15, 0.03), (15, 0.05), (31, 0.03)]
        ],
    }


def split_indices(target: pd.Series) -> dict[str, np.ndarray]:
    """Create a fixed 60/20/20 stratified split."""
    indices = np.arange(len(target))
    train_validation, test = train_test_split(
        indices,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=target.iloc[train_validation],
        random_state=RANDOM_STATE,
    )
    return {"train": train, "validation": validation, "test": test}


def plot_target_balance(target: pd.Series) -> None:
    counts = target.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(["No default (0)", "Default (1)"], counts.values, color=["#4C78A8", "#E45756"])
    ax.set_title("Target class balance")
    ax.set_ylabel("Clients")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 250, f"{count:,}\n({count/len(target):.1%})", ha="center")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_balance.png", dpi=180)
    plt.close(fig)


def train_and_select(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, float], dict[str, pd.DataFrame]]:
    """Select one candidate per model family using validation cost."""
    selected_models: dict[str, object] = {}
    selected_thresholds: dict[str, float] = {}
    selected_threshold_tables: dict[str, pd.DataFrame] = {}
    tuning_rows: list[dict[str, object]] = []

    for family, candidates in model_candidates().items():
        family_results: list[tuple[float, float, str, object, float, pd.DataFrame]] = []
        for candidate_name, model, parameters in candidates:
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_validation)[:, 1]
            threshold, threshold_rows = select_cost_threshold(
                y_validation.to_numpy(),
                probabilities,
                false_negative_cost=FALSE_NEGATIVE_COST,
                false_positive_cost=FALSE_POSITIVE_COST,
            )
            metrics = classification_metrics(
                y_validation.to_numpy(),
                probabilities,
                threshold,
                false_negative_cost=FALSE_NEGATIVE_COST,
                false_positive_cost=FALSE_POSITIVE_COST,
            )
            tuning_rows.append(
                {
                    "model": family,
                    "candidate": candidate_name,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    **metrics,
                }
            )
            family_results.append(
                (
                    float(metrics["illustrative_cost"]),
                    -float(metrics["roc_auc"]),
                    candidate_name,
                    model,
                    threshold,
                    pd.DataFrame(threshold_rows),
                )
            )

        best = min(family_results, key=lambda item: (item[0], item[1]))
        selected_models[family] = best[3]
        selected_thresholds[family] = best[4]
        selected_threshold_tables[family] = best[5]

    return (
        selected_models,
        pd.DataFrame(tuning_rows),
        selected_thresholds,
        selected_threshold_tables,
    )


def evaluate_selected_models(
    models: dict[str, object],
    thresholds: dict[str, float],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate selected candidates once on the untouched test set."""
    rows: list[dict[str, object]] = []
    probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        proba = model.predict_proba(x_test)[:, 1]
        probabilities[name] = proba
        for threshold_type, threshold in [
            ("default_0.50", 0.50),
            ("validation_cost", thresholds[name]),
        ]:
            rows.append(
                {
                    "model": name,
                    "threshold_type": threshold_type,
                    **classification_metrics(
                        y_test.to_numpy(),
                        proba,
                        threshold,
                        false_negative_cost=FALSE_NEGATIVE_COST,
                        false_positive_cost=FALSE_POSITIVE_COST,
                    ),
                }
            )
    return pd.DataFrame(rows), probabilities


def choose_final_model(tuning: pd.DataFrame) -> str:
    """Choose the non-dummy family with the lowest validation scenario cost."""
    best_per_family = (
        tuning.sort_values(["illustrative_cost", "roc_auc"], ascending=[True, False])
        .groupby("model", as_index=False)
        .first()
    )
    eligible = best_per_family[best_per_family["model"] != "dummy"]
    return str(
        eligible.sort_values(["illustrative_cost", "roc_auc"], ascending=[True, False])
        .iloc[0]["model"]
    )


def plot_model_curves(
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for name, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        metrics = classification_metrics(y_test.to_numpy(), proba, 0.5)
        ax.plot(fpr, tpr, label=f"{name} (AUC {metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Test-set ROC curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for name, proba in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        metrics = classification_metrics(y_test.to_numpy(), proba, 0.5)
        ax.plot(recall, precision, label=f"{name} (AP {metrics['pr_auc']:.3f})")
    ax.axhline(y_test.mean(), linestyle="--", color="grey", linewidth=1, label="Default prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Test-set precision-recall curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "precision_recall_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for name, proba in probabilities.items():
        observed, predicted = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate", title="Test-set calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration.png", dpi=180)
    plt.close(fig)


def save_explanations(
    models: dict[str, object],
    final_model_name: str,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save logistic, permutation and SHAP feature summaries."""
    logistic = models["logistic_regression"]
    coefficients = logistic.named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame(
        {
            "feature": x_test.columns,
            "standardized_coefficient": coefficients,
            "odds_ratio_per_standard_deviation": np.exp(coefficients),
        }
    ).sort_values("standardized_coefficient", ascending=False)
    coefficient_table.to_csv(REPORT_DIR / "logistic_coefficients.csv", index=False)

    final_model = models[final_model_name]
    permutation = permutation_importance(
        final_model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=7,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    permutation_table = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    permutation_table.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)

    sample = x_test.sample(n=min(750, len(x_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(final_model)
    shap_values = np.asarray(explainer.shap_values(sample))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    shap_table = pd.DataFrame(
        {
            "feature": sample.columns,
            "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    shap_table.to_csv(REPORT_DIR / "shap_importance.csv", index=False)

    top = permutation_table.head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4C78A8")
    ax.set(xlabel="Decrease in test ROC AUC", title=f"Permutation importance: {final_model_name}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)

    return coefficient_table, permutation_table, shap_table


def subgroup_audit(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """Describe subgroup outcomes without treating gaps as causal effects."""
    audit = demographics.copy()
    audit["actual"] = y_test.to_numpy()
    audit["probability"] = probabilities
    audit["prediction"] = (probabilities >= threshold).astype(int)
    rows: list[dict[str, object]] = []
    for field in ["sex_group", "education_group", "marriage_group", "age_group"]:
        for group, frame in audit.groupby(field, dropna=False):
            tn, fp, fn, tp = confusion_matrix(
                frame["actual"], frame["prediction"], labels=[0, 1]
            ).ravel()
            rows.append(
                {
                    "audit_field": field,
                    "group": str(group),
                    "n": int(len(frame)),
                    "observed_default_rate": float(frame["actual"].mean()),
                    "mean_predicted_probability": float(frame["probability"].mean()),
                    "recall": float(tp / (tp + fn)) if tp + fn else np.nan,
                    "false_positive_rate": float(fp / (fp + tn)) if fp + tn else np.nan,
                    "precision": float(tp / (tp + fp)) if tp + fp else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_technical_report(
    split_summary: dict[str, object],
    metrics: pd.DataFrame,
    final_model: str,
    final_threshold: float,
    permutation_table: pd.DataFrame,
    shap_table: pd.DataFrame,
) -> None:
    row = metrics[
        (metrics["model"] == final_model)
        & (metrics["threshold_type"] == "validation_cost")
    ].iloc[0]
    top_permutation = ", ".join(permutation_table.head(5)["feature"].tolist())
    top_shap = ", ".join(shap_table.head(5)["feature"].tolist())
    REPORT_DIR.joinpath("technical_report.md").write_text(
        f"""# Technical report: interpretable credit-default risk modelling

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

- Training: {split_summary['train']['rows']:,} rows
- Validation: {split_summary['validation']['rows']:,} rows
- Test: {split_summary['test']['rows']:,} rows
- Random seed: {RANDOM_STATE}
- Hyperparameters and thresholds selected without using the test set
- Scenario cost: missed default = {FALSE_NEGATIVE_COST:g}, unnecessary high-risk
  flag = {FALSE_POSITIVE_COST:g}

## Models

Dummy prior, scaled logistic regression, decision tree, random forest and
histogram gradient boosting were compared. The validation scenario selected
**{final_model}** with a threshold of **{final_threshold:.2f}**.

## Final test result

- ROC AUC: {row['roc_auc']:.3f}
- Precision-recall AUC: {row['pr_auc']:.3f}
- Brier score: {row['brier']:.3f}
- Precision: {row['precision']:.3f}
- Recall: {row['recall']:.3f}
- F1: {row['f1']:.3f}
- Illustrative cost per 1,000 clients: {row['cost_per_1000']:.1f}

These are held-out test estimates for this historical dataset, not guaranteed
future business outcomes.

## Explanation

The leading permutation-importance features were: {top_permutation}.
The leading mean-absolute SHAP features were: {top_shap}. Importance means the
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
"""
    )


def main() -> None:
    sns.set_theme(style="whitegrid")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(RAW_FILE)
    tables = build_project_tables(raw)
    x = tables["predictors"]
    y = tables["target"]["default_next_month"]
    demographics = tables["demographic_audit"]
    splits = split_indices(y)

    split_summary: dict[str, object] = {
        "random_state": RANDOM_STATE,
        "strategy": "stratified 60/20/20 train/validation/test",
    }
    for name, indices in splits.items():
        split_summary[name] = {
            "rows": int(len(indices)),
            "default_count": int(y.iloc[indices].sum()),
            "default_rate": float(y.iloc[indices].mean()),
        }
    (REPORT_DIR / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2) + "\n"
    )

    x_train, y_train = x.iloc[splits["train"]], y.iloc[splits["train"]]
    x_validation, y_validation = (
        x.iloc[splits["validation"]],
        y.iloc[splits["validation"]],
    )
    x_test, y_test = x.iloc[splits["test"]], y.iloc[splits["test"]]

    plot_target_balance(y)
    models, tuning, thresholds, threshold_tables = train_and_select(
        x_train, y_train, x_validation, y_validation
    )
    tuning = tuning.round(12)
    tuning.to_csv(REPORT_DIR / "tuning_results.csv", index=False)
    metrics, test_probabilities = evaluate_selected_models(
        models, thresholds, x_test, y_test
    )
    metrics = metrics.round(12)
    metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    final_model_name = choose_final_model(tuning)
    final_threshold = thresholds[final_model_name]

    plot_model_curves(y_test, test_probabilities)
    threshold_table = threshold_tables[final_model_name].round(12)
    threshold_table.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(threshold_table["threshold"], threshold_table["cost_per_1000"], color="#E45756")
    ax.axvline(final_threshold, linestyle="--", color="#4C78A8", label=f"Selected {final_threshold:.2f}")
    ax.set(xlabel="Classification threshold", ylabel="Illustrative cost per 1,000", title=f"Validation cost curve: {final_model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "threshold_cost.png", dpi=180)
    plt.close(fig)

    selected_predictions = (
        test_probabilities[final_model_name] >= final_threshold
    ).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, selected_predictions),
        display_labels=["No default", "Default"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test confusion matrix: {final_model_name}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    _, permutation_table, shap_table = save_explanations(
        models, final_model_name, x_test, y_test
    )
    subgroup = subgroup_audit(
        y_test.reset_index(drop=True),
        test_probabilities[final_model_name],
        final_threshold,
        demographics.iloc[splits["test"]].reset_index(drop=True),
    ).round(12)
    subgroup.to_csv(REPORT_DIR / "subgroup_audit.csv", index=False)

    summary = {
        "selected_model": final_model_name,
        "validation_selected_threshold": final_threshold,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "test_metrics_at_selected_threshold": metrics[
            (metrics["model"] == final_model_name)
            & (metrics["threshold_type"] == "validation_cost")
        ].iloc[0].to_dict(),
    }
    (REPORT_DIR / "model_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_technical_report(
        split_summary,
        metrics,
        final_model_name,
        final_threshold,
        permutation_table,
        shap_table,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
