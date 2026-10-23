from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PREDICTOR_COLUMNS, TARGET_COLUMN
from src.data.preprocess import stratified_train_test_split
from src.evaluation import (
    METRIC_NAMES,
    compute_binary_fraud_metrics,
    evaluate_extreme_baselines,
    plot_confusion_matrices_grid,
    plot_metrics_comparison,
    run_issue8_evaluation,
)


@pytest.fixture
def synthetic_fraud_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n_legit = 120
    n_fraud = 30
    X_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, len(PREDICTOR_COLUMNS)))
    X_fraud = rng.normal(loc=1.2, scale=1.0, size=(n_fraud, len(PREDICTOR_COLUMNS)))
    features = pd.DataFrame(
        np.vstack([X_legit, X_fraud]),
        columns=PREDICTOR_COLUMNS,
    )
    target = pd.Series(
        np.repeat([0, 1], [n_legit, n_fraud]),
        name=TARGET_COLUMN,
    )
    return features, target


def test_compute_binary_fraud_metrics_satisfies_fnr_identity():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])

    metrics = compute_binary_fraud_metrics(y_true, y_pred)

    # TP=3, FN=1, TN=3, FP=1
    assert metrics["tp"] == 3
    assert metrics["fn"] == 1
    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert metrics["sensitivity"] == pytest.approx(3 / 4)
    assert metrics["fnr"] == pytest.approx(1 / 4)
    assert metrics["fnr"] == pytest.approx(1.0 - metrics["sensitivity"])
    assert metrics["specificity"] == pytest.approx(3 / 4)
    assert metrics["precision"] == pytest.approx(3 / 4)
    assert metrics["accuracy"] == pytest.approx(6 / 8)


def test_extreme_baselines_match_theoretical_expectations():
    y_test = pd.Series(np.repeat([0, 1], [90, 10]), name=TARGET_COLUMN)
    baselines = evaluate_extreme_baselines(y_test).set_index("baseline")

    always_legit = baselines.loc["Siempre legítima (Class = 0)"]
    assert always_legit["sensitivity"] == pytest.approx(0.0)
    assert always_legit["fnr"] == pytest.approx(1.0)
    assert always_legit["specificity"] == pytest.approx(1.0)
    assert always_legit["accuracy"] == pytest.approx(0.90)

    always_fraud = baselines.loc["Siempre fraude (Class = 1)"]
    assert always_fraud["sensitivity"] == pytest.approx(1.0)
    assert always_fraud["fnr"] == pytest.approx(0.0)
    assert always_fraud["specificity"] == pytest.approx(0.0)
    assert always_fraud["precision"] == pytest.approx(0.10)


def test_run_issue8_evaluation_integrates_all_eight_configurations(
    synthetic_fraud_data,
    tmp_path,
):
    features, target = synthetic_fraud_data
    X_train, X_test, y_train, y_test = stratified_train_test_split(features, target)

    small_spaces = {
        "knn": [
            {
                "classifier__n_neighbors": [3],
                "classifier__weights": ["uniform"],
                "classifier__metric": ["euclidean"],
            }
        ],
        "svm": [{"classifier__kernel": ["linear"], "classifier__C": [1.0]}],
        "decision_tree": [
            {
                "classifier__max_depth": [3],
                "classifier__min_samples_split": [2],
                "classifier__min_samples_leaf": [1],
                "classifier__criterion": ["gini"],
            }
        ],
        "custom_random_forest": [
            {
                "classifier__n_estimators": [3],
                "classifier__max_depth": [3],
                "classifier__max_features": ["sqrt"],
            }
        ],
    }

    results = run_issue8_evaluation(
        X_train,
        X_test,
        y_train,
        y_test,
        search_spaces=small_spaces,
        n_bootstrap=20,
    )

    assert len(results.validation_metrics) == 8
    assert len(results.test_metrics) == 8
    assert len(results.confusion_matrices) == 8
    assert len(results.best_model_summary) == 1

    for metric in METRIC_NAMES:
        assert f"{metric}_cv_mean" in results.validation_metrics.columns
        assert f"{metric}_cv_std" in results.validation_metrics.columns
        assert f"{metric}_test" in results.test_metrics.columns
        assert f"{metric}_test_std" in results.test_metrics.columns

    # Verificar FNR = 1 - sensibilidad en validación y prueba
    np.testing.assert_allclose(
        results.validation_metrics["fnr_cv_mean"],
        1.0 - results.validation_metrics["sensitivity_cv_mean"],
    )
    np.testing.assert_allclose(
        results.test_metrics["fnr_test"],
        1.0 - results.test_metrics["sensitivity_test"],
    )

    # Verificar que las matrices de confusión suman exactamente len(y_test)
    cm = results.confusion_matrices
    assert ((cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]) == len(y_test)).all()

    # Verificar generación de figuras sin errores
    cm_fig_path = tmp_path / "cm.png"
    comp_fig_path = tmp_path / "comp.png"
    fig1, _ = plot_confusion_matrices_grid(results.confusion_matrices, cm_fig_path)
    fig1.clear()
    fig2, _ = plot_metrics_comparison(results.test_metrics, comp_fig_path)
    fig2.clear()
    assert cm_fig_path.is_file()
    assert comp_fig_path.is_file()
