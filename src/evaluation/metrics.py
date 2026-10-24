"""Evaluación final de las configuraciones seleccionadas sin reutilizar test.

La variabilidad de validación se calcula entre los cinco folds. En prueba se
reporta una única evaluación y, por separado, la dispersión estimada mediante
bootstrap estratificado de las predicciones fijas; esta última no representa
nuevos entrenamientos ni se usa para seleccionar modelos.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import RANDOM_STATE, TARGET_COLUMN
from src.validation.experiments import (
    DEFAULT_SEARCH_SPACES,
    ISSUE7_K_FEATURES,
    MODEL_LABELS,
    MODEL_SCALERS,
    RECALL_TIE_TOLERANCE,
    Issue7ExperimentResults,
    _PreprocessingCache,
    _classifier_parameters,
    create_experiment_estimator,
    create_issue7_cv,
    materialize_stratified_folds,
    run_issue7_experiments,
)

METRIC_NAMES = (
    "sensitivity",
    "specificity",
    "precision",
    "f1_score",
    "accuracy",
    "fnr",
)


@dataclass
class Issue8EvaluationResults:
    """Agrupa tablas y estimadores producidos por la evaluación final."""

    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    combined_comparison: pd.DataFrame
    confusion_matrices: pd.DataFrame
    extreme_baselines: pd.DataFrame
    best_model_summary: pd.DataFrame
    issue7_results: Issue7ExperimentResults


def compute_binary_fraud_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Calcula las 6 métricas oficiales y el desglose de la matriz de confusión."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("y_true e y_pred deben ser arreglos unidimensionales.")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true e y_pred deben tener la misma longitud.")
    if len(y_true_arr) == 0:
        return {metric: 0.0 for metric in METRIC_NAMES} | {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    sensitivity = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
        "fnr": fnr,
    }


def _bootstrap_test_metric_std(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 200,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Estima dispersión muestral en test sin reentrenar ni seleccionar modelos."""
    rng = np.random.default_rng(random_state)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    # Manejo de casos extremos: si falta una clase, no podemos calcular desviación real
    if len(pos_idx) == 0 or len(neg_idx) == 0 or n_bootstrap < 2:
        return {f"{metric}_test_std": 0.0 for metric in METRIC_NAMES}

    samples: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    for _ in range(n_bootstrap):
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        boot_idx = np.concatenate([boot_neg, boot_pos])
        metrics = compute_binary_fraud_metrics(y_true[boot_idx], y_pred[boot_idx])
        for metric in METRIC_NAMES:
            samples[metric].append(float(metrics[metric]))

    return {
        f"{metric}_test_std": float(np.std(values, ddof=0))
        for metric, values in samples.items()
    }


def compute_cv_metrics_for_selected_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    issue7_results: Issue7ExperimentResults,
    *,
    k_features: int = ISSUE7_K_FEATURES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Calcula media y desviación estándar de las 6 métricas en los 5 folds de CV."""
    cache = _PreprocessingCache(
        X_train,
        y_train,
        issue7_results.folds,
        k_features=k_features,
        random_state=random_state,
    )
    rows: list[dict[str, Any]] = []

    for _, summary_row in issue7_results.summary.iterrows():
        model_label = str(summary_row["model"])
        balanced = bool(summary_row["balanced"])
        model_key = next(key for key, label in MODEL_LABELS.items() if label == model_label)
        scaler_kind = MODEL_SCALERS[model_key]
        best_params = json.loads(str(summary_row["best_hyperparameters"]))
        prefixed_params = {f"classifier__{k}": v for k, v in best_params.items()}
        classifier_params = _classifier_parameters(prefixed_params)

        base_estimator = create_experiment_estimator(
            model_key,
            balanced=balanced,
            random_state=random_state,
            k_features=k_features,
        )

        fold_metrics: list[dict[str, float | int]] = []
        for fold_number in range(1, len(issue7_results.folds) + 1):
            prepared, fold_train_X, fold_train_y, _, _ = cache.get_training_data(
                scaler_kind,
                fold_number,
                balanced=balanced,
            )
            classifier = clone(base_estimator.classifier).set_params(**classifier_params)
            classifier.fit(fold_train_X, fold_train_y)
            predictions = classifier.predict(prepared.validation_features)
            fold_metrics.append(
                compute_binary_fraud_metrics(prepared.validation_target, predictions)
            )

        fold_df = pd.DataFrame(fold_metrics)
        record: dict[str, Any] = {
            "model_key": model_key,
            "model": model_label,
            "balanced": balanced,
            "balancing_strategy": str(summary_row["balancing_strategy"]),
            "best_hyperparameters": str(summary_row["best_hyperparameters"]),
            "selected_features": str(summary_row.get("selected_features", "[]")),
        }
        for metric in METRIC_NAMES:
            record[f"{metric}_cv_mean"] = float(fold_df[metric].mean())
            record[f"{metric}_cv_std"] = float(fold_df[metric].std(ddof=0))
        rows.append(record)

    return pd.DataFrame(rows)


def evaluate_models_on_test(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    issue7_results: Issue7ExperimentResults,
    *,
    n_bootstrap: int = 200,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa los 8 modelos reentrenados sobre el conjunto de prueba reservado (30 %)."""
    if TARGET_COLUMN in X_test.columns:
        raise ValueError(f"'{TARGET_COLUMN}' no puede estar en X_test.")
    if len(X_test) != len(y_test) or X_test.empty:
        raise ValueError("X_test e y_test deben tener la misma cantidad de filas y no ser vacíos.")

    y_test_arr = y_test.to_numpy(dtype=int)
    test_rows: list[dict[str, Any]] = []
    cm_rows: list[dict[str, Any]] = []

    for _, summary_row in issue7_results.summary.iterrows():
        model_label = str(summary_row["model"])
        balanced = bool(summary_row["balanced"])
        model_key = next(key for key, label in MODEL_LABELS.items() if label == model_label)
        estimator = issue7_results.best_estimators[(model_key, balanced)]

        predictions = estimator.predict(X_test)
        metrics = compute_binary_fraud_metrics(y_test_arr, predictions)
        boot_stds = _bootstrap_test_metric_std(
            y_test_arr,
            predictions,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        condition_label = f"{model_label} ({'Con balanceo' if balanced else 'Sin balanceo'})"
        test_record: dict[str, Any] = {
            "model_key": model_key,
            "model": model_label,
            "balanced": balanced,
            "condition_label": condition_label,
            "balancing_strategy": str(summary_row["balancing_strategy"]),
            "best_hyperparameters": str(summary_row["best_hyperparameters"]),
        }
        for metric in METRIC_NAMES:
            test_record[f"{metric}_test"] = float(metrics[metric])
            test_record[f"{metric}_test_std"] = float(boot_stds[f"{metric}_test_std"])
        test_record["tn"] = int(metrics["tn"])
        test_record["fp"] = int(metrics["fp"])
        test_record["fn"] = int(metrics["fn"])
        test_record["tp"] = int(metrics["tp"])
        test_rows.append(test_record)

        cm_rows.append(
            {
                "model_key": model_key,
                "model": model_label,
                "balanced": balanced,
                "condition_label": condition_label,
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "tp": int(metrics["tp"]),
                "total_legitimate": int(metrics["tn"] + metrics["fp"]),
                "total_fraud": int(metrics["tp"] + metrics["fn"]),
            }
        )

    return pd.DataFrame(test_rows), pd.DataFrame(cm_rows)


def evaluate_extreme_baselines(y_test: pd.Series | np.ndarray) -> pd.DataFrame:
    """Evalúa los clasificadores triviales exigidos en las preguntas 1 y 5 de la sección 4.7."""
    y_true = np.asarray(y_test, dtype=int)
    all_legitimate = np.zeros_like(y_true)
    all_fraud = np.ones_like(y_true)

    legit_metrics = compute_binary_fraud_metrics(y_true, all_legitimate)
    fraud_metrics = compute_binary_fraud_metrics(y_true, all_fraud)

    return pd.DataFrame(
        [
            {
                "baseline": "Siempre legítima (Class = 0)",
                **legit_metrics,
            },
            {
                "baseline": "Siempre fraude (Class = 1)",
                **fraud_metrics,
            },
        ]
    )


def identify_best_model(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Identifica la mejor configuración global seleccionada estrictamente por CV."""
    merged = validation_metrics.merge(
        test_metrics,
        on=["model_key", "model", "balanced", "balancing_strategy", "best_hyperparameters"],
        how="inner",
    )
    max_cv_sens = merged["sensitivity_cv_mean"].max()
    candidates = merged.loc[
        np.isclose(
            merged["sensitivity_cv_mean"],
            max_cv_sens,
            rtol=0.0,
            atol=RECALL_TIE_TOLERANCE,
        )
    ].copy()
    candidates = candidates.sort_values(
        by=["precision_cv_mean", "f1_score_cv_mean"],
        ascending=[False, False],
        kind="mergesort",
    )
    winner = candidates.iloc[[0]].copy()
    winner["selection_criterion"] = (
        "Mayor sensibilidad media en validación cruzada (equiv. menor FNR medio), "
        "con precisión media en CV como desempate."
    )
    return winner


def run_issue8_evaluation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    issue7_results: Issue7ExperimentResults | None = None,
    folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    search_spaces: Mapping[str, list[dict[str, list[Any]]]] | None = None,
    n_bootstrap: int = 200,
    random_state: int = RANDOM_STATE,
) -> Issue8EvaluationResults:
    """Ejecuta el flujo completo de integración y evaluación final en prueba."""
    if random_state != RANDOM_STATE:
        raise ValueError(f"random_state debe ser exactamente {RANDOM_STATE} en todo el pipeline para evitar fuga o sesgos.")
    if issue7_results is None:
        shared_folds = (
            tuple(folds)
            if folds is not None
            else materialize_stratified_folds(
                X_train,
                y_train,
                cv=create_issue7_cv(),
            )
        )
        issue7_results = run_issue7_experiments(
            X_train,
            y_train,
            folds=shared_folds,
            search_spaces=search_spaces,
        )

    validation_metrics = compute_cv_metrics_for_selected_models(
        X_train,
        y_train,
        issue7_results,
        random_state=random_state,
    )
    test_metrics, confusion_matrices = evaluate_models_on_test(
        X_test,
        y_test,
        issue7_results,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    combined_comparison = validation_metrics.merge(
        test_metrics,
        on=["model_key", "model", "balanced", "balancing_strategy", "best_hyperparameters"],
        how="inner",
    )
    extreme_baselines = evaluate_extreme_baselines(y_test)
    best_model_summary = identify_best_model(validation_metrics, test_metrics)

    return Issue8EvaluationResults(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        combined_comparison=combined_comparison,
        confusion_matrices=confusion_matrices,
        extreme_baselines=extreme_baselines,
        best_model_summary=best_model_summary,
        issue7_results=issue7_results,
    )


def plot_confusion_matrices_grid(
    confusion_df: pd.DataFrame,
    output_path: str | Path | None = None,
):
    """Genera una figura con las 8 matrices de confusión en el conjunto de prueba."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes_flat = axes.ravel()

    for idx, (_, row) in enumerate(confusion_df.iterrows()):
        ax = axes_flat[idx]
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues" if not row["balanced"] else "Oranges",
            cbar=False,
            xticklabels=["Pred: Legítima (0)", "Pred: Fraude (1)"],
            yticklabels=["Real: Legítima (0)", "Real: Fraude (1)"],
            ax=ax,
        )
        balance_tag = "Con balanceo" if row["balanced"] else "Sin balanceo"
        ax.set_title(f"{row['model']}\n({balance_tag})", fontsize=10, fontweight="bold")

    figure.suptitle(
        "Matrices de confusión en el conjunto de prueba (30 %) — 8 configuraciones",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
    return figure, axes


def plot_metrics_comparison(
    test_metrics: pd.DataFrame,
    output_path: str | Path | None = None,
):
    """Genera gráfico comparativo de Sensibilidad, FNR, Precisión y F1 en prueba."""
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(test_metrics["model"]))
    x = np.arange(len(models))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    unbalanced = test_metrics.loc[~test_metrics["balanced"]].set_index("model").loc[models]
    balanced = test_metrics.loc[test_metrics["balanced"]].set_index("model").loc[models]

    bars1 = axes[0].bar(
        x - width / 2,
        unbalanced["sensitivity_test"],
        width,
        label="Sensibilidad (Sin balanceo)",
        color="#4C78A8",
    )
    bars2 = axes[0].bar(
        x + width / 2,
        balanced["sensitivity_test"],
        width,
        label="Sensibilidad (Con balanceo)",
        color="#F58518",
    )
    axes[0].set_title("Sensibilidad de Fraude en Prueba (Mayor es mejor / Menor FNR)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Sensibilidad (1 - FNR)")
    axes[0].legend()
    axes[0].bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    axes[0].bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)

    bars3 = axes[1].bar(
        x - width / 2,
        unbalanced["f1_score_test"],
        width,
        label="F1-score (Sin balanceo)",
        color="#54A24B",
    )
    bars4 = axes[1].bar(
        x + width / 2,
        balanced["f1_score_test"],
        width,
        label="F1-score (Con balanceo)",
        color="#E45756",
    )
    axes[1].set_title("F1-Score de Fraude en Prueba (Efecto de las falsas alarmas)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("F1-Score")
    axes[1].legend()
    axes[1].bar_label(bars3, fmt="%.3f", padding=2, fontsize=8)
    axes[1].bar_label(bars4, fmt="%.3f", padding=2, fontsize=8)

    fig.suptitle(
        "Comparación de Desempeño en Conjunto de Prueba (Con y Sin Compensación del Desbalance)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150, bbox_inches="tight")
    return fig, axes
