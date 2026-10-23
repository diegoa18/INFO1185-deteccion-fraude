"""Evaluación final, métricas, matrices de confusión y comparación de modelos."""

from .metrics import (
    METRIC_NAMES,
    Issue8EvaluationResults,
    compute_binary_fraud_metrics,
    compute_cv_metrics_for_selected_models,
    evaluate_extreme_baselines,
    evaluate_models_on_test,
    identify_best_model,
    plot_confusion_matrices_grid,
    plot_metrics_comparison,
    run_issue8_evaluation,
)

__all__ = [
    "METRIC_NAMES",
    "Issue8EvaluationResults",
    "compute_binary_fraud_metrics",
    "compute_cv_metrics_for_selected_models",
    "evaluate_extreme_baselines",
    "evaluate_models_on_test",
    "identify_best_model",
    "plot_confusion_matrices_grid",
    "plot_metrics_comparison",
    "run_issue8_evaluation",
]
