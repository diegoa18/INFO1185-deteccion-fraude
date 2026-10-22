"""Ejecuta la integración y evaluación final de las 8 configuraciones (Issue #8).

Ejemplos desde la raíz del repositorio:

    .venv/bin/python scripts/run_issue_8.py --sample-size 30000
    .venv/bin/python scripts/run_issue_8.py --full-dataset

Las configuraciones se seleccionan únicamente con validación cruzada sobre el
70 % de entrenamiento (Issue #7) y luego se evalúan una sola vez sobre el 30 %
de prueba reservado. Los resultados de validación y prueba se reportan por
separado conforme a la sección 4.6 del enunciado.
"""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_issue_7 import (
    FIGURES_DIRECTORY,
    TABLES_DIRECTORY,
    _assert_enough_fraud_for_cv,
    build_argument_parser,
    select_experiment_sample,
)
from src.config import RANDOM_STATE
from src.data.load_data import load_dataset, separate_features_target
from src.data.preprocess import stratified_train_test_split
from src.evaluation import (
    plot_confusion_matrices_grid,
    plot_metrics_comparison,
    run_issue8_evaluation,
)
from src.validation import create_issue7_cv, materialize_stratified_folds


def main(argv: list[str] | None = None) -> None:
    arguments = build_argument_parser().parse_args(argv)
    started_at = perf_counter()
    TABLES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset()
    source_X, source_y = separate_features_target(dataframe)
    X, y, sampling_mode = select_experiment_sample(
        source_X,
        source_y,
        sample_size=arguments.sample_size,
        full_dataset=arguments.full_dataset,
    )

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        random_state=RANDOM_STATE,
    )
    _assert_enough_fraud_for_cv(y_train)

    shared_cv = create_issue7_cv()
    shared_folds = materialize_stratified_folds(X_train, y_train, cv=shared_cv)

    evaluation = run_issue8_evaluation(
        X_train,
        X_test,
        y_train,
        y_test,
        folds=shared_folds,
        random_state=RANDOM_STATE,
    )

    # Guardar tablas separadas de validación cruzada y prueba (Sección 4.6)
    val_path = TABLES_DIRECTORY / "issue_8_validation_metrics.csv"
    test_path = TABLES_DIRECTORY / "issue_8_test_metrics.csv"
    combined_path = TABLES_DIRECTORY / "issue_8_combined_comparison.csv"
    cm_path = TABLES_DIRECTORY / "issue_8_confusion_matrices.csv"
    baselines_path = TABLES_DIRECTORY / "issue_8_extreme_baselines.csv"
    best_path = TABLES_DIRECTORY / "issue_8_best_model.csv"

    evaluation.validation_metrics.to_csv(val_path, index=False)
    evaluation.test_metrics.to_csv(test_path, index=False)
    evaluation.combined_comparison.to_csv(combined_path, index=False)
    evaluation.confusion_matrices.to_csv(cm_path, index=False)
    evaluation.extreme_baselines.to_csv(baselines_path, index=False)
    evaluation.best_model_summary.to_csv(best_path, index=False)

    # Generar figuras comparativas y matrices de confusión
    cm_fig_path = FIGURES_DIRECTORY / "issue_8_confusion_matrices.png"
    comp_fig_path = FIGURES_DIRECTORY / "issue_8_metrics_comparison.png"

    fig_cm, _ = plot_confusion_matrices_grid(evaluation.confusion_matrices, cm_fig_path)
    fig_cm.clear()
    fig_comp, _ = plot_metrics_comparison(evaluation.test_metrics, comp_fig_path)
    fig_comp.clear()

    elapsed = perf_counter() - started_at

    print(
        f"Modo={sampling_mode} | Filas={len(y)} "
        f"(Train={len(y_train)}, Test={len(y_test)}) | Semilla={RANDOM_STATE}"
    )
    print("\n1. RESULTADOS EN VALIDACIÓN CRUZADA (5 FOLDS SOBRE 70 % TRAIN):")
    print(evaluation.validation_metrics.to_string(index=False))

    print("\n2. RESULTADOS EN CONJUNTO DE PRUEBA RESERVADO (30 % TEST):")
    print(evaluation.test_metrics.to_string(index=False))

    print("\n3. MATRICES DE CONFUSIÓN EN PRUEBA:")
    print(evaluation.confusion_matrices.to_string(index=False))

    print("\n4. CLASIFICADORES EXTREMOS DE REFERENCIA (SECCIÓN 4.7):")
    print(evaluation.extreme_baselines.to_string(index=False))

    print("\n5. MEJOR CONFIGURACIÓN SELECCIONADA (POR CV) Y SU DESEMPEÑO EN PRUEBA:")
    print(evaluation.best_model_summary.to_string(index=False))

    print(f"\nTiempo total: {elapsed:.2f} s")
    print(f"Tablas guardadas en: {TABLES_DIRECTORY}")
    print(f"Figuras guardadas en: {FIGURES_DIRECTORY}")


if __name__ == "__main__":
    main()
