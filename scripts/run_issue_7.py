"""Ejecuta los experimentos reproducibles de desbalance y optimización.

Ejemplos desde la raíz del repositorio:

    .venv/bin/python scripts/run_issue_7.py --sample-size 30000
    .venv/bin/python scripts/run_issue_7.py --full-dataset

La selección de hiperparámetros usa únicamente el 70 % de entrenamiento. El
30 % de test permanece aislado y no se evalúa en este script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

# Permite ejecutar ``python scripts/run_issue_7.py`` desde la raíz sin exigir
# instalar el proyecto como paquete antes de reproducir el experimento.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import N_SPLITS, RANDOM_STATE, TARGET_COLUMN
from src.data.load_data import load_dataset, separate_features_target
from src.data.preprocess import stratified_sample, stratified_train_test_split
from src.imbalance import analyze_class_imbalance, plot_class_imbalance
from src.validation import create_issue7_cv, materialize_stratified_folds, run_issue7_experiments

DEFAULT_SAMPLE_SIZE = 30_000
MINIMUM_SAMPLE_SIZE = 10_000
RESULTS_DIRECTORY = Path("results")
TABLES_DIRECTORY = RESULTS_DIRECTORY / "tables"
FIGURES_DIRECTORY = RESULTS_DIRECTORY / "figures"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta las ocho condiciones de Issue #7 con una muestra estratificada "
            "o con el dataset completo."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Cantidad de filas de la muestra estratificada (mínimo {MINIMUM_SAMPLE_SIZE}; "
            f"por defecto {DEFAULT_SAMPLE_SIZE})."
        ),
    )
    selection.add_argument(
        "--full-dataset",
        action="store_true",
        help="Usa las 284 807 filas originales, sin submuestreo.",
    )
    return parser


def select_experiment_sample(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_size: int | None,
    full_dataset: bool,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Selecciona una única muestra para los cuatro clasificadores.

    Se selecciona antes de la división 70/30 y no aprende parámetros de los
    datos. Después, la división estratificada y toda la validación se hacen
    sobre esa misma muestra, por lo que no existe contaminación entre modelos.
    """
    if full_dataset:
        return X.copy(), y.copy(), "full_dataset"

    effective_sample_size = DEFAULT_SAMPLE_SIZE if sample_size is None else sample_size
    if effective_sample_size < MINIMUM_SAMPLE_SIZE:
        raise ValueError(
            f"--sample-size debe ser al menos {MINIMUM_SAMPLE_SIZE} para conservar "
            f"fraudes suficientes en los {N_SPLITS} folds."
        )
    sampled_X, sampled_y = stratified_sample(
        X,
        y,
        sample_size=effective_sample_size,
        random_state=RANDOM_STATE,
    )
    return sampled_X, sampled_y, "stratified_sample"


def _class_counts(target: pd.Series) -> dict[int, int]:
    counts = target.value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def _build_metadata(
    *,
    mode: str,
    source_target: pd.Series,
    experiment_target: pd.Series,
    train_target: pd.Series,
    test_target: pd.Series,
) -> pd.DataFrame:
    source_counts = _class_counts(source_target)
    experiment_counts = _class_counts(experiment_target)
    train_counts = _class_counts(train_target)
    test_counts = _class_counts(test_target)
    return pd.DataFrame(
        [
            {
                "sampling_mode": mode,
                "random_state": RANDOM_STATE,
                "source_rows": len(source_target),
                "experiment_rows": len(experiment_target),
                "source_class_0": source_counts.get(0, 0),
                "source_class_1": source_counts.get(1, 0),
                "experiment_class_0": experiment_counts.get(0, 0),
                "experiment_class_1": experiment_counts.get(1, 0),
                "train_rows": len(train_target),
                "train_class_0": train_counts.get(0, 0),
                "train_class_1": train_counts.get(1, 0),
                "test_rows_reserved": len(test_target),
                "test_class_0": test_counts.get(0, 0),
                "test_class_1": test_counts.get(1, 0),
                "n_splits": N_SPLITS,
            }
        ]
    )


def _assert_enough_fraud_for_cv(train_target: pd.Series) -> None:
    fraud_count = int((train_target == 1).sum())
    if fraud_count < N_SPLITS:
        raise ValueError(
            f"La partición de entrenamiento solo contiene {fraud_count} fraudes; "
            f"se requieren al menos {N_SPLITS} para la validación cruzada estratificada."
        )


def main(argv: list[str] | None = None) -> None:
    arguments = build_argument_parser().parse_args(argv)
    total_started_at = perf_counter()
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

    source_imbalance = analyze_class_imbalance(source_y)
    experiment_imbalance = analyze_class_imbalance(y)
    print("Desbalance del dataset fuente (Class == 1 es fraude):")
    print(source_imbalance.to_string(index=False, formatters={"percentage": "{:.6f}".format}))
    print(
        f"\nDatos del experimento: modo={sampling_mode}, filas={len(y)}, "
        f"semilla={RANDOM_STATE}."
    )
    print(experiment_imbalance.to_string(index=False, formatters={"percentage": "{:.6f}".format}))
    experiment_imbalance.to_csv(TABLES_DIRECTORY / "issue_7_class_imbalance.csv", index=False)
    source_imbalance.to_csv(TABLES_DIRECTORY / "issue_7_source_class_imbalance.csv", index=False)
    figure, _ = plot_class_imbalance(
        experiment_imbalance,
        FIGURES_DIRECTORY / "issue_7_class_distribution.png",
    )
    figure.clear()

    # La única división oficial: 70 % entrenamiento y 30 % prueba. X_test e
    # y_test permanecen fuera de la optimización, del selector y del sampler.
    X_train, _X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        random_state=RANDOM_STATE,
    )
    _assert_enough_fraud_for_cv(y_train)
    metadata = _build_metadata(
        mode=sampling_mode,
        source_target=source_y,
        experiment_target=y,
        train_target=y_train,
        test_target=y_test,
    )

    shared_cv = create_issue7_cv()
    shared_folds = materialize_stratified_folds(X_train, y_train, cv=shared_cv)
    experiments = run_issue7_experiments(
        X_train,
        y_train,
        folds=shared_folds,
    )

    total_seconds = perf_counter() - total_started_at
    metadata["candidate_evaluations"] = len(experiments.candidate_results)
    metadata["preprocessing_fit_count"] = experiments.preprocessing_fit_count
    metadata["resampling_fit_count"] = experiments.resampling_fit_count
    metadata["total_seconds"] = total_seconds
    metadata.to_csv(TABLES_DIRECTORY / "issue_7_run_metadata.csv", index=False)
    summary = experiments.summary.copy()
    summary["experiment_rows"] = len(y)
    summary["random_state"] = RANDOM_STATE
    summary.to_csv(TABLES_DIRECTORY / "issue_7_cv_results.csv", index=False)
    experiments.candidate_results.to_csv(
        TABLES_DIRECTORY / "issue_7_cv_candidates.csv",
        index=False,
    )
    experiments.fold_balance.to_csv(
        TABLES_DIRECTORY / "issue_7_fold_balance.csv",
        index=False,
    )
    condition_timings = summary.loc[
        :, [
            "model",
            "balanced",
            "candidate_count",
            "search_seconds",
            "search_wall_seconds",
            "final_fit_seconds",
            "condition_seconds",
        ]
    ].copy()
    condition_timings = condition_timings.rename(columns={"condition_seconds": "seconds"})
    condition_timings.insert(0, "stage", "model_condition")
    cache_timings = experiments.timings.copy()
    timings = pd.concat([cache_timings, condition_timings], ignore_index=True, sort=False)
    timings.to_csv(TABLES_DIRECTORY / "issue_7_timings.csv", index=False)

    print("\nResultados de validación cruzada (sin utilizar el test):")
    print(summary.to_string(index=False))
    print(
        "\nFolds compartidos: "
        f"{'StratifiedGroupKFold' if X_train.duplicated().any() else 'StratifiedKFold'}"
        f"(n_splits={shared_cv.n_splits}, "
        f"shuffle={shared_cv.shuffle}, random_state={shared_cv.random_state})"
    )
    print(
        "Preprocesamientos ajustados: "
        f"{experiments.preprocessing_fit_count}; sobremuestreos de folds: "
        f"{experiments.resampling_fit_count}."
    )
    print(f"Tiempo total real: {total_seconds:.2f} s")
    print(f"Tablas: {TABLES_DIRECTORY}")
    print(f"Figura: {FIGURES_DIRECTORY / 'issue_7_class_distribution.png'}")


if __name__ == "__main__":
    main()
