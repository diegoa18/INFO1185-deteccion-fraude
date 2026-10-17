"""Ejecuta los experimentos reproducibles de desbalance y optimización.

Ejecute desde la raíz del repositorio:

    .venv/bin/python scripts/run_issue_7.py

La ejecución puede tardar, especialmente por las configuraciones SVM y el
Random Forest propio. No evalúa ni selecciona modelos con el conjunto de prueba:
solo escribe resultados de validación cruzada sobre el 70 % de entrenamiento.
"""

from __future__ import annotations

from pathlib import Path

from src.config import RANDOM_STATE
from src.data.load_data import load_dataset, separate_features_target
from src.data.preprocess import stratified_train_test_split
from src.imbalance import analyze_class_imbalance, plot_class_imbalance
from src.validation import create_issue7_cv, materialize_stratified_folds, run_issue7_experiments

RESULTS_DIRECTORY = Path("results")
TABLES_DIRECTORY = RESULTS_DIRECTORY / "tables"
FIGURES_DIRECTORY = RESULTS_DIRECTORY / "figures"


def main() -> None:
    dataframe = load_dataset()
    X, y = separate_features_target(dataframe)

    imbalance = analyze_class_imbalance(y)
    print("Desbalance del dataset completo (Class == 1 es fraude):")
    print(imbalance.to_string(index=False, formatters={"percentage": "{:.6f}".format}))
    print(
        "Razón mayoría/minoría: "
        f"{imbalance['majority_to_minority_ratio'].iloc[0]:.2f}:1\n"
    )
    TABLES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    imbalance.to_csv(TABLES_DIRECTORY / "issue_7_class_imbalance.csv", index=False)
    figure, _ = plot_class_imbalance(
        imbalance,
        FIGURES_DIRECTORY / "issue_7_class_distribution.png",
    )
    figure.clear()

    # La única división oficial: 70 % entrenamiento y 30 % prueba. X_test e
    # y_test permanecen deliberadamente fuera de los experimentos de abajo.
    X_train, _X_test, y_train, _y_test = stratified_train_test_split(
        X,
        y,
        random_state=RANDOM_STATE,
    )
    shared_cv = create_issue7_cv()
    shared_folds = materialize_stratified_folds(X_train, y_train, cv=shared_cv)
    experiments = run_issue7_experiments(
        X_train,
        y_train,
        folds=shared_folds,
    )

    experiments.summary.to_csv(TABLES_DIRECTORY / "issue_7_cv_results.csv", index=False)
    experiments.candidate_results.to_csv(
        TABLES_DIRECTORY / "issue_7_cv_candidates.csv",
        index=False,
    )
    experiments.fold_balance.to_csv(
        TABLES_DIRECTORY / "issue_7_fold_balance.csv",
        index=False,
    )
    print("Resultados de validación cruzada (sin utilizar el test):")
    print(experiments.summary.to_string(index=False))
    print(
        "\nFolds compartidos: "
        f"StratifiedKFold(n_splits={shared_cv.n_splits}, "
        f"shuffle={shared_cv.shuffle}, random_state={shared_cv.random_state})"
    )
    print(f"Tablas: {TABLES_DIRECTORY}")
    print(f"Figura: {FIGURES_DIRECTORY / 'issue_7_class_distribution.png'}")


if __name__ == "__main__":
    main()
