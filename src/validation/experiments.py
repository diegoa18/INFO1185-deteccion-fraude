"""Búsqueda reproducible de modelos para el Issue #7.

El módulo recibe exclusivamente ``X_train`` e ``y_train`` de la partición 70 %.
Materializa un único conjunto de folds estratificados y lo reutiliza para las ocho
condiciones experimentales. El conjunto de prueba no forma parte de esta API.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from src.config import RANDOM_STATE, TARGET_COLUMN
from src.imbalance import LeakageSafeResamplingPipeline, RandomOverSampler
from src.models.decision_tree import create_decision_tree_pipeline
from src.models.knn import create_knn_pipeline
from src.models.pipeline_utils import create_preprocessing_pipeline
from src.models.random_forest import CustomRandomForest
from src.models.svm import create_svm_pipeline
from src.validation.cross_validation import make_stratified_kfold, materialize_stratified_folds

RECALL_TIE_TOLERANCE = 1e-9

# Las listas de diccionarios representan configuraciones acotadas, no un producto
# cartesiano innecesariamente grande. Cada espacio cubre los hiperparámetros
# requeridos y mantiene viable la ejecución sobre las 284 807 transacciones.
DEFAULT_SEARCH_SPACES: dict[str, list[dict[str, list[Any]]]] = {
    "knn": [
        {
            "classifier__n_neighbors": [3],
            "classifier__weights": ["uniform"],
            "classifier__metric": ["euclidean"],
        },
        {
            "classifier__n_neighbors": [5],
            "classifier__weights": ["distance"],
            "classifier__metric": ["euclidean"],
        },
        {
            "classifier__n_neighbors": [7],
            "classifier__weights": ["uniform"],
            "classifier__metric": ["manhattan"],
        },
        {
            "classifier__n_neighbors": [11],
            "classifier__weights": ["distance"],
            "classifier__metric": ["manhattan"],
        },
    ],
    "svm": [
        {
            "classifier__kernel": ["linear"],
            "classifier__C": [0.5],
        },
        {
            "classifier__kernel": ["linear"],
            "classifier__C": [2.0],
        },
        {
            "classifier__kernel": ["rbf"],
            "classifier__C": [0.5],
            "classifier__gamma": ["scale"],
        },
        {
            "classifier__kernel": ["rbf"],
            "classifier__C": [2.0],
            "classifier__gamma": [0.1],
        },
    ],
    "decision_tree": [
        {
            "classifier__max_depth": [None],
            "classifier__min_samples_split": [2],
            "classifier__min_samples_leaf": [1],
            "classifier__criterion": ["gini"],
        },
        {
            "classifier__max_depth": [12],
            "classifier__min_samples_split": [2],
            "classifier__min_samples_leaf": [1],
            "classifier__criterion": ["entropy"],
        },
        {
            "classifier__max_depth": [None],
            "classifier__min_samples_split": [20],
            "classifier__min_samples_leaf": [5],
            "classifier__criterion": ["gini"],
        },
        {
            "classifier__max_depth": [12],
            "classifier__min_samples_split": [20],
            "classifier__min_samples_leaf": [5],
            "classifier__criterion": ["entropy"],
        },
    ],
    "custom_random_forest": [
        {
            "classifier__n_estimators": [5],
            "classifier__max_depth": [None],
            "classifier__max_features": ["sqrt"],
        },
        {
            "classifier__n_estimators": [10],
            "classifier__max_depth": [None],
            "classifier__max_features": ["sqrt"],
        },
        {
            "classifier__n_estimators": [5],
            "classifier__max_depth": [12],
            "classifier__max_features": ["log2"],
        },
        {
            "classifier__n_estimators": [10],
            "classifier__max_depth": [12],
            "classifier__max_features": ["log2"],
        },
    ],
}

MODEL_LABELS = {
    "knn": "k-NN",
    "svm": "SVM",
    "decision_tree": "Árbol de Decisión",
    "custom_random_forest": "Random Forest propio",
}


@dataclass
class Issue7ExperimentResults:
    """Resultados de las ocho optimizaciones sin evaluar el conjunto de prueba."""

    summary: pd.DataFrame
    candidate_results: pd.DataFrame
    fold_balance: pd.DataFrame
    best_estimators: dict[tuple[str, bool], LeakageSafeResamplingPipeline]
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]


def create_issue7_cv() -> StratifiedKFold:
    """Devuelve el único esquema común: 5 folds, shuffle y semilla 42."""
    return make_stratified_kfold(n_splits=5, random_state=RANDOM_STATE)


def create_experiment_estimator(
    model_key: str,
    *,
    balanced: bool,
    random_state: int = RANDOM_STATE,
    k_features: int = 10,
) -> LeakageSafeResamplingPipeline:
    """Construye uno de los cuatro modelos dentro del pipeline seguro.

    k-NN y SVM conservan su escalado estándar; los modelos de árboles usan
    ``passthrough``. La selección de las 10 variables con mayor información
    mutua se ajusta antes del muestreador en cada fold; no se aprende sobre todo
    el entrenamiento ni sobre test. El valor se puede cambiar para experimentos
    posteriores, pero no se mezcla con esta búsqueda para mantenerla acotada.
    """
    if model_key == "knn":
        base_pipeline = create_knn_pipeline(k_features=k_features, random_state=random_state)
        preprocessor, classifier = _split_model_pipeline(base_pipeline)
    elif model_key == "svm":
        base_pipeline = create_svm_pipeline(k_features=k_features, random_state=random_state)
        preprocessor, classifier = _split_model_pipeline(base_pipeline)
    elif model_key == "decision_tree":
        base_pipeline = create_decision_tree_pipeline(k_features=k_features, random_state=random_state)
        preprocessor, classifier = _split_model_pipeline(base_pipeline)
    elif model_key == "custom_random_forest":
        preprocessor = create_preprocessing_pipeline(
            scaler_kind="none",
            k_features=k_features,
            random_state=random_state,
        )
        classifier = CustomRandomForest(
            n_estimators=5,
            scaler_kind="none",
            # La selección ya ocurrió en el preprocessor exterior, antes del
            # sobremuestreo. Evitar repetirla en cada árbol también impide que
            # un transformador se ajuste después del sampler.
            k_features=None,
            random_state=random_state,
        )
    else:
        supported = ", ".join(MODEL_LABELS)
        raise ValueError(f"model_key debe ser uno de: {supported}.")

    sampler = RandomOverSampler(random_state=random_state) if balanced else None
    return LeakageSafeResamplingPipeline(
        preprocessor=preprocessor,
        classifier=classifier,
        sampler=sampler,
    )


def run_issue7_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    search_spaces: Mapping[str, list[dict[str, list[Any]]]] | None = None,
) -> Issue7ExperimentResults:
    """Optimiza las ocho condiciones utilizando solo el entrenamiento 70 %.

    La selección se hace por mayor recall medio de ``Class == 1``. Las
    configuraciones cuyo recall es igual dentro de ``1e-9`` se desempatan por
    mayor precision media de fraude. El estimador ganador se vuelve a ajustar
    sobre todo ``X_train`` al final, para quedar disponible para una evaluación
    final posterior y separada sobre ``X_test``.
    """
    _validate_training_data(X_train, y_train)
    shared_folds = tuple(folds) if folds is not None else materialize_stratified_folds(
        X_train,
        y_train,
        cv=create_issue7_cv(),
    )
    _validate_folds(shared_folds, len(X_train))
    spaces = search_spaces if search_spaces is not None else DEFAULT_SEARCH_SPACES
    missing_spaces = set(MODEL_LABELS).difference(spaces)
    if missing_spaces:
        raise ValueError(f"Faltan espacios de búsqueda para: {', '.join(sorted(missing_spaces))}.")

    summaries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    balances: list[dict[str, Any]] = []
    best_estimators: dict[tuple[str, bool], LeakageSafeResamplingPipeline] = {}

    for model_key, model_label in MODEL_LABELS.items():
        for balanced in (False, True):
            outcome = optimize_model(
                model_key=model_key,
                model_label=model_label,
                balanced=balanced,
                X_train=X_train,
                y_train=y_train,
                folds=shared_folds,
                parameter_space=spaces[model_key],
            )
            summaries.append(outcome.summary)
            candidates.extend(outcome.candidate_results)
            balances.extend(outcome.fold_balance)
            best_estimators[(model_key, balanced)] = outcome.estimator

    return Issue7ExperimentResults(
        summary=pd.DataFrame(summaries),
        candidate_results=pd.DataFrame(candidates),
        fold_balance=pd.DataFrame(balances),
        best_estimators=best_estimators,
        folds=shared_folds,
    )


@dataclass
class _OptimizationOutcome:
    summary: dict[str, Any]
    candidate_results: list[dict[str, Any]]
    fold_balance: list[dict[str, Any]]
    estimator: LeakageSafeResamplingPipeline


def optimize_model(
    *,
    model_key: str,
    model_label: str,
    balanced: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    parameter_space: list[dict[str, list[Any]]],
) -> _OptimizationOutcome:
    """Evalúa un espacio con los folds compartidos y escoge el mejor candidato."""
    parameter_sets = list(ParameterGrid(parameter_space))
    if not parameter_sets:
        raise ValueError(f"El espacio de búsqueda de {model_label} está vacío.")

    candidate_results: list[dict[str, Any]] = []
    fold_records_by_candidate: list[list[dict[str, Any]]] = []
    base_estimator = create_experiment_estimator(model_key, balanced=balanced)

    for candidate_index, parameters in enumerate(parameter_sets):
        recall_values: list[float] = []
        precision_values: list[float] = []
        candidate_fold_records: list[dict[str, Any]] = []
        for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
            estimator = clone(base_estimator)
            estimator.set_params(**parameters)
            estimator.fit(X_train.iloc[train_index], y_train.iloc[train_index])
            predictions = estimator.predict(X_train.iloc[validation_index])
            validation_target = y_train.iloc[validation_index]
            recall_values.append(recall_score(validation_target, predictions, pos_label=1, zero_division=0))
            precision_values.append(precision_score(validation_target, predictions, pos_label=1, zero_division=0))
            candidate_fold_records.append(
                {
                    "model": model_label,
                    "balanced": balanced,
                    "fold": fold_number,
                    "class_0_before": estimator.class_counts_before_.get(0, 0),
                    "class_1_before": estimator.class_counts_before_.get(1, 0),
                    "class_0_after": estimator.class_counts_after_.get(0, 0),
                    "class_1_after": estimator.class_counts_after_.get(1, 0),
                }
            )

        candidate_results.append(
            {
                "model": model_label,
                "balanced": balanced,
                "candidate_index": candidate_index,
                "hyperparameters": _serialise_parameters(parameters),
                "recall_cv_mean": float(np.mean(recall_values)),
                "recall_cv_std": float(np.std(recall_values, ddof=0)),
                "precision_cv_mean": float(np.mean(precision_values)),
            }
        )
        fold_records_by_candidate.append(candidate_fold_records)

    best_index = select_best_configuration(candidate_results)
    best_candidate = candidate_results[best_index]
    best_parameters = parameter_sets[best_index]
    final_estimator = clone(base_estimator).set_params(**best_parameters)
    # Este fit usa únicamente el 70 % de entrenamiento. X_test no se entrega a
    # la búsqueda, al balanceador ni al ajuste final del modelo seleccionado.
    final_estimator.fit(X_train, y_train)

    strategy = "RandomOverSampler" if balanced else "sin compensación"
    summary = {
        "model": model_label,
        "balanced": balanced,
        "balancing_strategy": strategy,
        "best_hyperparameters": _serialise_parameters(best_parameters),
        "recall_cv_mean": best_candidate["recall_cv_mean"],
        "recall_cv_std": best_candidate["recall_cv_std"],
        "precision_cv_mean": best_candidate["precision_cv_mean"],
        "class_0_before": final_estimator.class_counts_before_.get(0, 0),
        "class_1_before": final_estimator.class_counts_before_.get(1, 0),
        "class_0_after": final_estimator.class_counts_after_.get(0, 0),
        "class_1_after": final_estimator.class_counts_after_.get(1, 0),
    }
    return _OptimizationOutcome(
        summary=summary,
        candidate_results=candidate_results,
        fold_balance=fold_records_by_candidate[best_index],
        estimator=final_estimator,
    )


def select_best_configuration(candidate_results: Sequence[Mapping[str, Any]]) -> int:
    """Prioriza recall y usa precision como desempate explícito."""
    if not candidate_results:
        raise ValueError("No hay configuraciones para seleccionar.")
    recalls = np.asarray([row["recall_cv_mean"] for row in candidate_results], dtype=float)
    highest_recall = recalls.max()
    tied_indices = np.flatnonzero(
        np.isclose(recalls, highest_recall, rtol=0.0, atol=RECALL_TIE_TOLERANCE)
    )
    # max conserva el primer índice si también hay empate en precision, por lo
    # que el resultado es determinista y sigue el orden declarado del espacio.
    return int(
        max(
            tied_indices,
            key=lambda index: candidate_results[int(index)]["precision_cv_mean"],
        )
    )


def _split_model_pipeline(pipeline):
    """Separa pasos de transformación y clasificador de un pipeline existente."""
    if len(pipeline.steps) < 2 or pipeline.steps[-1][0] != "classifier":
        raise ValueError("El pipeline debe terminar con un paso llamado 'classifier'.")
    from sklearn.pipeline import Pipeline

    return Pipeline(pipeline.steps[:-1]), pipeline.steps[-1][1]


def _serialise_parameters(parameters: Mapping[str, Any]) -> str:
    readable = {
        key.removeprefix("classifier__"): value
        for key, value in parameters.items()
    }
    return json.dumps(readable, ensure_ascii=False, sort_keys=True, default=str)


def _validate_training_data(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    if TARGET_COLUMN in X_train.columns:
        raise ValueError(f"'{TARGET_COLUMN}' no puede formar parte de las características predictoras.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train e y_train deben tener la misma cantidad de filas.")
    if X_train.empty:
        raise ValueError("No es posible optimizar con un conjunto vacío.")
    observed_classes = set(pd.Series(y_train).unique())
    if not {0, 1}.issubset(observed_classes):
        raise ValueError("El entrenamiento debe contener Class == 0 y Class == 1.")


def _validate_folds(
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    sample_count: int,
) -> None:
    if len(folds) != 5:
        raise ValueError("Issue #7 requiere exactamente cinco folds estratificados.")
    for train_index, validation_index in folds:
        if not len(train_index) or not len(validation_index):
            raise ValueError("Los folds no pueden tener subconjuntos vacíos.")
        if train_index.max() >= sample_count or validation_index.max() >= sample_count:
            raise ValueError("Los índices de los folds exceden X_train.")
        if np.intersect1d(train_index, validation_index).size:
            raise ValueError("Los conjuntos de entrenamiento y validación de un fold se solapan.")
