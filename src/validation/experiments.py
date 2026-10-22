"""Búsqueda reproducible y eficiente de modelos para el Issue #7.

El módulo recibe exclusivamente ``X_train`` e ``y_train`` de la partición 70 %.
Materializa un único conjunto de folds estratificados y lo reutiliza para las
ocho condiciones experimentales. Para evitar trabajo redundante, cada
preprocesamiento equivalente se ajusta una vez por training fold y se reutiliza
solo entre candidatos que usan esas mismas filas.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from time import perf_counter
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from src.config import RANDOM_STATE, TARGET_COLUMN
from src.data.preprocess import ScalerName
from src.features.feature_selection import selected_feature_names
from src.imbalance import LeakageSafeResamplingPipeline, RandomOverSampler
from src.models.decision_tree import create_decision_tree_pipeline
from src.models.knn import create_knn_pipeline
from src.models.pipeline_utils import create_preprocessing_pipeline
from src.models.random_forest import CustomRandomForest
from src.models.svm import create_svm_pipeline
from src.validation.cross_validation import make_stratified_kfold, materialize_stratified_folds

RECALL_TIE_TOLERANCE = 1e-9
ISSUE7_K_FEATURES = 10

# Las listas de diccionarios representan configuraciones acotadas, no un producto
# cartesiano innecesariamente grande. SVC conserva ambos kernels y dos valores de
# C por kernel; ``gamma`` se especifica únicamente donde RBF lo utiliza.
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
            "classifier__n_estimators": [11],
            "classifier__max_depth": [None],
            "classifier__max_features": ["sqrt"],
        },
        {
            "classifier__n_estimators": [5],
            "classifier__max_depth": [12],
            "classifier__max_features": ["log2"],
        },
        {
            "classifier__n_estimators": [11],
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

# k-NN y SVM comparten el mismo preprocesamiento estándar. Los dos modelos de
# árboles comparten el preprocesamiento sin escalado. Es seguro porque el
# selector recibe exactamente los mismos datos de training en cada clave.
MODEL_SCALERS: dict[str, ScalerName] = {
    "knn": "standard",
    "svm": "standard",
    "decision_tree": "none",
    "custom_random_forest": "none",
}


@dataclass
class Issue7ExperimentResults:
    """Resultados de las ocho optimizaciones sin evaluar el conjunto de prueba."""

    summary: pd.DataFrame
    candidate_results: pd.DataFrame
    fold_balance: pd.DataFrame
    timings: pd.DataFrame
    best_estimators: dict[tuple[str, bool], LeakageSafeResamplingPipeline]
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    preprocessing_fit_count: int
    resampling_fit_count: int


@dataclass
class _PreparedFold:
    fitted_preprocessor: BaseEstimator
    train_features: np.ndarray
    train_target: np.ndarray
    validation_features: np.ndarray
    validation_target: np.ndarray
    selected_features: list[str]


@dataclass
class _PreparedFullTraining:
    fitted_preprocessor: BaseEstimator
    features: np.ndarray
    target: np.ndarray
    selected_features: list[str]


class _PreprocessingCache:
    """Caché local a una búsqueda; nunca ajusta con validación ni test.

    Las claves incluyen tipo de escalador y número de fold. Por ello, no puede
    reutilizar un selector aprendido en un fold distinto. El sampler se memoriza
    solo después de transformar el training fold y con la misma semilla para
    comparaciones justas entre modelos equivalentes.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        folds: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        k_features: int = ISSUE7_K_FEATURES,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.folds = tuple(folds)
        self.k_features = k_features
        self.random_state = random_state
        self._fold_cache: dict[tuple[ScalerName, int], _PreparedFold] = {}
        self._full_cache: dict[ScalerName, _PreparedFullTraining] = {}
        self._resampled_fold_cache: dict[
            tuple[ScalerName, int], tuple[np.ndarray, np.ndarray, dict[Any, int]]
        ] = {}
        self.records: list[dict[str, Any]] = []

    @property
    def preprocessing_fit_count(self) -> int:
        return len(self._fold_cache) + len(self._full_cache)

    @property
    def resampling_fit_count(self) -> int:
        return len(self._resampled_fold_cache)

    def get_fold(self, scaler_kind: ScalerName, fold_number: int) -> _PreparedFold:
        key = (scaler_kind, fold_number)
        if key in self._fold_cache:
            return self._fold_cache[key]

        train_index, validation_index = self.folds[fold_number - 1]
        X_fold_train = self.X_train.iloc[train_index]
        y_fold_train = self.y_train.iloc[train_index]
        X_fold_validation = self.X_train.iloc[validation_index]
        y_fold_validation = self.y_train.iloc[validation_index]
        preprocessor = create_preprocessing_pipeline(
            scaler_kind=scaler_kind,
            k_features=self.k_features,
            random_state=self.random_state,
        )
        started_at = perf_counter()
        fitted_preprocessor = preprocessor.fit(X_fold_train, y_fold_train)
        prepared = _PreparedFold(
            fitted_preprocessor=fitted_preprocessor,
            train_features=fitted_preprocessor.transform(X_fold_train),
            train_target=y_fold_train.to_numpy(copy=True),
            validation_features=fitted_preprocessor.transform(X_fold_validation),
            validation_target=y_fold_validation.to_numpy(copy=True),
            selected_features=_selected_features(
                fitted_preprocessor,
                feature_names=self.X_train.columns,
            ),
        )
        elapsed = perf_counter() - started_at
        self.records.append(
            {
                "stage": "preprocessing",
                "scope": "fold",
                "scaler": scaler_kind,
                "fold": fold_number,
                "balanced": pd.NA,
                "rows_input": len(X_fold_train),
                "rows_output": len(prepared.train_features),
                "seconds": elapsed,
            }
        )
        self._fold_cache[key] = prepared
        return prepared

    def get_full_training(self, scaler_kind: ScalerName) -> _PreparedFullTraining:
        if scaler_kind in self._full_cache:
            return self._full_cache[scaler_kind]

        preprocessor = create_preprocessing_pipeline(
            scaler_kind=scaler_kind,
            k_features=self.k_features,
            random_state=self.random_state,
        )
        started_at = perf_counter()
        fitted_preprocessor = preprocessor.fit(self.X_train, self.y_train)
        prepared = _PreparedFullTraining(
            fitted_preprocessor=fitted_preprocessor,
            features=fitted_preprocessor.transform(self.X_train),
            target=self.y_train.to_numpy(copy=True),
            selected_features=_selected_features(
                fitted_preprocessor,
                feature_names=self.X_train.columns,
            ),
        )
        elapsed = perf_counter() - started_at
        self.records.append(
            {
                "stage": "preprocessing",
                "scope": "full_training",
                "scaler": scaler_kind,
                "fold": pd.NA,
                "balanced": pd.NA,
                "rows_input": len(self.X_train),
                "rows_output": len(prepared.features),
                "seconds": elapsed,
            }
        )
        self._full_cache[scaler_kind] = prepared
        return prepared

    def get_training_data(
        self,
        scaler_kind: ScalerName,
        fold_number: int,
        *,
        balanced: bool,
    ) -> tuple[_PreparedFold, np.ndarray, np.ndarray, dict[Any, int], dict[Any, int]]:
        prepared = self.get_fold(scaler_kind, fold_number)
        before = _class_counts(prepared.train_target)
        if not balanced:
            return prepared, prepared.train_features, prepared.train_target, before, before.copy()

        key = (scaler_kind, fold_number)
        if key not in self._resampled_fold_cache:
            sampler = RandomOverSampler(random_state=self.random_state)
            started_at = perf_counter()
            X_resampled, y_resampled = sampler.fit_resample(
                prepared.train_features,
                prepared.train_target,
            )
            elapsed = perf_counter() - started_at
            after = sampler.class_counts_after_.copy()
            self._resampled_fold_cache[key] = (X_resampled, y_resampled, after)
            self.records.append(
                {
                    "stage": "resampling",
                    "scope": "fold",
                    "scaler": scaler_kind,
                    "fold": fold_number,
                    "balanced": True,
                    "rows_input": len(prepared.train_features),
                    "rows_output": len(X_resampled),
                    "seconds": elapsed,
                }
            )
        X_resampled, y_resampled, after = self._resampled_fold_cache[key]
        return prepared, X_resampled, y_resampled, before, after.copy()


def create_issue7_cv() -> StratifiedKFold:
    """Devuelve el único esquema común: 5 folds, shuffle y semilla 42."""
    return make_stratified_kfold(n_splits=5, random_state=RANDOM_STATE)


def create_experiment_estimator(
    model_key: str,
    *,
    balanced: bool,
    random_state: int = RANDOM_STATE,
    k_features: int = ISSUE7_K_FEATURES,
) -> LeakageSafeResamplingPipeline:
    """Construye uno de los cuatro modelos dentro del pipeline seguro.

    Esta función conserva una API útil fuera de la búsqueda. La búsqueda de
    Issue #7 reutiliza sus componentes ya ajustados mediante
    :meth:`LeakageSafeResamplingPipeline.fit_from_preprocessed`.
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
            # La selección ya ocurrió antes del clasificador. Evitar repetirla
            # en cada árbol también preserva el orden previo al sampler.
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
    """Optimiza las ocho condiciones exclusivamente con el entrenamiento 70 %.

    La selección maximiza recall de ``Class == 1`` y usa precision como
    desempate. El preprocesamiento se ajusta por separado dentro de cada fold,
    y solo se comparte cuando la partición, el escalado y el selector son
    idénticos. El conjunto de prueba no forma parte de esta API.
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

    preprocessing_cache = _PreprocessingCache(X_train, y_train, shared_folds)
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
                preprocessing_cache=preprocessing_cache,
            )
            summaries.append(outcome.summary)
            candidates.extend(outcome.candidate_results)
            balances.extend(outcome.fold_balance)
            best_estimators[(model_key, balanced)] = outcome.estimator

    return Issue7ExperimentResults(
        summary=pd.DataFrame(summaries),
        candidate_results=pd.DataFrame(candidates),
        fold_balance=pd.DataFrame(balances),
        timings=pd.DataFrame(preprocessing_cache.records),
        best_estimators=best_estimators,
        folds=shared_folds,
        preprocessing_fit_count=preprocessing_cache.preprocessing_fit_count,
        resampling_fit_count=preprocessing_cache.resampling_fit_count,
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
    preprocessing_cache: _PreprocessingCache | None = None,
) -> _OptimizationOutcome:
    """Evalúa un espacio, reutilizando únicamente datos matemáticamente iguales."""
    parameter_sets = list(ParameterGrid(parameter_space))
    if not parameter_sets:
        raise ValueError(f"El espacio de búsqueda de {model_label} está vacío.")

    base_estimator = create_experiment_estimator(model_key, balanced=balanced)
    _validate_parameter_sets(model_key, parameter_sets, base_estimator)
    scaler_kind = MODEL_SCALERS[model_key]
    cache = preprocessing_cache or _PreprocessingCache(X_train, y_train, folds)
    candidate_results: list[dict[str, Any]] = []
    fold_records_by_candidate: list[list[dict[str, Any]]] = []
    condition_started_at = perf_counter()
    search_started_at = perf_counter()
    classifier_search_seconds = 0.0

    for candidate_index, parameters in enumerate(parameter_sets):
        classifier_parameters = _classifier_parameters(parameters)
        recall_values: list[float] = []
        precision_values: list[float] = []
        candidate_fold_records: list[dict[str, Any]] = []
        candidate_started_at = perf_counter()
        convergence_warning_count = 0
        candidate_classifier_seconds = 0.0

        for fold_number in range(1, len(folds) + 1):
            prepared, fold_train_X, fold_train_y, before, after = cache.get_training_data(
                scaler_kind,
                fold_number,
                balanced=balanced,
            )
            classifier = clone(base_estimator.classifier).set_params(**classifier_parameters)
            predictions, warning_count, fit_predict_seconds = _fit_and_predict(
                classifier,
                fold_train_X,
                fold_train_y,
                prepared.validation_features,
                model_label=model_label,
                balanced=balanced,
                candidate_index=candidate_index,
                fold_number=fold_number,
            )
            convergence_warning_count += warning_count
            classifier_search_seconds += fit_predict_seconds
            candidate_classifier_seconds += fit_predict_seconds
            recall_values.append(
                recall_score(prepared.validation_target, predictions, pos_label=1, zero_division=0)
            )
            precision_values.append(
                precision_score(prepared.validation_target, predictions, pos_label=1, zero_division=0)
            )
            candidate_fold_records.append(
                {
                    "model": model_label,
                    "balanced": balanced,
                    "fold": fold_number,
                    "class_0_before": before.get(0, 0),
                    "class_1_before": before.get(1, 0),
                    "class_0_after": after.get(0, 0),
                    "class_1_after": after.get(1, 0),
                    "validation_class_0": _class_counts(prepared.validation_target).get(0, 0),
                    "validation_class_1": _class_counts(prepared.validation_target).get(1, 0),
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
                "convergence_warning_count": convergence_warning_count,
                "candidate_seconds": perf_counter() - candidate_started_at,
                "classifier_seconds": candidate_classifier_seconds,
            }
        )
        fold_records_by_candidate.append(candidate_fold_records)

    search_wall_seconds = perf_counter() - search_started_at
    best_index = select_best_configuration(candidate_results)
    best_candidate = candidate_results[best_index]
    best_parameters = parameter_sets[best_index]
    final_prepared = cache.get_full_training(scaler_kind)
    final_estimator = clone(base_estimator).set_params(**best_parameters)
    final_started_at = perf_counter()
    try:
        final_estimator.fit_from_preprocessed(
            fitted_preprocessor=final_prepared.fitted_preprocessor,
            transformed_X=final_prepared.features,
            y=final_prepared.target,
        )
    except Exception as error:  # pragma: no cover - contextualiza errores de librerías externas
        raise RuntimeError(
            f"No se pudo ajustar el mejor candidato de {model_label} "
            f"(balanced={balanced})."
        ) from error
    final_fit_seconds = perf_counter() - final_started_at
    condition_seconds = perf_counter() - condition_started_at

    strategy = "RandomOverSampler" if balanced else "sin compensación"
    summary = {
        "model": model_label,
        "balanced": balanced,
        "balancing_strategy": strategy,
        "best_hyperparameters": _serialise_parameters(best_parameters),
        "recall_cv_mean": best_candidate["recall_cv_mean"],
        "recall_cv_std": best_candidate["recall_cv_std"],
        "precision_cv_mean": best_candidate["precision_cv_mean"],
        "selected_features": json.dumps(final_prepared.selected_features, ensure_ascii=False),
        "class_0_before": final_estimator.class_counts_before_.get(0, 0),
        "class_1_before": final_estimator.class_counts_before_.get(1, 0),
        "class_0_after": final_estimator.class_counts_after_.get(0, 0),
        "class_1_after": final_estimator.class_counts_after_.get(1, 0),
        "candidate_count": len(parameter_sets),
        "convergence_warning_count": best_candidate["convergence_warning_count"],
        "search_seconds": classifier_search_seconds,
        "search_wall_seconds": search_wall_seconds,
        "final_fit_seconds": final_fit_seconds,
        "condition_seconds": condition_seconds,
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


def _fit_and_predict(
    classifier: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    *,
    model_label: str,
    balanced: bool,
    candidate_index: int,
    fold_number: int,
) -> tuple[np.ndarray, int, float]:
    """Entrena preservando advertencias de convergencia y contexto del error."""
    try:
        started_at = perf_counter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(X_train, y_train)
        convergence_warnings = sum(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught_warnings
        )
        predictions = classifier.predict(X_validation)
        return predictions, convergence_warnings, perf_counter() - started_at
    except Exception as error:
        raise RuntimeError(
            f"Falló {model_label} (balanced={balanced}, candidato={candidate_index}, "
            f"fold={fold_number})."
        ) from error


def _selected_features(
    preprocessor: BaseEstimator,
    *,
    feature_names: Sequence[str] | None = None,
) -> list[str]:
    selector = getattr(preprocessor, "named_steps", {}).get("select")
    if selector in (None, "passthrough"):
        return []
    return selected_feature_names(selector, feature_names)


def _class_counts(y: np.ndarray) -> dict[Any, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {
        label.item() if isinstance(label, np.generic) else label: int(count)
        for label, count in zip(labels, counts)
    }


def _classifier_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "classifier__"
    unsupported = [key for key in parameters if not key.startswith(prefix)]
    if unsupported:
        formatted = ", ".join(sorted(unsupported))
        raise ValueError(f"Los parámetros de búsqueda deben comenzar con '{prefix}': {formatted}.")
    return {key.removeprefix(prefix): value for key, value in parameters.items()}


def _validate_parameter_sets(
    model_key: str,
    parameter_sets: Sequence[Mapping[str, Any]],
    base_estimator: LeakageSafeResamplingPipeline,
) -> None:
    if model_key != "custom_random_forest":
        return
    default_estimators = base_estimator.classifier.n_estimators
    for parameters in parameter_sets:
        n_estimators = parameters.get("classifier__n_estimators", default_estimators)
        if not isinstance(n_estimators, int) or n_estimators % 2 == 0:
            raise ValueError(
                "La búsqueda de CustomRandomForest debe usar n_estimators entero e impar."
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
        if (
            train_index.min() < 0
            or validation_index.min() < 0
            or train_index.max() >= sample_count
            or validation_index.max() >= sample_count
        ):
            raise ValueError("Los índices de los folds exceden X_train.")
        if np.intersect1d(train_index, validation_index).size:
            raise ValueError("Los conjuntos de entrenamiento y validación de un fold se solapan.")
