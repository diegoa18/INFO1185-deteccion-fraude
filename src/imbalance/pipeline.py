"""Pipeline que encapsula preprocesamiento, balanceo y clasificador.

``sklearn.pipeline.Pipeline`` no admite pasos que cambien simultáneamente ``X`` e
``y``. Esta implementación equivalente mantiene el orden seguro requerido por el
proyecto: preprocesamiento y selección aprendidos en el training fold, balanceo
solo de dicho resultado y ajuste del clasificador.
"""

from __future__ import annotations

from collections.abc import Hashable
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


class LeakageSafeResamplingPipeline(ClassifierMixin, BaseEstimator):
    """Combina preprocesador, sobremuestreador opcional y clasificador.

    Parameters are regular nested scikit-learn parameters, so callers can set
    ``classifier__C`` or ``preprocessor__select__k`` during a model search. The
    sampler is cloned and fitted anew in every ``fit`` call, which prevents any
    sample generated from a validation fold from reaching a training fold.
    """

    def __init__(
        self,
        *,
        preprocessor: BaseEstimator,
        classifier: BaseEstimator,
        sampler: BaseEstimator | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.sampler = sampler

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "LeakageSafeResamplingPipeline":
        """Ajusta pasos con los datos de entrenamiento recibidos únicamente."""
        y_values = np.asarray(y)
        if y_values.ndim != 1:
            raise ValueError("y debe ser un arreglo unidimensional.")
        if len(X) != len(y_values):
            raise ValueError("X e y deben tener la misma cantidad de filas.")
        if len(y_values) == 0:
            raise ValueError("No es posible entrenar con un conjunto vacío.")

        fitted_preprocessor = clone(self.preprocessor).fit(X, y_values)
        transformed_X = fitted_preprocessor.transform(X)
        return self.fit_from_preprocessed(
            fitted_preprocessor=fitted_preprocessor,
            transformed_X=transformed_X,
            y=y_values,
        )

    def fit_from_preprocessed(
        self,
        *,
        fitted_preprocessor: BaseEstimator,
        transformed_X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "LeakageSafeResamplingPipeline":
        """Ajusta el clasificador a una transformación ya aprendida de forma segura.

        Esta variante permite que una búsqueda reutilice el mismo escalador y
        selector para candidatos que comparten exactamente el mismo training
        fold. El llamador debe haber ajustado ``fitted_preprocessor`` y obtenido
        ``transformed_X`` *solo* a partir de las filas de entrenamiento de esa
        llamada. Se realiza una copia profunda del preprocesador ajustado para
        que el estimador resultante sea independiente de la caché.
        """
        y_values = np.asarray(y)
        if y_values.ndim != 1:
            raise ValueError("y debe ser un arreglo unidimensional.")
        if len(transformed_X) != len(y_values):
            raise ValueError("transformed_X e y deben tener la misma cantidad de filas.")
        if len(y_values) == 0:
            raise ValueError("No es posible entrenar con un conjunto vacío.")

        self.preprocessor_ = deepcopy(fitted_preprocessor)
        self.classifier_ = clone(self.classifier)
        self.class_counts_before_ = _class_counts(y_values)

        if self.sampler is None:
            self.sampler_ = None
            training_X, training_y = transformed_X, y_values
            self.class_counts_after_ = self.class_counts_before_.copy()
        else:
            self.sampler_ = clone(self.sampler)
            training_X, training_y = self.sampler_.fit_resample(transformed_X, y_values)
            self.class_counts_after_ = self.sampler_.class_counts_after_.copy()

        self.classifier_.fit(training_X, training_y)
        self.classes_ = np.unique(y_values)
        self.n_features_in_ = getattr(self.preprocessor_, "n_features_in_", None)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predice sin modificar ni balancear ``X``."""
        check_is_fitted(self, ["preprocessor_", "classifier_", "classes_"])
        transformed_X = self.preprocessor_.transform(X)
        return self.classifier_.predict(transformed_X)


def _class_counts(y: np.ndarray) -> dict[Hashable, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {
        (label.item() if isinstance(label, np.generic) else label): int(count)
        for label, count in zip(labels, counts)
    }
