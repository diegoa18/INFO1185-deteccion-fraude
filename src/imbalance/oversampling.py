"""Sobremuestreo aleatorio reproducible sin dependencia externa.

El proyecto no depende de ``imbalanced-learn``. ``RandomOverSampler`` ofrece la
parte necesaria de su API para poder usar una alternativa equivalente: duplica
aleatoriamente ejemplos minoritarios *solo* al entrenar un estimador.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.config import RANDOM_STATE


class RandomOverSampler(BaseEstimator):
    """Iguala cada clase a la mayoritaria mediante muestreo con reemplazo.

    La instancia no modifica los datos de entrada. ``fit_resample`` se invoca
    desde :class:`~src.imbalance.pipeline.LeakageSafeResamplingPipeline`, después
    de ajustar el preprocesamiento del fold de entrenamiento y nunca sobre el
    fold de validación o el conjunto de prueba.
    """

    def __init__(self, *, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state

    @staticmethod
    def _as_1d_array(y: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(y)
        if values.ndim != 1:
            raise ValueError("y debe ser un arreglo unidimensional.")
        if len(values) == 0:
            raise ValueError("No es posible balancear un conjunto vacío.")
        return values

    def fit_resample(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve una copia balanceada de ``X`` e ``y``.

        Se preservan todas las observaciones originales y se añaden duplicados
        de cada clase minoritaria hasta alcanzar la cantidad de la clase con más
        muestras. La permutación final evita que las clases queden agrupadas.
        """
        X_values = X.to_numpy(copy=True) if isinstance(X, pd.DataFrame) else np.asarray(X).copy()
        y_values = self._as_1d_array(y).copy()
        if len(X_values) != len(y_values):
            raise ValueError("X e y deben tener la misma cantidad de filas.")

        labels, counts = np.unique(y_values, return_counts=True)
        if len(labels) < 2:
            raise ValueError("Se requieren al menos dos clases para sobremuestrear.")

        self.class_counts_before_ = {
            _as_python_scalar(label): int(count)
            for label, count in zip(labels, counts)
        }
        majority_count = int(counts.max())
        rng = np.random.default_rng(self.random_state)
        sampled_indices = [np.arange(len(y_values), dtype=int)]

        for label, count in zip(labels, counts):
            missing = majority_count - int(count)
            if missing:
                class_indices = np.flatnonzero(y_values == label)
                sampled_indices.append(rng.choice(class_indices, size=missing, replace=True))

        combined_indices = np.concatenate(sampled_indices)
        self.sample_indices_ = rng.permutation(combined_indices)
        X_resampled = X_values[self.sample_indices_]
        y_resampled = y_values[self.sample_indices_]
        resampled_labels, resampled_counts = np.unique(y_resampled, return_counts=True)
        self.class_counts_after_ = {
            _as_python_scalar(label): int(count)
            for label, count in zip(resampled_labels, resampled_counts)
        }
        return X_resampled, y_resampled


def _as_python_scalar(value: Hashable | np.generic) -> Hashable:
    """Convierte escalares NumPy para serializarlos y mostrarlos con claridad."""
    return value.item() if isinstance(value, np.generic) else value
