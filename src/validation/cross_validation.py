"""Utilidades de validación cruzada reproducible.

Las particiones que se creen aquí se usan únicamente sobre el conjunto de
entrenamiento que entrega ``stratified_train_test_split``. El conjunto de
prueba nunca debe entregarse a estas funciones durante la selección de un
modelo o de su preprocesamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import N_SPLITS, RANDOM_STATE, TARGET_COLUMN


def make_stratified_kfold(
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Crea el particionador estratificado común para todos los experimentos.

    Con los mismos datos de entrenamiento, ``n_splits`` y ``random_state``,
    instancias separadas producen las mismas cinco particiones. Para comparar
    condiciones, se recomienda materializar una sola vez los índices mediante
    :func:`materialize_stratified_folds` y reutilizarlos.
    """
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits debe ser un entero mayor o igual que 2.")
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def materialize_stratified_folds(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    cv: StratifiedKFold | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Materializa folds para reutilizarlos entre modelos y condiciones.

    ``features`` y ``target`` deben corresponder solo al 70 % de entrenamiento.
    Los arreglos contienen posiciones, por lo que se pueden aplicar con
    ``.iloc`` incluso si el índice original no es consecutivo.
    """
    if len(features) != len(target):
        raise ValueError("Las características y el objetivo deben tener la misma cantidad de filas.")
    if target.empty:
        raise ValueError("No es posible crear folds para un conjunto vacío.")
    if TARGET_COLUMN in features.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' no puede formar parte de las características predictoras."
        )

    splitter = cv if cv is not None else make_stratified_kfold()
    return tuple(
        (train_index, validation_index)
        for train_index, validation_index in splitter.split(features, target)
    )
