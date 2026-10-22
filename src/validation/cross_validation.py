"""Utilidades de validación cruzada reproducible.

Las particiones que se creen aquí se usan únicamente sobre el conjunto de
entrenamiento que entrega ``stratified_train_test_split``. El conjunto de
prueba nunca debe entregarse a estas funciones durante la selección de un
modelo o de su preprocesamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

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
    cv: StratifiedKFold | StratifiedGroupKFold | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Materializa folds para reutilizarlos entre modelos y condiciones.

    ``features`` y ``target`` deben corresponder solo al 70 % de entrenamiento.
    Los arreglos contienen posiciones, por lo que se pueden aplicar con
    ``.iloc`` incluso si el índice original no es consecutivo.

    Con duplicados se adapta StratifiedKFold a StratifiedGroupKFold conservando
    n_splits, shuffle y random_state. Sin duplicados se conserva StratifiedKFold.
    También se acepta StratifiedGroupKFold explícito; otros tipos se rechazan
    para no descartar silenciosamente una estrategia personalizada. Los grupos
    dependen solo de los predictores, incluso si sus etiquetas son distintas.
    La estratificación por grupos es aproximada; se rechazan folds sin ambas
    clases en vez de separar copias para completar tamaños o proporciones.
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
    if type(splitter) not in (StratifiedKFold, StratifiedGroupKFold):
        raise TypeError("cv debe ser StratifiedKFold o StratifiedGroupKFold.")
    # Factorización de tuplas completas: no incluye índice ni depende de que
    # un hash de fila sea único para definir igualdad de predictores.
    groups, unique_groups = pd.factorize(pd.MultiIndex.from_frame(features), sort=False)
    if set(target.unique()) != {0, 1}:
        raise ValueError("La validación cruzada requiere ambas clases: 0 y 1.")
    for label in (0, 1):
        count = len(np.unique(groups[target.to_numpy() == label]))
        if count < splitter.n_splits:
            raise ValueError(
                f"No es posible crear {splitter.n_splits} folds válidos: "
                f"la clase {label} solo aparece en {count} grupos de predictores."
            )
    if len(unique_groups) < len(features) and type(splitter) is StratifiedKFold:
        splitter = StratifiedGroupKFold(
            n_splits=splitter.n_splits,
            shuffle=splitter.shuffle,
            random_state=splitter.random_state,
        )
    splits = (
        splitter.split(features, target, groups)
        if isinstance(splitter, StratifiedGroupKFold)
        else splitter.split(features, target)
    )
    folds = tuple(
        (train_index, validation_index)
        for train_index, validation_index in splits
    )
    for number, (train, validation) in enumerate(folds, start=1):
        if (set(target.iloc[train]) != {0, 1}
                or set(target.iloc[validation]) != {0, 1}):
            raise ValueError(
                f"No se pudieron generar {splitter.n_splits} folds válidos: "
                f"el fold {number} no contiene ambas clases en entrenamiento y "
                "validación. La estratificación por grupos es aproximada."
            )
        if np.intersect1d(groups[train], groups[validation]).size:
            raise ValueError(f"El fold {number} contiene grupos cruzados.")
    return folds
