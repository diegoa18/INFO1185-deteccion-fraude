"""División y transformadores de preprocesamiento sin fuga de información."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE

ScalerName = Literal["standard", "minmax", "none"]


def _validate_features_target(features: pd.DataFrame, target: pd.Series) -> None:
    if TARGET_COLUMN in features.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' no puede formar parte de las características predictoras."
        )
    if len(features) != len(target):
        raise ValueError("Las características y el objetivo deben tener la misma cantidad de filas.")
    if target.empty:
        raise ValueError("No es posible dividir un conjunto de datos vacío.")


def stratified_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Crea una división reproducible 70/30, estratificada y sin duplicados cruzados.

    No ajusta transformadores: cualquier escalador o selector debe ajustarse después,
    dentro de un pipeline entrenado solo con cada partición de entrenamiento.

    Los registros con predictores idénticos se mantienen juntos. Así se conserva la
    información original (no se eliminan duplicados automáticamente) y se evita que
    una copia exacta aparezca tanto en entrenamiento como en prueba. La proporción
    por clase queda lo más cercana posible a ``test_size``; con grupos de más de una
    fila podría diferir por unas pocas observaciones.
    """
    _validate_features_target(features, target)
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    group_hashes = pd.util.hash_pandas_object(features, index=False)
    grouping = pd.DataFrame(
        {
            "position": np.arange(len(features)),
            "group": group_hashes.to_numpy(),
            "target": target.to_numpy(),
        }
    )
    conflicting_groups = grouping.groupby("group")["target"].nunique()
    if (conflicting_groups > 1).any():
        raise ValueError(
            "Existen predictores idénticos asociados a clases distintas; "
            "revise esos registros antes de dividir."
        )

    random_generator = np.random.default_rng(random_state)
    test_positions: list[int] = []
    for _, class_rows in grouping.groupby("target", sort=True):
        groups = [
            group["position"].to_numpy()
            for _, group in class_rows.groupby("group", sort=False)
        ]
        random_generator.shuffle(groups)
        target_test_count = round(len(class_rows) * test_size)
        selected_count = 0
        for group_positions in groups:
            candidate_count = selected_count + len(group_positions)
            if abs(candidate_count - target_test_count) <= abs(selected_count - target_test_count):
                test_positions.extend(group_positions.tolist())
                selected_count = candidate_count

    test_position_array = np.asarray(sorted(test_positions), dtype=int)
    train_mask = np.ones(len(features), dtype=bool)
    train_mask[test_position_array] = False
    train_position_array = np.flatnonzero(train_mask)
    return (
        features.iloc[train_position_array].copy(),
        features.iloc[test_position_array].copy(),
        target.iloc[train_position_array].copy(),
        target.iloc[test_position_array].copy(),
    )


def stratified_sample(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    sample_size: int,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extrae un subconjunto reproducible sin alterar la proporción de clases.

    El muestreo se realiza antes de la división oficial 70/30. No ajusta ni
    inspecciona transformadores, por lo que el escalador, selector y
    sobremuestreador siguen aprendiendo exclusivamente del training fold. El
    resultado conserva los índices originales para poder auditar que los
    conjuntos posteriores no se solapan.
    """
    _validate_features_target(features, target)
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise ValueError("sample_size debe ser un entero positivo.")
    if sample_size <= 0:
        raise ValueError("sample_size debe ser un entero positivo.")
    if sample_size > len(features):
        raise ValueError("sample_size no puede superar la cantidad de registros disponible.")
    if sample_size == len(features):
        return features.copy(), target.copy()

    class_counts = target.value_counts()
    if len(class_counts) < 2:
        raise ValueError("El muestreo estratificado requiere al menos dos clases.")
    if sample_size < len(class_counts):
        raise ValueError("sample_size es demasiado pequeño para conservar todas las clases.")

    positions = np.arange(len(features))
    selected_positions, _ = train_test_split(
        positions,
        train_size=sample_size,
        stratify=target.to_numpy(),
        random_state=random_state,
    )
    return (
        features.iloc[selected_positions].copy(),
        target.iloc[selected_positions].copy(),
    )


def create_scaler(kind: ScalerName = "standard") -> StandardScaler | MinMaxScaler | str:
    """Construye un escalador nuevo compatible con ``sklearn.pipeline.Pipeline``.

    ``"none"`` devuelve ``"passthrough"`` para modelos basados en árboles. El
    escalador resultante no se ajusta aquí; Pipeline lo ajustará solamente sobre sus
    datos de entrenamiento.
    """
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "none":
        return "passthrough"
    raise ValueError("kind debe ser 'standard', 'minmax' o 'none'.")


def clean_dataset(dataframe: pd.DataFrame, *, remove_exact_duplicates: bool = False) -> pd.DataFrame:
    """Devuelve una copia y elimina duplicados solo bajo decisión explícita.

    El conjunto oficial no se limpia automáticamente: quitar duplicados podría alterar
    la clase minoritaria. Antes de activar ``remove_exact_duplicates``, inspeccione el
    impacto por clase con ``diagnose_dataset``. Los valores faltantes tampoco se imputan
    aquí, pues su tratamiento debe definirse y ajustarse dentro del pipeline futuro.
    """
    cleaned = dataframe.copy()
    if remove_exact_duplicates:
        cleaned = cleaned.drop_duplicates(ignore_index=False).copy()
    return cleaned
