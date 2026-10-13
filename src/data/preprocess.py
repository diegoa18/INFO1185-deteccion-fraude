"""División y transformadores de preprocesamiento sin fuga de información."""

from __future__ import annotations

from typing import Literal

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
    """Crea una división reproducible 70/30 y estratificada.

    No ajusta transformadores: cualquier escalador o selector debe ajustarse después,
    dentro de un pipeline entrenado solo con cada partición de entrenamiento.
    """
    _validate_features_target(features, target)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
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
