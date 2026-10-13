"""Funciones para cargar y describir el conjunto Credit Card Fraud Detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DEFAULT_DATASET_PATH, PREDICTOR_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN


class DataFileNotFoundError(FileNotFoundError):
    """El CSV del conjunto de datos no se encuentra en la ruta indicada."""


class DatasetValidationError(ValueError):
    """El CSV no respeta el esquema esperado para este proyecto."""


def validate_dataset_columns(dataframe: pd.DataFrame) -> None:
    """Comprueba que el dataframe tenga todas las columnas del conjunto oficial."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        formatted_columns = ", ".join(missing_columns)
        raise DatasetValidationError(
            "El dataset no contiene las columnas requeridas: " f"{formatted_columns}."
        )


def load_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Carga y valida el CSV sin modificar su contenido.

    Parameters
    ----------
    csv_path:
        Ruta al CSV. Si se omite, se usa ``data/raw/creditcard.csv`` desde la raíz
        del proyecto.
    """
    dataset_path = Path(csv_path) if csv_path is not None else DEFAULT_DATASET_PATH
    if not dataset_path.is_file():
        raise DataFileNotFoundError(
            f"No se encontró el dataset en '{dataset_path}'. "
            "Descárguelo desde Kaggle y ubíquelo en data/raw/creditcard.csv, "
            "o entregue una ruta mediante csv_path."
        )

    try:
        dataframe = pd.read_csv(dataset_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise DatasetValidationError(
            f"No fue posible leer el CSV '{dataset_path}': {error}"
        ) from error

    validate_dataset_columns(dataframe)
    return dataframe


def separate_features_target(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa predictores y objetivo, preservando el orden de las características."""
    validate_dataset_columns(dataframe)
    if target_column != TARGET_COLUMN:
        raise DatasetValidationError(
            f"La columna objetivo debe ser '{TARGET_COLUMN}', no '{target_column}'."
        )

    features = dataframe.loc[:, list(PREDICTOR_COLUMNS)].copy()
    target = dataframe.loc[:, target_column].copy()
    return features, target


def diagnose_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Devuelve métricas descriptivas para decidir la limpieza de forma explícita.

    El diagnóstico no elimina filas ni ajusta transformadores; por tanto puede usarse
    durante la exploración sin alterar el dataframe original.
    """
    validate_dataset_columns(dataframe)
    class_counts = dataframe[TARGET_COLUMN].value_counts().sort_index()
    class_percentages = (class_counts / len(dataframe) * 100).round(6) if len(dataframe) else class_counts
    return {
        "shape": dataframe.shape,
        "dtypes": dataframe.dtypes.copy(),
        "missing_values": dataframe.isna().sum(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "descriptive_statistics": dataframe.describe().transpose(),
    }
