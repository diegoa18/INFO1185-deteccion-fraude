"""Componentes compartidos de los pipelines de clasificadores."""

from __future__ import annotations

from typing import Literal

from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.data.preprocess import ScalerName, create_scaler
from src.features.feature_selection import make_mutual_information_selector


def create_preprocessing_pipeline(
    *,
    scaler_kind: ScalerName = "standard",
    k_features: int | Literal["all"] | None = "all",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Crea escalado y selección sin ajustar.

    Esta pieza se comparte entre los clasificadores y el pipeline que realiza
    sobremuestreo. Al quedar dentro de un estimador, ambos transformadores se
    ajustan exclusivamente con los datos que recibe cada llamada a ``fit``.
    """
    selector = (
        "passthrough"
        if k_features is None
        else make_mutual_information_selector(
            k=k_features,
            random_state=random_state,
        )
    )
    return Pipeline(
        [
            ("scale", create_scaler(scaler_kind)),
            (
                "select",
                selector,
            ),
        ]
    )
