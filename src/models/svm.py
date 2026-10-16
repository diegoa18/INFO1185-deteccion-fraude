"""Pipeline configurable de Support Vector Machine para detección de fraude."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import RANDOM_STATE
from src.data.preprocess import ScalerName, create_scaler
from src.features.feature_selection import make_mutual_information_selector

SVMKernel = Literal["linear", "poly", "rbf", "sigmoid"]


def create_svm_pipeline(
    *,
    scaler_kind: ScalerName = "standard",
    k_features: int | Literal["all"] = "all",
    kernel: SVMKernel = "rbf",
    C: float = 1.0,
    gamma: float | Literal["scale", "auto"] = "scale",
    degree: int = 3,
    class_weight: Mapping[Any, float] | Literal["balanced"] | None = None,
    random_state: int = RANDOM_STATE,
    **svc_kwargs: Any,
) -> Pipeline:
    """Construye un pipeline SVM sin ajustarlo.

    El escalado y la selección se ajustan cuando se invoca ``fit`` con ``X_train`` y
    ``y_train``. De ese modo, la misma partición preparada se reutiliza sin filtrar
    información desde ``X_test``. Los parámetros de ``SVC`` quedan expuestos para
    que una etapa posterior pueda evaluarlos u optimizarlos sin cambiar esta función.
    """
    if "random_state" in svc_kwargs:
        svc_kwargs.pop("random_state")

    return Pipeline(
        [
            ("scale", create_scaler(scaler_kind)),
            (
                "select",
                make_mutual_information_selector(
                    k=k_features,
                    random_state=random_state,
                ),
            ),
            (
                "classifier",
                SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    degree=degree,
                    class_weight=class_weight,
                    random_state=random_state,
                    **svc_kwargs,
                ),
            ),
        ]
    )
