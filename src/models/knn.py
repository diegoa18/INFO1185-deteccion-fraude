"""Constructor reutilizable del clasificador k-Nearest Neighbors."""

from __future__ import annotations

from typing import Any, Callable, Literal

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.data.preprocess import ScalerName, create_scaler
from src.features.feature_selection import make_mutual_information_selector


def create_knn_classifier(
    *,
    n_neighbors: int = 5,
    metric: str | Callable[..., float] = "minkowski",
    weights: str | Callable[..., Any] = "uniform",
    p: int | float = 2,
    **knn_kwargs: Any,
) -> KNeighborsClassifier:
    """Crea un clasificador k-NN configurable sin ajustarlo.

    La validez de métricas y pesos específicos se delega a scikit-learn para
    conservar su API. ``n_neighbors`` se valida aquí porque debe ser un entero
    positivo antes de entrenar el modelo.
    """
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors <= 0:
        raise ValueError("n_neighbors debe ser un entero positivo.")

    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
        p=p,
        **knn_kwargs,
    )


def create_knn_pipeline(
    *,
    scaler_kind: ScalerName = "standard",
    k_features: int | Literal["all"] = "all",
    random_state: int = RANDOM_STATE,
    n_neighbors: int = 5,
    metric: str | Callable[..., float] = "minkowski",
    weights: str | Callable[..., Any] = "uniform",
    p: int | float = 2,
    **knn_kwargs: Any,
) -> Pipeline:
    """Crea el pipeline de escalado, selección y clasificación k-NN.

    Por defecto, ``k_features="all"`` conserva todas las características;
    entregue un entero positivo para seleccionar esa cantidad mediante
    información mutua. Los transformadores se devuelven sin ajustar. Al
    invocar ``fit`` sobre el pipeline, el escalado y la selección se ajustan
    únicamente con los datos de entrenamiento recibidos, evitando fuga de
    información durante la validación cruzada.
    """
    pipeline_steps = [
        ("scale", create_scaler(scaler_kind)),
        ("select", make_mutual_information_selector(k=k_features, random_state=random_state)),
        (
            "classifier",
            create_knn_classifier(
                n_neighbors=n_neighbors,
                metric=metric,
                weights=weights,
                p=p,
                **knn_kwargs,
            ),
        ),
    ]

    return Pipeline(pipeline_steps)
