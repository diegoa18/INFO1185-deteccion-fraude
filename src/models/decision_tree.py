from __future__ import annotations
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.config import RANDOM_STATE
from src.data.preprocess import ScalerName
from src.models.pipeline_utils import create_preprocessing_pipeline


def create_decision_tree_pipeline(
    *,
    scaler_kind: ScalerName = "none",
    k_features: int | str | None = "all",
    random_state: int = RANDOM_STATE,
    **tree_kwargs: Any,
) -> Pipeline:
    if "random_state" in tree_kwargs:
        tree_kwargs.pop("random_state")

    preprocessor = create_preprocessing_pipeline(
        scaler_kind=scaler_kind,
        k_features=k_features,
        random_state=random_state,
    )
    pipeline_steps = [
        *preprocessor.steps,
        ("classifier", DecisionTreeClassifier(random_state=random_state, **tree_kwargs)),
    ]

    return Pipeline(pipeline_steps)
