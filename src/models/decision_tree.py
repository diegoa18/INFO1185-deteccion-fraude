from __future__ import annotations
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.config import RANDOM_STATE
from src.data.preprocess import ScalerName, create_scaler
from src.features.feature_selection import make_mutual_information_selector


def create_decision_tree_pipeline(
    *,
    scaler_kind: ScalerName = "none",
    k_features: int | str = "all",
    random_state: int = RANDOM_STATE,
    **tree_kwargs: Any,
) -> Pipeline:
    if "random_state" in tree_kwargs:
        tree_kwargs.pop("random_state")

    pipeline_steps = [
        ("scale", create_scaler(scaler_kind)),
        ("select", make_mutual_information_selector(k=k_features, random_state=random_state)),
        ("classifier", DecisionTreeClassifier(random_state=random_state, **tree_kwargs)),
    ]

    return Pipeline(pipeline_steps)
