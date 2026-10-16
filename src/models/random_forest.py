from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from src.config import RANDOM_STATE
from src.data.preprocess import ScalerName
from src.models.decision_tree import create_decision_tree_pipeline


class CustomRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 15,
        *,
        scaler_kind: ScalerName = "none",
        k_features: int | str = "all",
        random_state: int = RANDOM_STATE,
        max_features: int | float | str | None = "sqrt",
        **tree_kwargs: Any,
    ):
        self.n_estimators = n_estimators
        self.scaler_kind = scaler_kind
        self.k_features = k_features
        self.random_state = random_state
        self.max_features = max_features
        self.tree_kwargs = tree_kwargs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CustomRandomForest:
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators debe ser un entero positivo.")

        rng = np.random.default_rng(self.random_state)
        self.seeds_ = rng.integers(0, 10**9, size=self.n_estimators)
        self.estimators_ = []
        n_samples = len(X)

        for seed in self.seeds_:
            tree_rng = np.random.default_rng(seed)
            bootstrap_indices = tree_rng.choice(n_samples, size=n_samples, replace=True)

            X_boot = X.iloc[bootstrap_indices].copy()
            y_boot = y.iloc[bootstrap_indices].copy()

            pipeline = create_decision_tree_pipeline(
                scaler_kind=self.scaler_kind,
                k_features=self.k_features,
                random_state=int(seed),
                max_features=self.max_features,
                **self.tree_kwargs,
            )

            pipeline.fit(X_boot, y_boot)
            self.estimators_.append(pipeline)

        self.classes_ = np.unique(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "estimators_")

        predictions = np.column_stack([
            estimator.predict(X) for estimator in self.estimators_
        ])

        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=1,
            arr=predictions
        )

        return majority_votes
