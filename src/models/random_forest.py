from __future__ import annotations
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
        max_depth: int | None = None,
        max_features: int | float | str | None = "sqrt",
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        criterion: str = "gini",
        class_weight: dict | str | None = None,
        scaler_kind: ScalerName = "none",
        k_features: int | str | None = "all",
        random_state: int = RANDOM_STATE,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.class_weight = class_weight
        self.scaler_kind = scaler_kind
        self.k_features = k_features
        self.random_state = random_state

    @staticmethod
    def _generate_unique_seeds(n_estimators: int, random_state: int) -> np.ndarray:
        """Genera ``n_estimators`` semillas estrictamente distintas de forma determinista."""
        rng = np.random.default_rng(random_state)
        upper_bound = max(10**9, n_estimators * 10)
        seeds: list[int] = []
        seen: set[int] = set()
        while len(seeds) < n_estimators:
            candidate = int(rng.integers(0, upper_bound))
            if candidate not in seen:
                seen.add(candidate)
                seeds.append(candidate)
        return np.asarray(seeds, dtype=np.int64)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> CustomRandomForest:
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators debe ser un entero positivo.")

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ser = y if isinstance(y, pd.Series) else pd.Series(y)

        if len(X_df) == 0 or len(y_ser) == 0:
            raise ValueError("No es posible entrenar con un conjunto de datos vacío.")
        if len(X_df) != len(y_ser):
            raise ValueError("X e y deben tener la misma cantidad de filas.")

        self.seeds_ = self._generate_unique_seeds(self.n_estimators, self.random_state)
        self.estimators_ = []
        self.bootstrap_indices_: list[np.ndarray] = []
        n_samples = len(X_df)

        for seed in self.seeds_:
            tree_rng = np.random.default_rng(int(seed))
            bootstrap_idx = tree_rng.choice(n_samples, size=n_samples, replace=True)
            self.bootstrap_indices_.append(bootstrap_idx)
            X_boot = X_df.iloc[bootstrap_idx].copy()
            y_boot = y_ser.iloc[bootstrap_idx].copy()

            pipeline = create_decision_tree_pipeline(
                scaler_kind=self.scaler_kind,
                k_features=self.k_features,
                random_state=int(seed),
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                class_weight=self.class_weight,
            )

            pipeline.fit(X_boot, y_boot)
            self.estimators_.append(pipeline)

        self.classes_ = np.sort(np.unique(y_ser))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["estimators_", "classes_"])
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        predictions = np.column_stack([
            estimator.predict(X_df) for estimator in self.estimators_
        ])

        def _majority_vote(row_votes: np.ndarray) -> int:
            counts = np.array([np.sum(row_votes == cls) for cls in self.classes_])
            return int(self.classes_[int(np.argmax(counts))])

        majority_votes = np.apply_along_axis(_majority_vote, axis=1, arr=predictions)
        return majority_votes
