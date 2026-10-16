from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from src.config import PREDICTOR_COLUMNS, TARGET_COLUMN
from src.models.random_forest import CustomRandomForest


@pytest.fixture
def synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    row_count = 100
    features = pd.DataFrame(
        rng.normal(size=(row_count, len(PREDICTOR_COLUMNS))),
        columns=PREDICTOR_COLUMNS,
    )
    target = pd.Series(np.repeat([0, 1], [90, 10]), name=TARGET_COLUMN)
    return features, target


def test_custom_random_forest_fit_and_predict(synthetic_dataset):
    features, target = synthetic_dataset
    rf = CustomRandomForest(n_estimators=5, max_depth=3, random_state=42)

    rf.fit(features, target)
    predictions = rf.predict(features)

    assert len(rf.estimators_) == 5
    assert len(predictions) == len(target)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert len(set(rf.seeds_)) == 5  # Semillas estrictamente únicas


def test_custom_random_forest_bootstrap_samples_and_diversity(synthetic_dataset):
    features, target = synthetic_dataset
    rf = CustomRandomForest(n_estimators=5, random_state=123)
    rf.fit(features, target)

    assert len(rf.bootstrap_indices_) == 5
    for idx in rf.bootstrap_indices_:
        assert len(idx) == len(features)
        assert len(np.unique(idx)) < len(features)

    assert not np.array_equal(rf.bootstrap_indices_[0], rf.bootstrap_indices_[1])


def test_custom_random_forest_reproducibility(synthetic_dataset):
    features, target = synthetic_dataset
    rf1 = CustomRandomForest(n_estimators=5, max_depth=4, random_state=99)
    rf2 = CustomRandomForest(n_estimators=5, max_depth=4, random_state=99)
    rf_diff = CustomRandomForest(n_estimators=5, max_depth=4, random_state=100)

    rf1.fit(features, target)
    rf2.fit(features, target)
    rf_diff.fit(features, target)

    np.testing.assert_array_equal(rf1.seeds_, rf2.seeds_)
    for b1, b2 in zip(rf1.bootstrap_indices_, rf2.bootstrap_indices_):
        np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(rf1.predict(features), rf2.predict(features))
    assert not np.array_equal(rf1.seeds_, rf_diff.seeds_)


def test_custom_random_forest_sklearn_get_params_and_clone(synthetic_dataset):
    features, target = synthetic_dataset
    rf = CustomRandomForest(
        n_estimators=7,
        max_depth=6,
        max_features="sqrt",
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    params = rf.get_params()
    assert params["n_estimators"] == 7
    assert params["max_depth"] == 6
    assert params["max_features"] == "sqrt"
    assert params["min_samples_split"] == 4
    assert params["min_samples_leaf"] == 2
    assert params["class_weight"] == "balanced"

    cloned = clone(rf)
    cloned.set_params(max_depth=3, n_estimators=3)
    cloned.fit(features, target)
    assert cloned.estimators_[0].named_steps["classifier"].max_depth == 3


def test_custom_random_forest_tie_breaking_selects_smaller_class(synthetic_dataset):
    features, target = synthetic_dataset
    rf = CustomRandomForest(n_estimators=2, random_state=42)
    rf.fit(features, target)

    class _ConstantEstimator:
        def __init__(self, value: int):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value, dtype=int)

    rf.estimators_ = [_ConstantEstimator(1), _ConstantEstimator(0)]
    preds = rf.predict(features.iloc[:5])
    np.testing.assert_array_equal(preds, np.zeros(5, dtype=int))


def test_custom_random_forest_rejects_invalid_n_estimators():
    with pytest.raises(ValueError, match="n_estimators debe ser un entero positivo"):
        CustomRandomForest(n_estimators=0).fit(pd.DataFrame(), pd.Series(dtype=int))
