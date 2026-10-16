from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.config import PREDICTOR_COLUMNS, TARGET_COLUMN
from src.models.random_forest import CustomRandomForest


@pytest.fixture
def synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    row_count = 100
    features = pd.DataFrame(
        rng.normal(size=(row_count, len(PREDICTOR_COLUMNS))),
        columns=PREDICTOR_COLUMNS
    )
    target = pd.Series(np.repeat([0, 1], [90, 10]), name=TARGET_COLUMN)
    return features, target


def test_custom_random_forest_fit_and_predict(synthetic_dataset):
    features, target = synthetic_dataset
    rf = CustomRandomForest(n_estimators=3, max_depth=2, random_state=42)
    rf.fit(features, target)
    predictions = rf.predict(features)
    assert len(rf.estimators_) == 3
    assert len(predictions) == len(target)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert len(set(rf.seeds_)) == 3


def test_custom_random_forest_bootstrap_diversity(synthetic_dataset):
    features, target = synthetic_dataset
    rf = CustomRandomForest(n_estimators=2, random_state=123)
    rf.fit(features, target)
    tree1_seed = rf.estimators_[0].named_steps["classifier"].random_state
    tree2_seed = rf.estimators_[1].named_steps["classifier"].random_state
    assert tree1_seed != tree2_seed


def test_custom_random_forest_rejects_invalid_n_estimators():
    with pytest.raises(ValueError, match="n_estimators debe ser un entero positivo"):
        CustomRandomForest(n_estimators=0).fit(pd.DataFrame(), pd.Series())
