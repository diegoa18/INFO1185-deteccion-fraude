from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.config import PREDICTOR_COLUMNS, TARGET_COLUMN
from src.models.decision_tree import create_decision_tree_pipeline


@pytest.fixture
def synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(123)
    row_count = 100
    features = pd.DataFrame(
        rng.normal(size=(row_count, len(PREDICTOR_COLUMNS))),
        columns=PREDICTOR_COLUMNS
    )
    target = pd.Series(np.repeat([0, 1], [90, 10]), name=TARGET_COLUMN)
    return features, target


def test_decision_tree_pipeline_creation_and_structure():
    pipeline = create_decision_tree_pipeline(
        scaler_kind="none",
        k_features=10,
        max_depth=5
    )

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == "scale"
    assert pipeline.steps[1][0] == "select"
    assert pipeline.steps[2][0] == "classifier"

    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, DecisionTreeClassifier)
    assert classifier.max_depth == 5
    assert classifier.random_state == 42


def test_decision_tree_pipeline_fit_and_predict(synthetic_dataset):
    features, target = synthetic_dataset
    pipeline = create_decision_tree_pipeline(k_features=5, max_depth=3)
    pipeline.fit(features, target)
    predictions = pipeline.predict(features)

    assert len(predictions) == len(target)
    assert set(np.unique(predictions)).issubset({0, 1})
