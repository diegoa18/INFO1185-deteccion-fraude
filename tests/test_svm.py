from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import PREDICTOR_COLUMNS, TARGET_COLUMN
from src.data.preprocess import stratified_train_test_split
from src.models.svm import create_svm_pipeline


@pytest.fixture
def synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(123)
    row_count = 100
    features = pd.DataFrame(
        rng.normal(size=(row_count, len(PREDICTOR_COLUMNS))),
        columns=PREDICTOR_COLUMNS,
    )
    target = pd.Series(np.repeat([0, 1], [90, 10]), name=TARGET_COLUMN)
    return features, target


def test_svm_pipeline_creation_and_configuration():
    pipeline = create_svm_pipeline(
        kernel="poly",
        C=2.5,
        gamma="auto",
        degree=4,
    )

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["scale", "select", "classifier"]

    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, SVC)
    assert classifier.kernel == "poly"
    assert classifier.C == 2.5
    assert classifier.gamma == "auto"
    assert classifier.degree == 4
    assert classifier.class_weight is None
    assert classifier.random_state == 42


def test_svm_pipeline_fit_and_predict_uses_prepared_split(synthetic_dataset):
    features, target = synthetic_dataset
    X_train, X_test, y_train, y_test = stratified_train_test_split(features, target)
    pipeline = create_svm_pipeline(k_features=5)

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(np.unique(predictions)).issubset(set(y_train.unique()))
