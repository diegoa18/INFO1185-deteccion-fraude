from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PREDICTOR_COLUMNS, TARGET_COLUMN
from src.models.knn import create_knn_classifier, create_knn_pipeline


@pytest.fixture
def synthetic_features_target() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(321)
    row_count = 100
    features = pd.DataFrame(
        rng.normal(size=(row_count, len(PREDICTOR_COLUMNS))),
        columns=PREDICTOR_COLUMNS,
    )
    target = pd.Series(np.repeat([0, 1], [90, 10]), name=TARGET_COLUMN)
    return features, target


def test_creates_configurable_knn_classifier():
    classifier = create_knn_classifier(
        n_neighbors=3,
        metric="manhattan",
        weights="distance",
        p=1,
    )

    assert isinstance(classifier, KNeighborsClassifier)
    assert classifier.n_neighbors == 3
    assert classifier.metric == "manhattan"
    assert classifier.weights == "distance"
    assert classifier.p == 1


@pytest.mark.parametrize("n_neighbors", [0, -1, 1.5, True])
def test_knn_classifier_rejects_non_positive_integer_neighbors(n_neighbors):
    with pytest.raises(ValueError, match="entero positivo"):
        create_knn_classifier(n_neighbors=n_neighbors)


def test_knn_classifier_fits_and_predicts(synthetic_features_target):
    features, target = synthetic_features_target
    classifier = create_knn_classifier(n_neighbors=3)
    classifier.fit(features, target)
    predictions = classifier.predict(features)

    assert len(predictions) == len(features)
    assert set(np.unique(predictions)).issubset(set(target.unique()))


def test_knn_pipeline_creation_and_structure():
    pipeline = create_knn_pipeline(k_features=5, n_neighbors=3)

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["scale", "select", "classifier"]
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)
    assert isinstance(pipeline.named_steps["select"], SelectKBest)
    assert isinstance(pipeline.named_steps["classifier"], KNeighborsClassifier)


def test_knn_pipeline_fits_predicts_and_excludes_target_from_predictors(synthetic_features_target):
    features, target = synthetic_features_target
    pipeline = create_knn_pipeline(k_features=5, n_neighbors=3)
    pipeline.fit(features, target)
    predictions = pipeline.predict(features)

    assert TARGET_COLUMN not in features.columns
    assert TARGET_COLUMN not in pipeline.named_steps["scale"].feature_names_in_
    assert len(predictions) == len(features)
    assert set(np.unique(predictions)).issubset(set(target.unique()))


def test_knn_pipeline_exposes_classifier_parameters_for_future_search():
    pipeline = create_knn_pipeline(n_neighbors=3, metric="minkowski")

    assert pipeline.get_params()["classifier__n_neighbors"] == 3
    assert pipeline.get_params()["classifier__metric"] == "minkowski"

    pipeline.set_params(classifier__n_neighbors=7, classifier__metric="manhattan")

    assert pipeline.named_steps["classifier"].n_neighbors == 7
    assert pipeline.named_steps["classifier"].metric == "manhattan"
