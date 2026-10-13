from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import PREDICTOR_COLUMNS, TARGET_COLUMN
from src.data.load_data import DatasetValidationError, DataFileNotFoundError, load_dataset, separate_features_target
from src.data.preprocess import create_scaler, stratified_train_test_split
from src.features.feature_selection import make_mutual_information_selector, selected_feature_names


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    row_count = 100
    data = pd.DataFrame(rng.normal(size=(row_count, len(PREDICTOR_COLUMNS))), columns=PREDICTOR_COLUMNS)
    data[TARGET_COLUMN] = np.repeat([0, 1], [90, 10])
    return data


def test_loads_valid_csv_and_separates_features_target(tmp_path, synthetic_dataset):
    csv_path = tmp_path / "creditcard.csv"
    synthetic_dataset.to_csv(csv_path, index=False)

    loaded = load_dataset(csv_path)
    features, target = separate_features_target(loaded)

    pd.testing.assert_frame_equal(loaded, synthetic_dataset)
    assert list(features.columns) == list(PREDICTOR_COLUMNS)
    assert TARGET_COLUMN not in features
    assert target.name == TARGET_COLUMN


def test_load_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(DataFileNotFoundError, match="No se encontró el dataset"):
        load_dataset(tmp_path / "missing.csv")


def test_load_dataset_rejects_missing_required_columns(tmp_path, synthetic_dataset):
    csv_path = tmp_path / "incomplete.csv"
    synthetic_dataset.drop(columns="V28").to_csv(csv_path, index=False)

    with pytest.raises(DatasetValidationError, match="V28"):
        load_dataset(csv_path)


def test_stratified_split_is_reproducible_and_preserves_class_proportions(synthetic_dataset):
    features, target = separate_features_target(synthetic_dataset)
    first_split = stratified_train_test_split(features, target)
    second_split = stratified_train_test_split(features, target)
    train_features, test_features, train_target, test_target = first_split

    pd.testing.assert_frame_equal(train_features, second_split[0])
    pd.testing.assert_series_equal(test_target, second_split[3])
    assert len(train_features) == 70
    assert len(test_features) == 30
    assert train_target.mean() == pytest.approx(target.mean(), abs=0.01)
    assert test_target.mean() == pytest.approx(target.mean(), abs=0.01)
    assert TARGET_COLUMN not in train_features
    assert list(train_features.columns) == list(PREDICTOR_COLUMNS)


def test_split_rejects_target_as_feature(synthetic_dataset):
    features, target = separate_features_target(synthetic_dataset)
    features[TARGET_COLUMN] = target

    with pytest.raises(ValueError, match="no puede formar parte"):
        stratified_train_test_split(features, target)


def test_scaler_is_fit_only_on_training_partition(synthetic_dataset):
    features, target = separate_features_target(synthetic_dataset)
    train_features, test_features, _, _ = stratified_train_test_split(features, target)
    scaler = create_scaler("standard")
    scaler.fit(train_features)

    np.testing.assert_allclose(scaler.mean_, train_features.mean().to_numpy())
    assert not np.allclose(scaler.mean_, test_features.mean().to_numpy())


def test_selector_dimensions_names_and_pipeline_integration(synthetic_dataset):
    features, target = separate_features_target(synthetic_dataset)
    train_features, test_features, train_target, _ = stratified_train_test_split(features, target)
    selector = make_mutual_information_selector(k=5)
    pipeline = Pipeline([
        ("scale", create_scaler("standard")),
        ("select", selector),
    ])

    transformed_train = pipeline.fit_transform(train_features, train_target)
    transformed_test = pipeline.transform(test_features)
    names = selected_feature_names(pipeline.named_steps["select"], train_features.columns)

    assert transformed_train.shape == (len(train_features), 5)
    assert transformed_test.shape == (len(test_features), 5)
    assert len(names) == 5
    assert set(names).issubset(PREDICTOR_COLUMNS)
