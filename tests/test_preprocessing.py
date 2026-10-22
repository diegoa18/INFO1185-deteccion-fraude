from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import PREDICTOR_COLUMNS, TARGET_COLUMN
from src.data.load_data import (
    DatasetValidationError,
    DataFileNotFoundError,
    diagnose_dataset,
    load_dataset,
    separate_features_target,
)
from src.data.preprocess import (
    clean_dataset,
    create_scaler,
    stratified_sample,
    stratified_train_test_split,
)
from src.features.feature_selection import make_mutual_information_selector, selected_feature_names
from src.validation import make_stratified_kfold, materialize_stratified_folds


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


def test_train_test_split_has_no_overlapping_row_identifiers(synthetic_dataset):
    features, target = separate_features_target(synthetic_dataset)
    train_features, test_features, _, _ = stratified_train_test_split(features, target)

    assert train_features.index.is_unique
    assert test_features.index.is_unique
    assert train_features.index.intersection(test_features.index).empty


def test_train_test_split_keeps_identical_predictors_on_the_same_side(synthetic_dataset):
    duplicated = pd.concat([synthetic_dataset, synthetic_dataset.iloc[[0]]], ignore_index=True)
    features, target = separate_features_target(duplicated)
    train_features, test_features, _, _ = stratified_train_test_split(features, target)
    train_hashes = set(pd.util.hash_pandas_object(train_features, index=False))
    test_hashes = set(pd.util.hash_pandas_object(test_features, index=False))

    assert train_hashes.isdisjoint(test_hashes)
    assert len(train_features) + len(test_features) == len(features)


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


def test_pipeline_transformers_are_fit_only_on_each_fold_training_partition(synthetic_dataset):
    features, target = separate_features_target(synthetic_dataset)
    train_features, test_features, train_target, _ = stratified_train_test_split(features, target)
    folds = materialize_stratified_folds(train_features, train_target)
    fold_train_index, fold_validation_index = folds[0]
    fold_train = train_features.iloc[fold_train_index]
    fold_validation = train_features.iloc[fold_validation_index]
    pipeline = Pipeline([
        ("scale", create_scaler("standard")),
        ("select", make_mutual_information_selector(k=5)),
    ])

    pipeline.fit(fold_train, train_target.iloc[fold_train_index])

    np.testing.assert_allclose(pipeline.named_steps["scale"].mean_, fold_train.mean().to_numpy())
    assert not np.allclose(pipeline.named_steps["scale"].mean_, fold_validation.mean().to_numpy())
    assert pipeline.transform(fold_validation).shape == (len(fold_validation), 5)
    assert pipeline.transform(test_features).shape == (len(test_features), 5)


def test_stratified_five_fold_configuration_is_reproducible_and_uses_training_only(synthetic_dataset):
    features, target = separate_features_target(synthetic_dataset)
    train_features, test_features, train_target, _ = stratified_train_test_split(features, target)
    first_folds = materialize_stratified_folds(train_features, train_target)
    second_folds = materialize_stratified_folds(train_features, train_target)

    assert len(first_folds) == 5
    assert test_features.index.intersection(train_features.index).empty
    for (first_train, first_validation), (second_train, second_validation) in zip(first_folds, second_folds):
        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_validation, second_validation)
        assert len(first_train) == 56
        assert len(first_validation) == 14
        for label, total in train_target.value_counts().items():
            expected_validation_count = total / 5
            actual_validation_count = (train_target.iloc[first_validation] == label).sum()
            assert abs(actual_validation_count - expected_validation_count) <= 1

    cv = make_stratified_kfold()
    assert cv.n_splits == 5
    assert cv.shuffle is True


def test_stratified_sample_is_reproducible_and_preserves_class_proportions(synthetic_dataset):
    features, target = separate_features_target(synthetic_dataset)
    enlarged_features = pd.concat([features] * 10, ignore_index=True)
    enlarged_target = pd.concat([target] * 10, ignore_index=True)

    first_features, first_target = stratified_sample(
        enlarged_features,
        enlarged_target,
        sample_size=300,
    )
    second_features, second_target = stratified_sample(
        enlarged_features,
        enlarged_target,
        sample_size=300,
    )

    pd.testing.assert_frame_equal(first_features, second_features)
    pd.testing.assert_series_equal(first_target, second_target)
    assert first_target.value_counts().to_dict() == {0: 270, 1: 30}
    assert first_target.mean() == pytest.approx(enlarged_target.mean())
    assert TARGET_COLUMN not in first_features


def test_cross_validation_rejects_target_as_feature(synthetic_dataset):
    features, target = separate_features_target(synthetic_dataset)
    features[TARGET_COLUMN] = target

    with pytest.raises(ValueError, match="no puede formar parte"):
        materialize_stratified_folds(features, target)


def test_clean_dataset_keeps_duplicates_unless_explicitly_requested(synthetic_dataset):
    duplicated = pd.concat([synthetic_dataset, synthetic_dataset.iloc[[0]]])

    preserved = clean_dataset(duplicated)
    deduplicated = clean_dataset(duplicated, remove_exact_duplicates=True)

    assert len(preserved) == len(duplicated)
    assert preserved.duplicated().sum() == 1
    assert len(deduplicated) == len(synthetic_dataset)


def test_diagnosis_reports_duplicate_impact_by_class(synthetic_dataset):
    fraudulent_duplicate = pd.concat([synthetic_dataset, synthetic_dataset.iloc[[90]]])

    diagnosis = diagnose_dataset(fraudulent_duplicate)

    assert diagnosis["duplicate_rows"] == 1
    assert diagnosis["duplicate_rows_by_class"].to_dict() == {1: 1}
