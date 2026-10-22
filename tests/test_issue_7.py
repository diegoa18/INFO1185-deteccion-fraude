from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score

from src.config import PREDICTOR_COLUMNS, TARGET_COLUMN
from src.imbalance import (
    LeakageSafeResamplingPipeline,
    RandomOverSampler,
    analyze_class_imbalance,
)
from src.models.knn import create_knn_classifier
from src.models.pipeline_utils import create_preprocessing_pipeline
from src.validation import (
    DEFAULT_SEARCH_SPACES,
    create_experiment_estimator,
    create_issue7_cv,
    materialize_stratified_folds,
    run_issue7_experiments,
    select_best_configuration,
)


@pytest.fixture
def imbalanced_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    features = pd.DataFrame(
        rng.normal(size=(100, len(PREDICTOR_COLUMNS))),
        columns=PREDICTOR_COLUMNS,
    )
    target = pd.Series(np.repeat([0, 1], [80, 20]), name=TARGET_COLUMN)
    return features, target


def test_analyze_class_imbalance_keeps_fraud_as_positive_class(imbalanced_dataset):
    _, target = imbalanced_dataset

    summary = analyze_class_imbalance(target)

    assert summary[["class", "label", "count"]].to_dict("records") == [
        {"class": 0, "label": "legítima", "count": 80},
        {"class": 1, "label": "fraude", "count": 20},
    ]
    assert summary.loc[1, "percentage"] == pytest.approx(20.0)
    assert summary["majority_to_minority_ratio"].iloc[0] == pytest.approx(4.0)


def test_random_oversampler_balances_only_the_data_given_to_it(imbalanced_dataset):
    features, target = imbalanced_dataset
    sampler = RandomOverSampler(random_state=42)

    resampled_features, resampled_target = sampler.fit_resample(features, target)

    assert features.shape == (100, len(PREDICTOR_COLUMNS))
    assert target.value_counts().to_dict() == {0: 80, 1: 20}
    assert resampled_features.shape[0] == 160
    assert pd.Series(resampled_target).value_counts().to_dict() == {0: 80, 1: 80}
    assert sampler.class_counts_before_ == {0: 80, 1: 20}
    assert sampler.class_counts_after_ == {0: 80, 1: 80}


def test_resampling_pipeline_fits_preprocessing_before_balancing(imbalanced_dataset):
    features, target = imbalanced_dataset
    train_positions = np.r_[np.arange(64), np.arange(80, 96)]
    validation_positions = np.setdiff1d(np.arange(len(features)), train_positions)
    fold_train = features.iloc[train_positions]
    fold_target = target.iloc[train_positions]
    fold_validation = features.iloc[validation_positions]
    estimator = LeakageSafeResamplingPipeline(
        preprocessor=create_preprocessing_pipeline(k_features=5),
        sampler=RandomOverSampler(random_state=42),
        classifier=create_knn_classifier(n_neighbors=3),
    )

    estimator.fit(fold_train, fold_target)

    np.testing.assert_allclose(
        estimator.preprocessor_.named_steps["scale"].mean_,
        fold_train.mean().to_numpy(),
    )
    assert estimator.class_counts_before_ == {0: 64, 1: 16}
    assert estimator.class_counts_after_ == {0: 64, 1: 64}
    assert len(estimator.predict(fold_validation)) == len(fold_validation)
    assert len(fold_validation) == 20


def test_select_best_configuration_uses_precision_only_after_recall_tie():
    candidates = [
        {"recall_cv_mean": 0.75, "precision_cv_mean": 0.10},
        {"recall_cv_mean": 0.75, "precision_cv_mean": 0.40},
        {"recall_cv_mean": 0.74, "precision_cv_mean": 0.99},
    ]

    assert select_best_configuration(candidates) == 1


def test_custom_random_forest_default_search_uses_only_odd_tree_counts():
    candidates = [
        configuration["classifier__n_estimators"]
        for configuration in DEFAULT_SEARCH_SPACES["custom_random_forest"]
    ]

    assert candidates == [[5], [11], [5], [11]]
    assert all(tree_count % 2 == 1 for values in candidates for tree_count in values)


def test_issue_7_runs_eight_conditions_on_exactly_shared_folds(imbalanced_dataset):
    features, target = imbalanced_dataset
    folds = materialize_stratified_folds(features, target, cv=create_issue7_cv())
    small_spaces = {
        "knn": [
            {
                "classifier__n_neighbors": [3],
                "classifier__weights": ["uniform"],
                "classifier__metric": ["euclidean"],
            }
        ],
        "svm": [{"classifier__kernel": ["linear"], "classifier__C": [1.0]}],
        "decision_tree": [
            {
                "classifier__max_depth": [3],
                "classifier__min_samples_split": [2],
                "classifier__min_samples_leaf": [1],
                "classifier__criterion": ["gini"],
            }
        ],
        "custom_random_forest": [
            {
                "classifier__n_estimators": [1],
                "classifier__max_depth": [3],
                "classifier__max_features": ["sqrt"],
            }
        ],
    }

    result = run_issue7_experiments(
        features,
        target,
        folds=folds,
        search_spaces=small_spaces,
    )

    assert len(result.folds) == 5
    for (expected_train, expected_validation), (actual_train, actual_validation) in zip(
        folds,
        result.folds,
    ):
        np.testing.assert_array_equal(actual_train, expected_train)
        np.testing.assert_array_equal(actual_validation, expected_validation)
    assert len(result.summary) == 8
    assert set(result.summary["model"]) == {
        "k-NN",
        "SVM",
        "Árbol de Decisión",
        "Random Forest propio",
    }
    assert set(result.summary["balanced"]) == {False, True}
    assert result.fold_balance.groupby(["model", "balanced"]).size().eq(5).all()
    balanced_rows = result.summary.loc[result.summary["balanced"]]
    assert (balanced_rows["class_0_after"] == balanced_rows["class_1_after"]).all()
    assert len(result.best_estimators) == 8
    # Dos tipos de preprocesamiento (standard/none), cinco folds y el ajuste
    # final: 2 * (5 + 1), en vez de recalcularlos por candidato y condición.
    assert result.preprocessing_fit_count == 12
    assert result.resampling_fit_count == 10
    assert result.timings.query("stage == 'preprocessing'").shape[0] == 12
    assert result.timings.query("stage == 'resampling'").shape[0] == 10

    # Los validation folds no se sobremuestrean: sus conteos suman exactamente
    # la distribución original y no aparecen en class_*_after.
    validation_counts = result.fold_balance.groupby(["model", "balanced"]).agg(
        class_0=("validation_class_0", "sum"),
        class_1=("validation_class_1", "sum"),
    )
    assert (validation_counts["class_0"] == 80).all()
    assert (validation_counts["class_1"] == 20).all()
    assert result.summary["selected_features"].map(lambda value: len(json.loads(value))).eq(10).all()


def test_cached_preprocessing_preserves_naive_candidate_selection(imbalanced_dataset):
    features, target = imbalanced_dataset
    folds = materialize_stratified_folds(features, target, cv=create_issue7_cv())
    knn_candidates = [
        {
            "classifier__n_neighbors": [3],
            "classifier__weights": ["uniform"],
            "classifier__metric": ["euclidean"],
        },
        {
            "classifier__n_neighbors": [5],
            "classifier__weights": ["distance"],
            "classifier__metric": ["manhattan"],
        },
    ]
    spaces = {
        "knn": knn_candidates,
        "svm": [{"classifier__kernel": ["linear"], "classifier__C": [1.0]}],
        "decision_tree": [{
            "classifier__max_depth": [3],
            "classifier__min_samples_split": [2],
            "classifier__min_samples_leaf": [1],
            "classifier__criterion": ["gini"],
        }],
        "custom_random_forest": [{
            "classifier__n_estimators": [1],
            "classifier__max_depth": [3],
            "classifier__max_features": ["sqrt"],
        }],
    }
    cached_result = run_issue7_experiments(features, target, folds=folds, search_spaces=spaces)

    for balanced in (False, True):
        naive_candidates = []
        for candidate in knn_candidates:
            parameters = {name: values[0] for name, values in candidate.items()}
            recalls: list[float] = []
            precisions: list[float] = []
            for train_index, validation_index in folds:
                estimator = clone(create_experiment_estimator("knn", balanced=balanced))
                estimator.set_params(**parameters)
                estimator.fit(features.iloc[train_index], target.iloc[train_index])
                predictions = estimator.predict(features.iloc[validation_index])
                validation_target = target.iloc[validation_index]
                recalls.append(recall_score(validation_target, predictions, pos_label=1, zero_division=0))
                precisions.append(precision_score(validation_target, predictions, pos_label=1, zero_division=0))
            naive_candidates.append(
                {
                    "recall_cv_mean": float(np.mean(recalls)),
                    "precision_cv_mean": float(np.mean(precisions)),
                }
            )

        expected_index = select_best_configuration(naive_candidates)
        actual = cached_result.summary.loc[
            (cached_result.summary["model"] == "k-NN")
            & (cached_result.summary["balanced"] == balanced)
        ].iloc[0]
        expected_parameters = {
            name.removeprefix("classifier__"): values[0]
            for name, values in knn_candidates[expected_index].items()
        }
        assert json.loads(actual["best_hyperparameters"]) == expected_parameters
        assert actual["recall_cv_mean"] == pytest.approx(naive_candidates[expected_index]["recall_cv_mean"])
        assert actual["precision_cv_mean"] == pytest.approx(
            naive_candidates[expected_index]["precision_cv_mean"]
        )
