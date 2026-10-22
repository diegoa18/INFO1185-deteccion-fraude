import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold

from src.validation import materialize_stratified_folds


def test_duplicate_predictors_never_cross_folds():
    # Duplicados de ambas clases, con índices distintos y no consecutivos.
    features = pd.DataFrame({"V1": np.repeat(np.arange(50), 2)})
    features.index = np.arange(100) * 7
    target = pd.Series(np.repeat([0, 1], [80, 20]), index=features.index)
    folds = materialize_stratified_folds(features, target)
    repeated = materialize_stratified_folds(features, target)
    assert len(folds) == 5
    np.testing.assert_array_equal(
        np.sort(np.concatenate([valid for _, valid in folds])), np.arange(100)
    )
    for (train, valid), (train_again, valid_again) in zip(folds, repeated):
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(valid, valid_again)
        assert set(target.iloc[train]) == set(target.iloc[valid]) == {0, 1}
        assert not set(train) & set(valid)
        assert not set(features.iloc[train].V1) & set(features.iloc[valid].V1)


@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('duplicates', [False, True])
def test_explicit_cv_configuration_is_preserved(shuffle, duplicates):
    features = pd.DataFrame({'V1': np.repeat(np.arange(60), 2) if duplicates else np.arange(120)})
    target = pd.Series(np.repeat([0, 1], [80, 40]))
    options = dict(n_splits=4, shuffle=shuffle, random_state=19 if shuffle else None)
    cv = StratifiedKFold(**options)
    actual = materialize_stratified_folds(features, target, cv=cv)
    if duplicates:
        groups = pd.factorize(pd.MultiIndex.from_frame(features))[0]
        expected = StratifiedGroupKFold(**options).split(features, target, groups)
    else:
        expected = cv.split(features, target)
    for (t, v), (et, ev) in zip(actual, expected):
        np.testing.assert_array_equal(t, et)
        np.testing.assert_array_equal(v, ev)
    assert len(actual) == 4


def test_groups_ignore_target_and_row_index():
    features = pd.DataFrame({'V1': np.repeat(np.arange(20), 2)}, index=[99] * 40)
    target = pd.Series([0, 1] * 20, index=features.index)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train, valid in materialize_stratified_folds(features, target, cv=cv):
        assert not set(features.iloc[train].V1) & set(features.iloc[valid].V1)
        assert set(target.iloc[train]) == set(target.iloc[valid]) == {0, 1}


def test_insufficient_class_groups_raise_clear_error():
    features = pd.DataFrame({'V1': list(range(20)) + [100] * 10})
    target = pd.Series([0] * 20 + [1] * 10)
    with pytest.raises(ValueError, match='5 folds válidos.*clase 1.*1 grupos'):
        materialize_stratified_folds(features, target)


def test_unsupported_cv_is_not_silently_ignored():
    with pytest.raises(TypeError, match='cv debe ser'):
        materialize_stratified_folds(pd.DataFrame({'V1': range(20)}), pd.Series([0, 1] * 10), cv=KFold())


def test_group_heuristic_missing_class_raises_clear_error(monkeypatch):
    def invalid_split(self, X, y, groups):
        yield np.arange(10, 40), np.arange(10)
    monkeypatch.setattr(StratifiedGroupKFold, 'split', invalid_split)
    features = pd.DataFrame({'V1': np.repeat(np.arange(20), 2)})
    with pytest.raises(ValueError, match='fold 1 no contiene ambas clases'):
        materialize_stratified_folds(features, pd.Series([0] * 20 + [1] * 20))
