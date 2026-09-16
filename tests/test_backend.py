"""Tests for probability ordering, class limits, and feature coverage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from layers.tabpfn import FrozenTabPFN, _check_coverage


class ClassifierDouble:
    def __init__(self, class_limit=False):
        self.class_limit = class_limit

    def fit(self, x, y):
        if self.class_limit and len(np.unique(y)) > 2:
            raise ValueError("maximum number of classes exceeded")
        self.classes_ = np.unique(y)[::-1]
        self.n_estimators_ = 1
        self.ensemble_preprocessor_ = SimpleNamespace(subsample_feature_indices=[None])

    def predict_proba(self, x):
        # Known probabilities in reverse class order, independent of any model.
        row = np.arange(1, len(self.classes_) + 1, dtype=float)
        return np.tile(row / row.sum(), (len(x), 1))


class BackendTests(unittest.TestCase):
    def test_native_probability_columns_follow_classes(self):
        with (
            patch("layers.tabpfn.version", return_value="8.4.0"),
            patch.object(FrozenTabPFN, "_make_model", return_value=ClassifierDouble()),
        ):
            backend = FrozenTabPFN().fit(np.ones((6, 5)), np.repeat([0, 1, 2], 2))
            probability = backend.predict_proba(np.ones((2, 5)))
        np.testing.assert_allclose(probability[0], [3 / 6, 2 / 6, 1 / 6])

    def test_native_class_limit_uses_frozen_binary_models(self):
        def factory(*args):
            return ClassifierDouble(class_limit=True)

        with (
            patch("layers.tabpfn.version", return_value="8.4.0"),
            patch.object(FrozenTabPFN, "_make_model", side_effect=factory),
        ):
            backend = FrozenTabPFN().fit(np.ones((6, 5)), np.repeat([0, 1, 2], 2))
            probability = backend.predict_proba(np.ones((2, 5)))
        self.assertEqual(len(backend.models_), 3)
        np.testing.assert_allclose(probability.sum(axis=1), 1)

    def test_missing_feature_coverage_is_rejected(self):
        model = SimpleNamespace(
            n_estimators_=2,
            ensemble_preprocessor_=SimpleNamespace(
                subsample_feature_indices=[[0, 1], [1, 2]]
            ),
        )
        with self.assertRaises(RuntimeError):
            _check_coverage(model, 4)
        model.ensemble_preprocessor_.subsample_feature_indices[1] = [2, 3]
        _check_coverage(model, 4)


if __name__ == "__main__":
    unittest.main()
