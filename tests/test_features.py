"""Numerical tests for feature extraction and UEA data loading."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from data_provider.data_loader import load_uea_dataset
from layers.features import FeatureExtractor
from layers.logsignature import logsignature
from models.config import Config
from threadpoolctl import threadpool_limits


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)
        rng = np.random.default_rng(10)
        cls.rows = [rng.normal(size=(32 + i, 3)) for i in range(9)]
        cls.rows[0][3:5, 1] = np.nan
        cls.labels = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2])
        cls.extractor = FeatureExtractor(Config())
        cls.context = cls.extractor.fit_transform(cls.rows, cls.labels)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_selected_features_and_standardization(self):
        self.assertEqual(self.context.shape, (9, 1024))
        self.assertTrue(np.isfinite(self.context).all())
        np.testing.assert_allclose(self.context.mean(axis=0), 0, atol=1e-9)
        selected = self.extractor.selected_indices_
        np.testing.assert_array_equal(selected[:256], np.arange(256))
        self.assertEqual(len(np.unique(selected)), 1024)

    def test_singleton_bank_excludes_self(self):
        dynamics = self.extractor.dynamics_
        bank = dynamics._operator_bank(exclude_index=0)
        for h in dynamics.config.operator_horizons:
            remaining = dynamics._sum_stats(dynamics.sequence_stats_[h][1:])
            expected = dynamics._solve_stats(remaining, remaining)
            actual = bank[(0, h)]
            np.testing.assert_allclose(actual.operator, expected.operator, atol=1e-10)
            np.testing.assert_allclose(actual.intercept, expected.intercept, atol=1e-10)
            self.assertEqual(actual.count, 8)

    def test_query_does_not_refit_and_is_batch_independent(self):
        before = pickle.dumps(self.extractor)
        joint = self.extractor.transform(self.rows[:2])
        alone = self.extractor.transform(self.rows[:1])
        np.testing.assert_array_equal(joint[:1], alone)
        self.assertEqual(before, pickle.dumps(self.extractor))
        self.assertGreater(np.max(np.abs(joint - self.context[:2])), 1e-6)

    def test_input_errors(self):
        with self.assertRaises(RuntimeError):
            FeatureExtractor(Config()).transform(self.rows)
        with self.assertRaises(ValueError):
            self.extractor.transform([np.ones((10, 2))])
        with self.assertRaises(ValueError):
            self.extractor.transform([np.full((10, 3), np.inf)])
        with self.assertRaises(ValueError):
            FeatureExtractor(Config()).fit_transform(self.rows, np.zeros(9))


class PathTests(unittest.TestCase):
    def test_straight_path_has_only_first_level(self):
        delta = np.array([[0.2, -0.3, 0.7]])
        features = logsignature(delta, 3)
        np.testing.assert_allclose(features[:3], delta[0], atol=1e-13)
        np.testing.assert_allclose(features[3:], 0, atol=1e-13)

    def test_inverse_path_negates_logsignature(self):
        increments = np.random.default_rng(8).normal(size=(15, 4))
        np.testing.assert_allclose(
            logsignature(-increments[::-1], 3),
            -logsignature(increments, 3),
            atol=1e-11,
            rtol=1e-11,
        )


class DataTests(unittest.TestCase):
    def test_official_split_and_training_label_order(self):
        header = "@problemName Tiny\n@timestamps false\n@classLabel true a b\n@data\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "Tiny_TRAIN.ts"
            test = root / "Tiny_TEST.ts"
            train.write_text(header + "1,?,3:4,5,6:b\n2,3:4,5:a\n")
            test.write_text(header + "3,4,5:6,7,8:a\n")
            data = load_uea_dataset(root, "Tiny")
            np.testing.assert_array_equal(data.classes, ["a", "b"])
            np.testing.assert_array_equal(data.y_train, [1, 0])
            np.testing.assert_array_equal(data.x_train[0][:, 0], [1, 2, 3])
            self.assertEqual(data.x_train[1].shape, (2, 2))
            test.write_text(header + "3,4:6,7:unseen\n")
            with self.assertRaises(ValueError):
                load_uea_dataset(root, "Tiny")


if __name__ == "__main__":
    unittest.main()
