"""SwitchPFN: shared switching dynamics for frozen time-series classification."""

import random
from dataclasses import replace

import numpy as np
from layers.features import FeatureExtractor
from layers.tabpfn import FrozenTabPFN
from sklearn.preprocessing import LabelEncoder

from .config import Config


class Model:
    """A fit/predict interface for variable-length [time, channels] sequences.

    fit() estimates the representation and prepares a labeled TabPFN context.
    It never fine-tunes the neural weights. transform() always returns query
    features; context_features_ stores the excluded-bank training features.
    """

    def __init__(self, checkpoint=None, device="cuda", random_state=2027, config=None):
        self.config = replace(config or Config(), random_state=random_state)
        self.checkpoint = checkpoint
        self.device = device

    def fit(self, X, y):
        import torch

        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)
        self.label_encoder_ = LabelEncoder().fit(y)
        encoded = self.label_encoder_.transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.features_ = FeatureExtractor(self.config)
        self.context_features_ = self.features_.fit_transform(X, encoded)
        self.predictor_ = FrozenTabPFN(
            checkpoint=self.checkpoint,
            device=self.device,
            random_state=self.config.random_state,
            n_estimators=self.config.n_estimators,
        ).fit(self.context_features_, encoded)
        return self

    def transform(self, X):
        if not hasattr(self, "predictor_"):
            raise RuntimeError("Call fit(X_train, y_train) first")
        return self.features_.transform(X)

    def predict_proba(self, X):
        """Columns follow the order in classes_."""
        features = self.transform(X)
        return self.predictor_.predict_proba(features)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
