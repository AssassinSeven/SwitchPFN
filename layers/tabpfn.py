"""Frozen TabPFN-v3 inference with complete selected-feature coverage."""

import math
from importlib.metadata import version
from pathlib import Path

import numpy as np


def _check_coverage(model, width):
    indices = model.ensemble_preprocessor_.subsample_feature_indices
    if len(indices) != model.n_estimators_:
        raise RuntimeError("Unexpected TabPFN feature-subsampling layout")
    covered = set()
    for selected in indices:
        if selected is None:
            covered.update(range(width))
        else:
            selected = np.asarray(selected)
            if selected.ndim != 1 or selected.dtype.kind not in "iu":
                raise RuntimeError("Invalid TabPFN feature indices")
            if np.any((selected < 0) | (selected >= width)):
                raise RuntimeError("TabPFN feature index is out of range")
            covered.update(selected.tolist())
    if len(covered) != width:
        raise RuntimeError(
            f"TabPFN only covers {len(covered)}/{width} selected columns"
        )


class FrozenTabPFN:
    def __init__(
        self, checkpoint=None, device="cuda", random_state=2027, n_estimators=8
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.random_state = random_state
        self.n_estimators = n_estimators

    def _make_model(self, seed, width, n_samples):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        from tabpfn.preprocessing.configs import FeatureSubsamplingMethod
        from tabpfn.preprocessing.ensemble import (
            _resolve_feature_subsampling_method,
            _resolve_importance_top_k,
        )

        kwargs = dict(
            device=self.device,
            n_estimators=self.n_estimators,
            fit_mode="fit_with_cache",
            inference_precision="auto",
            memory_saving_mode="auto",
            random_state=seed,
            show_progress_bar=False,
            ignore_pretraining_limits=True,
            auto_scale_n_estimators=True,
        )
        if self.checkpoint is None:
            model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V3, **kwargs
            )
        else:
            path = Path(self.checkpoint).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            model = TabPFNClassifier(model_path=str(path), **kwargs)

        # Ensure the ensemble covers every selected feature.
        cfg = model.get_inference_config()
        budget = min(
            c.max_features_per_estimator
            for c in cfg.PREPROCESS_TRANSFORMS
            if c.max_features_per_estimator > 0
        )
        method = _resolve_feature_subsampling_method(
            method=FeatureSubsamplingMethod(cfg.FEATURE_SUBSAMPLING_METHOD),
            needs_subsampling=budget < width,
            n_samples=n_samples,
        )
        repeated = 0
        if method is FeatureSubsamplingMethod.CONSTANT_AND_BALANCED:
            repeated = min(cfg.FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT, width)
        elif method is FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE:
            repeated = _resolve_importance_top_k(
                importance_top_k_count=cfg.FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT,
                n_total_features=width,
            )
        elif method is FeatureSubsamplingMethod.RANDOM:
            raise RuntimeError(
                "Random subsampling cannot guarantee complete feature coverage"
            )
        if budget <= repeated and width > repeated:
            raise RuntimeError("No remaining TabPFN feature coverage budget")
        required = math.ceil((width - repeated) / max(1, budget - repeated))
        model.n_estimators = max(self.n_estimators, required, 1)
        return model

    def fit(self, features, labels):
        # Feature subsampling uses a version-specific TabPFN API.
        if version("tabpfn") != "8.4.0":
            raise RuntimeError("Install tabpfn==8.4.0 for this TabPFN-v3 adapter")
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        self.classes_ = np.unique(y)
        model = self._make_model(self.random_state, x.shape[1], len(x))
        self.ovr_ = False
        try:
            model.fit(x, y)
        except ValueError as exc:
            message = str(exc).lower()
            if not (
                "class" in message
                and any(s in message for s in ("limit", "maximum", "too many"))
            ):
                raise
            # Fall back to one-vs-rest if the checkpoint's class limit is exceeded.
            self.ovr_ = True
            self.models_ = []
            for i, cls in enumerate(self.classes_):
                binary = self._make_model(self.random_state + i, x.shape[1], len(x))
                binary.fit(x, (y == cls).astype(int))
                _check_coverage(binary, x.shape[1])
                self.models_.append(binary)
        else:
            _check_coverage(model, x.shape[1])
            self.models_ = [model]
        return self

    def predict_proba(self, features):
        if not self.ovr_:
            model = self.models_[0]
            order = [np.flatnonzero(model.classes_ == c)[0] for c in self.classes_]
            return np.asarray(model.predict_proba(features), dtype=np.float64)[:, order]
        scores = []
        for model in self.models_:
            positive = np.flatnonzero(model.classes_ == 1)[0]
            scores.append(model.predict_proba(features)[:, positive])
        scores = np.clip(np.column_stack(scores), 1e-8, None)
        return scores / scores.sum(axis=1, keepdims=True)
