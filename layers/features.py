"""Multiscale trajectory statistics and training-only feature selection."""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .dynamics import DynamicsEncoder
from .logsignature import multiscale_logsignature


def _finite(values: np.ndarray, clip: float = 1e5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(values, nan=0.0, posinf=clip, neginf=-clip).clip(-clip, clip)


def _segments(length: int, scales: tuple[int, ...]):
    for count in scales:
        if count < 1:
            raise ValueError("segment counts must be positive")
        for index in np.array_split(np.arange(length), count):
            if index.size:
                yield int(count), index


def _autocorrelation(values: np.ndarray, lag: int, eps: float = 1e-8) -> np.ndarray:
    if len(values) <= lag:
        return np.zeros(values.shape[1], dtype=np.float64)
    left = values[:-lag] - values[:-lag].mean(axis=0)
    right = values[lag:] - values[lag:].mean(axis=0)
    numerator = np.sum(left * right, axis=0)
    denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
    return numerator / np.maximum(denominator, eps)


def _spectral(values: np.ndarray, bands: int, eps: float = 1e-8) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    total = np.maximum(power.sum(axis=0, keepdims=True), eps)
    normalized = power / total
    chunks = np.array_split(np.arange(power.shape[0]), bands)
    band_rows = [
        normalized[index].sum(axis=0) if index.size else np.zeros(values.shape[1])
        for index in chunks
    ]
    frequency = np.linspace(0.0, 1.0, power.shape[0])[:, None]
    centroid = np.sum(frequency * normalized, axis=0)
    entropy = -np.sum(normalized * np.log(np.maximum(normalized, eps)), axis=0)
    return np.concatenate([*band_rows, centroid, entropy])


def _operator_spectrum(
    values: np.ndarray, lag: int, ridge: float, width: int
) -> np.ndarray:
    if len(values) <= lag:
        return np.zeros(4 * width + 4, dtype=np.float64)
    x, y = values[:-lag], values[lag:]
    scale = max(float(np.trace(x.T @ x)) / max(1, x.shape[1] * len(x)), 1e-8)
    gram = x.T @ x + np.eye(x.shape[1]) * ridge * scale
    try:
        operator = np.linalg.solve(gram, x.T @ y)
    except np.linalg.LinAlgError:
        operator = np.linalg.pinv(gram) @ (x.T @ y)
    prediction = x @ operator
    relative_error = np.sqrt(np.mean((prediction - y) ** 2, axis=0)) / np.maximum(
        np.sqrt(np.mean(y**2, axis=0)), 1e-8
    )
    singular = np.linalg.svd(operator, compute_uv=False)
    eigen = np.linalg.eigvals(operator)

    def pad(row: np.ndarray) -> np.ndarray:
        row = np.asarray(row, dtype=np.float64).ravel()[:width]
        return np.pad(row, (0, max(0, width - len(row))))

    return _finite(
        np.concatenate(
            [
                pad(singular),
                pad(np.sort(np.abs(eigen))[::-1]),
                pad(np.sort(eigen.real)[::-1]),
                pad(relative_error),
                [
                    np.linalg.norm(operator),
                    np.trace(operator).real,
                    np.mean(relative_error),
                    np.max(relative_error),
                ],
            ]
        )
    )


def _trajectory_summary(
    latent: np.ndarray,
    normalized: np.ndarray,
    config,
) -> np.ndarray:
    q = min(config.latent_rank, latent.shape[1])
    z = _finite(latent[:, :q])
    rows: list[np.ndarray] = []
    for _scale, index in _segments(len(z), config.segment_scales):
        segment = z[index]
        velocity = (
            np.diff(segment, axis=0) if len(segment) > 1 else np.zeros_like(segment)
        )
        quantiles = np.quantile(segment, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0).ravel()
        rows.extend(
            [
                segment.mean(axis=0),
                segment.std(axis=0),
                segment.min(axis=0),
                segment.max(axis=0),
                quantiles,
                segment[0],
                segment[-1],
                segment[-1] - segment[0],
                np.mean(np.abs(segment), axis=0),
                np.sqrt(np.mean(segment**2, axis=0)),
                np.sum(np.abs(velocity), axis=0),
                _spectral(segment, config.spectral_bands),
            ]
        )
        for lag in config.autocorr_lags:
            rows.append(_autocorrelation(segment, lag))
        centered = segment - segment.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(1, len(segment) - 1)
        rows.append(covariance[np.triu_indices(q)])
        for lag in config.cross_lags:
            if len(segment) > lag:
                rows.append(
                    (segment[:-lag].T @ segment[lag:] / (len(segment) - lag)).ravel()
                )
            else:
                rows.append(np.zeros(q * q))
    spectrum = np.fft.rfft(z - z.mean(axis=0, keepdims=True), axis=0)
    k = min(config.low_frequency_bins, spectrum.shape[0])
    low = spectrum[:k]
    rows.extend([low.real.ravel(), low.imag.ravel(), np.abs(low).ravel()])
    for lag in config.cross_lags:
        rows.append(_operator_spectrum(z, lag, config.operator_ridge, min(16, q)))

    # Cross-channel operator singular spectra retain interactions without a
    # channel-count-squared feature explosion (important for Heartbeat/SCP1).
    raw = _finite(normalized)
    operator_width = min(16, raw.shape[1])
    for lag in config.cross_lags:
        rows.append(_operator_spectrum(raw, lag, config.operator_ridge, operator_width))
    covariance = np.cov(raw, rowvar=False)
    covariance = np.atleast_2d(covariance)
    eigen = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
    eigen = eigen[:operator_width]
    eigen = np.pad(eigen, (0, max(0, operator_width - len(eigen))))
    rows.append(eigen / max(float(eigen.sum()), 1e-8))
    return _finite(np.concatenate([np.ravel(row) for row in rows]))


def _fisher_order(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = _finite(features)
    y = np.asarray(labels)
    overall = x.mean(axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    within = np.zeros(x.shape[1], dtype=np.float64)
    for cls in np.unique(y):
        block = x[y == cls]
        center = block.mean(axis=0)
        between += len(block) * (center - overall) ** 2
        within += np.sum((block - center) ** 2, axis=0)
    score = between / np.maximum(within, 1e-12)
    score[~np.isfinite(score)] = -np.inf
    return np.lexsort((np.arange(x.shape[1]), -score))


class FeatureExtractor:
    """Build context rows with excluded class banks and query rows with full banks."""

    def __init__(self, config):
        self.config = config

    def _prepare(self, series):
        rows = [np.asarray(row, dtype=np.float64) for row in series]
        if not rows:
            raise ValueError("Provide at least one time series")
        for row in rows:
            if row.ndim != 2 or row.shape[0] < 2 or row.shape[1] < 1:
                raise ValueError(
                    "Each sequence must have shape [time >= 2, channels >= 1]"
                )
            if np.isinf(row).any():
                raise ValueError("Sequences may contain NaN, but not infinity")
            if row.shape[1] != rows[0].shape[1]:
                raise ValueError("All sequences must have the same channel count")
        return rows

    def _path(self, latent, view):
        values = _finite(latent[:, : min(self.config.path_dim, latent.shape[1])])
        if view == "velocity":
            values = np.gradient(values, axis=0)
        elif view != "latent":
            raise ValueError(f"Unsupported path view: {view}")
        path = np.column_stack([np.linspace(0, 1, len(values)), values])
        return _finite(
            multiscale_logsignature(
                np.diff(path, axis=0),
                self.config.path_order,
                self.config.path_partitions,
            )
        )

    def _raw_features(self, rows, context=False):
        output = []
        for i, row in enumerate(rows):
            z = self.dynamics_._latent(row)
            output.append(
                np.concatenate(
                    [
                        self.dynamics_.row(z, context_index=i if context else None),
                        *(self._path(z, view) for view in self.config.path_views),
                        _trajectory_summary(
                            z, self.dynamics_._normalize(row), self.config
                        ),
                    ]
                )
            )
        return _finite(np.stack(output))

    def fit_transform(self, series, labels):
        """Fit on training data and return its leave-one-sequence-out context."""
        rows = self._prepare(series)
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(rows):
            raise ValueError("Provide one label per training sequence")
        encoded = LabelEncoder().fit_transform(labels)
        if len(np.unique(encoded)) < 2:
            raise ValueError("Classification requires at least two training classes")
        if self.config.feature_budget < 1 or not 0 <= self.config.prefix_fraction <= 1:
            raise ValueError("Invalid feature budget or prefix fraction")
        self.dynamics_ = DynamicsEncoder(self.config).fit(rows, encoded)
        raw = self._raw_features(rows, context=True)
        width = min(self.config.feature_budget, raw.shape[1])
        prefix = np.arange(int(width * self.config.prefix_fraction))
        ranked = _fisher_order(raw, encoded)
        ranked = ranked[~np.isin(ranked, prefix)][: width - len(prefix)]
        self.selected_indices_ = np.sort(np.concatenate([prefix, ranked]))
        selected = raw[:, self.selected_indices_]
        self.center_ = selected.mean(axis=0)
        self.scale_ = np.maximum(selected.std(axis=0), self.config.eps)
        return _finite((selected - self.center_) / self.scale_)

    def transform(self, series):
        """Transform queries using the complete training banks and frozen selector."""
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("Call fit_transform on the training split first")
        raw = self._raw_features(self._prepare(series))
        return _finite((raw[:, self.selected_indices_] - self.center_) / self.scale_)
