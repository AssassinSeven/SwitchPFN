"""Shared coordinates, local regimes, and leave-one-sequence-out class banks."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def finite_matrix(x, clip=1e4):
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clip(-clip, clip)


@dataclass(frozen=True)
class _SufficientStats:
    gram: np.ndarray
    cross: np.ndarray
    yy: np.ndarray
    count: int

    def __add__(self, other: "_SufficientStats") -> "_SufficientStats":
        return _SufficientStats(
            self.gram + other.gram,
            self.cross + other.cross,
            self.yy + other.yy,
            self.count + other.count,
        )

    def __sub__(self, other: "_SufficientStats") -> "_SufficientStats":
        return _SufficientStats(
            self.gram - other.gram,
            self.cross - other.cross,
            self.yy - other.yy,
            self.count - other.count,
        )


@dataclass(frozen=True)
class _AffineModel:
    operator: np.ndarray
    intercept: np.ndarray
    sigma: float
    count: int


def _fill_nan(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            "Each time series must have shape [time, channels] with time>=2"
        )
    output = values.copy()
    grid = np.arange(values.shape[0])
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        if finite.all():
            continue
        if not finite.any():
            output[:, column] = 0.0
        else:
            output[:, column] = np.interp(grid, grid[finite], values[finite, column])
    return output


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    if values.shape[0] == length:
        return values
    old = np.linspace(0.0, 1.0, values.shape[0])
    new = np.linspace(0.0, 1.0, length)
    return np.stack(
        [np.interp(new, old, values[:, col]) for col in range(values.shape[1])], axis=1
    )


def _entropy(values: np.ndarray, eps: float) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0]
    return float(-np.sum(positive * np.log(positive + eps)))


def _state(values):
    derivative = np.diff(values, axis=0, prepend=values[:1])
    return np.concatenate([values, derivative], axis=1)


class DynamicsEncoder:
    """Fit shared objects on training sequences; query transforms never refit."""

    def __init__(self, config):
        self.config = config

    def _normalize(self, sequence: np.ndarray) -> np.ndarray:
        values = _fill_nan(sequence)
        if values.shape[1] != self.n_channels_:
            raise ValueError("All sequences must share the fold-TRAIN channel count")
        return (values - self.train_mean_) / np.maximum(
            self.train_scale_, self.config.eps
        )

    def _latent(self, sequence: np.ndarray) -> np.ndarray:
        state = _state(self._normalize(sequence))
        latent = self.pca_.transform(state)
        return finite_matrix(
            _resample(latent, self.config.target_length), self.config.clip_value
        )

    def _window_ranges(self) -> list[tuple[int, int]]:
        length = min(max(3, self.config.window_length), self.config.target_length)
        stride = max(1, self.config.window_stride)
        starts = list(range(0, max(1, self.config.target_length - length + 1), stride))
        last = self.config.target_length - length
        if starts[-1] != last:
            starts.append(last)
        return [(start, start + length) for start in starts]

    def _local_affine_raw(self, window: np.ndarray) -> tuple[np.ndarray, float]:
        x, y = window[:-1], window[1:]
        augmented = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        gram = augmented.T @ augmented / max(1, x.shape[0])
        cross = augmented.T @ y / max(1, x.shape[0])
        scale = max(float(np.trace(gram)) / max(1, gram.shape[0]), self.config.eps)
        penalty = np.eye(gram.shape[0]) * self.config.operator_ridge * scale
        penalty[-1, -1] = self.config.eps
        try:
            weights = np.linalg.solve(gram + penalty, cross)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(gram + penalty) @ cross
        prediction = augmented @ weights
        residual_by_coordinate = np.sqrt(np.mean((prediction - y) ** 2, axis=0))
        target_scale = np.maximum(np.sqrt(np.mean(y**2, axis=0)), self.config.eps)
        normalized_residual = residual_by_coordinate / target_scale
        operator = weights[:-1]
        intercept = weights[-1]
        raw = np.concatenate(
            [
                (operator - np.eye(self.latent_rank_)).ravel(),
                intercept,
                normalized_residual,
            ]
        )
        return finite_matrix(raw[None, :], self.config.clip_value)[0], float(
            np.mean(normalized_residual)
        )

    def _raw_windows(self, latent):
        raw, dispersion, auxiliary = [], [], []
        for start, end in self.window_ranges_:
            window = latent[start:end]
            row, residual = self._local_affine_raw(window)
            velocity = np.diff(window, axis=0)
            raw.append(row)
            dispersion.append(residual)
            auxiliary.append(
                np.concatenate(
                    [
                        [residual],
                        velocity.mean(axis=0),
                        velocity.std(axis=0),
                    ]
                )
            )
        return np.stack(raw), np.asarray(dispersion), np.stack(auxiliary)

    def _fit_projection(
        self, rows: np.ndarray, requested_dim: int, salt: int
    ) -> tuple[PCA, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.random_state + salt)
        if rows.shape[0] > self.config.codebook_fit_windows:
            chosen = np.sort(
                rng.choice(
                    rows.shape[0], self.config.codebook_fit_windows, replace=False
                )
            )
            fit_rows = rows[chosen]
        else:
            fit_rows = rows
        mean = np.mean(fit_rows, axis=0)
        scale = np.std(fit_rows, axis=0)
        standardized = (fit_rows - mean) / np.maximum(scale, self.config.eps)
        dimension = min(
            int(requested_dim), standardized.shape[0], standardized.shape[1]
        )
        if dimension < 1:
            raise ValueError("Cannot fit a non-empty local descriptor projection")
        model = PCA(
            n_components=dimension,
            svd_solver="full",
            random_state=self.config.random_state,
        )
        model.fit(standardized)
        return model, mean, scale

    @staticmethod
    def _canonicalize_codebook(model: MiniBatchKMeans) -> None:
        centers = np.asarray(model.cluster_centers_)
        keys = tuple(centers[:, index] for index in reversed(range(centers.shape[1])))
        order = np.lexsort(keys)
        model.cluster_centers_ = centers[order]

    def _fit_codebook(self, descriptors: np.ndarray, salt: int):
        rng = np.random.default_rng(self.config.random_state + salt)
        if descriptors.shape[0] > self.config.codebook_fit_windows:
            chosen = np.sort(
                rng.choice(
                    descriptors.shape[0],
                    self.config.codebook_fit_windows,
                    replace=False,
                )
            )
            fit_rows = descriptors[chosen]
        else:
            fit_rows = descriptors
        mean = np.mean(fit_rows, axis=0)
        scale = np.std(fit_rows, axis=0)
        standardized = (fit_rows - mean) / np.maximum(scale, self.config.eps)
        if standardized.shape[0] < self.config.n_states:
            raise ValueError("Fewer fold-TRAIN windows than requested aligned states")
        model = MiniBatchKMeans(
            n_clusters=self.config.n_states,
            random_state=self.config.random_state + salt,
            batch_size=min(2048, max(32, standardized.shape[0])),
            n_init=5,
            reassignment_ratio=0.0,
        ).fit(standardized)
        self._canonicalize_codebook(model)
        train_distance = model.transform(standardized)
        temperature = max(
            float(np.median(np.min(train_distance, axis=1))), self.config.eps
        )
        return model, mean, scale, temperature

    def _project_windows(
        self, latent: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw, dispersion, auxiliary = self._raw_windows(latent)
        projector, raw_mean, raw_scale = (
            self.operator_pca_,
            self.operator_raw_mean_,
            self.operator_raw_scale_,
        )
        codebook, desc_mean, desc_scale, temperature = (
            self.dynamic_codebook_,
            self.dynamic_desc_mean_,
            self.dynamic_desc_scale_,
            self.dynamic_temperature_,
        )
        code = projector.transform(
            (raw - raw_mean) / np.maximum(raw_scale, self.config.eps)
        )
        descriptor = np.concatenate([code, auxiliary], axis=1)
        standardized = (descriptor - desc_mean) / np.maximum(
            desc_scale, self.config.eps
        )
        distance = codebook.transform(standardized)
        logits = -distance / max(temperature, self.config.eps)
        logits -= np.max(logits, axis=1, keepdims=True)
        soft = np.exp(logits)
        soft /= np.maximum(np.sum(soft, axis=1, keepdims=True), self.config.eps)
        hard = np.argmax(soft, axis=1).astype(int)
        return code, dispersion, soft, hard

    def _empty_stats(self) -> _SufficientStats:
        width = self.latent_rank_ + 1
        return _SufficientStats(
            np.zeros((width, width), dtype=np.float64),
            np.zeros((width, self.latent_rank_), dtype=np.float64),
            np.zeros((self.latent_rank_, self.latent_rank_), dtype=np.float64),
            0,
        )

    def _sequence_stats(self, latent: np.ndarray, horizon: int) -> _SufficientStats:
        source = latent
        if source.shape[0] <= horizon:
            return self._empty_stats()
        x, y = source[:-horizon], source[horizon:]
        augmented = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        count = x.shape[0]
        return _SufficientStats(
            augmented.T @ augmented / count,
            augmented.T @ y / count,
            y.T @ y / count,
            1,
        )

    def _sum_stats(self, rows: list[_SufficientStats]) -> _SufficientStats:
        output = self._empty_stats()
        for row in rows:
            output = output + row
        return output

    def _solve_stats(
        self, class_stats: _SufficientStats, global_stats: _SufficientStats
    ) -> _AffineModel:
        if global_stats.count <= 0:
            zero = np.zeros((self.latent_rank_, self.latent_rank_), dtype=np.float64)
            return _AffineModel(zero, np.zeros(self.latent_rank_), 1.0, 0)
        if class_stats.count <= 0:
            class_stats = global_stats
        weight = max(float(class_stats.count), self.config.eps)
        gram = class_stats.gram / weight
        cross = class_stats.cross / weight
        yy = class_stats.yy / weight
        scale = max(float(np.trace(gram)) / max(1, gram.shape[0]), self.config.eps)
        penalty = np.eye(gram.shape[0]) * self.config.operator_ridge * scale
        penalty[-1, -1] = self.config.eps
        try:
            weights = np.linalg.solve(gram + penalty, cross)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(gram + penalty) @ cross
        residual_second = (
            yy - weights.T @ cross - cross.T @ weights + weights.T @ gram @ weights
        )
        sigma2 = max(
            float(np.trace(residual_second)) / max(1, self.latent_rank_),
            self.config.eps,
        )
        return _AffineModel(
            weights[:-1], weights[-1], float(np.sqrt(sigma2)), class_stats.count
        )

    def _operator_bank(
        self, exclude_index: int | None
    ) -> dict[tuple[int, int], _AffineModel]:
        per_sequence = self.sequence_stats_
        output: dict[tuple[int, int], _AffineModel] = {}
        for horizon in self.config.operator_horizons:
            rows = per_sequence[int(horizon)]
            global_stats = self.global_stats_[int(horizon)]
            if exclude_index is not None:
                global_stats = global_stats - rows[exclude_index]
            for cls in self.classes_:
                indices = self.class_indices_[int(cls)]
                class_stats = self.class_stats_[(int(cls), int(horizon))]
                if exclude_index is not None and exclude_index in indices:
                    class_stats = class_stats - rows[exclude_index]
                output[(int(cls), int(horizon))] = self._solve_stats(
                    class_stats, global_stats
                )
        return output

    def fit(self, train_series, labels):
        filled = [_fill_nan(row) for row in train_series]
        self.n_channels_ = filled[0].shape[1]
        pooled = np.concatenate(filled, axis=0)
        self.train_mean_ = pooled.mean(axis=0, keepdims=True)
        self.train_scale_ = pooled.std(axis=0, keepdims=True)

        rng = np.random.default_rng(self.config.random_state)
        quota = max(2, self.config.pca_fit_rows // len(filled))
        sampled = []
        for values in filled:
            state = _state(
                (values - self.train_mean_)
                / np.maximum(self.train_scale_, self.config.eps)
            )
            if len(state) > quota:
                state = state[np.sort(rng.choice(len(state), quota, replace=False))]
            sampled.append(state)
        pca_rows = np.concatenate(sampled, axis=0)
        self.latent_rank_ = min(self.config.latent_rank, *pca_rows.shape)
        self.pca_ = PCA(n_components=self.latent_rank_, svd_solver="full").fit(pca_rows)
        self.window_ranges_ = self._window_ranges()
        latents = [self._latent(row) for row in train_series]

        windows = [self._raw_windows(z) for z in latents]
        raw = np.concatenate([row[0] for row in windows])
        auxiliary = np.concatenate([row[2] for row in windows])
        self.operator_pca_, self.operator_raw_mean_, self.operator_raw_scale_ = (
            self._fit_projection(raw, self.config.operator_code_dim, 101)
        )
        self.operator_code_dim_ = self.operator_pca_.n_components_
        code = self.operator_pca_.transform(
            (raw - self.operator_raw_mean_)
            / np.maximum(self.operator_raw_scale_, self.config.eps)
        )
        descriptor = np.concatenate([code, auxiliary], axis=1)
        (
            self.dynamic_codebook_,
            self.dynamic_desc_mean_,
            self.dynamic_desc_scale_,
            self.dynamic_temperature_,
        ) = self._fit_codebook(descriptor, 303)

        self.classes_ = np.unique(labels)
        self.class_indices_ = {
            int(c): set(np.flatnonzero(labels == c)) for c in self.classes_
        }
        self.sequence_stats_ = {
            h: [self._sequence_stats(z, h) for z in latents]
            for h in self.config.operator_horizons
        }
        self.global_stats_, self.class_stats_ = {}, {}
        for h, rows in self.sequence_stats_.items():
            self.global_stats_[h] = self._sum_stats(rows)
            for c in self.classes_:
                self.class_stats_[(int(c), h)] = self._sum_stats(
                    [rows[i] for i in sorted(self.class_indices_[int(c)])]
                )
        self.full_operator_bank_ = self._operator_bank(exclude_index=None)
        return self

    def _local_block(
        self, code: np.ndarray, dispersion: np.ndarray, soft: np.ndarray
    ) -> np.ndarray:
        k, q = self.config.n_states, self.operator_code_dim_
        means = np.zeros((k, q), dtype=np.float64)
        stds = np.zeros((k, q), dtype=np.float64)
        residual_mean = np.zeros(k, dtype=np.float64)
        residual_std = np.zeros(k, dtype=np.float64)
        for state in range(k):
            weight = soft[:, state]
            total = float(np.sum(weight))
            if total <= self.config.eps:
                continue
            mean = np.sum(code * weight[:, None], axis=0) / total
            variance = np.sum((code - mean) ** 2 * weight[:, None], axis=0) / total
            means[state] = mean
            stds[state] = np.sqrt(np.maximum(variance, 0.0))
            rmean = float(np.sum(dispersion * weight) / total)
            residual_mean[state] = rmean
            residual_std[state] = float(
                np.sqrt(np.sum((dispersion - rmean) ** 2 * weight) / total)
            )
        pyramid = [
            np.mean(code[index], axis=0)
            for partitions in (1, 2, 4)
            for index in np.array_split(np.arange(code.shape[0]), partitions)
        ]
        output = np.concatenate(
            [
                means.ravel(),
                stds.ravel(),
                np.concatenate(pyramid),
                residual_mean,
                residual_std,
            ]
        )
        return output

    def _transition_block(self, soft: np.ndarray, hard: np.ndarray) -> np.ndarray:
        occupancy = np.mean(soft, axis=0)
        dwell_values = [[] for _ in range(self.config.n_states)]
        if hard.size:
            start = 0
            for index in range(1, hard.size + 1):
                if index == hard.size or hard[index] != hard[start]:
                    dwell_values[int(hard[start])].append(index - start)
                    start = index
        dwell_mean = np.asarray(
            [
                np.mean(values) / max(1, hard.size) if values else 0.0
                for values in dwell_values
            ]
        )
        dwell_std = np.asarray(
            [
                np.std(values) / max(1, hard.size) if values else 0.0
                for values in dwell_values
            ]
        )
        transitions = []
        for lag in self.config.transition_lags:
            if soft.shape[0] <= lag:
                joint = np.outer(occupancy, occupancy)
            else:
                joint = np.einsum("ti,tj->ij", soft[:-lag], soft[lag:]) / (
                    soft.shape[0] - lag
                )
                joint /= max(float(np.sum(joint)), self.config.eps)
            transitions.append(joint)
        first = transitions[0]
        switch_rate = float(np.mean(hard[1:] != hard[:-1])) if hard.size > 1 else 0.0
        extras = np.asarray(
            [
                switch_rate,
                _entropy(occupancy, self.config.eps),
                float(np.trace(first)),
                _entropy(first.ravel(), self.config.eps),
            ]
        )
        return np.concatenate(
            [
                occupancy,
                dwell_mean,
                dwell_std,
                *(joint.ravel() for joint in transitions),
                extras,
            ]
        )

    def _operator_features(
        self,
        latent: np.ndarray,
        bank: dict[tuple[int, int], _AffineModel],
    ) -> np.ndarray:
        horizons = self.config.operator_horizons
        per_class = []
        for cls in self.classes_:
            horizon_scores = []
            residuals: dict[int, np.ndarray] = {}
            sigmas: dict[int, float] = {}
            for horizon in horizons:
                model = bank[(int(cls), int(horizon))]
                y = latent[horizon:]
                residual = latent[:-horizon] @ model.operator + model.intercept - y
                sigma = model.sigma
                normalized = np.sqrt(np.mean(residual**2)) / max(
                    float(np.sqrt(np.mean(y**2))), self.config.eps
                )
                horizon_scores.append(float(np.log1p(normalized)))
                residuals[int(horizon)] = residual
                sigmas[int(horizon)] = sigma

            reference_horizon = int(min(horizons))
            reference_residual = residuals[reference_horizon]
            residual_energy = np.mean(reference_residual**2, axis=1)
            variance = float(np.log1p(np.var(residual_energy)))
            sigma2 = max(sigmas[reference_horizon] ** 2, self.config.eps)
            nll = float(
                0.5 * np.mean(residual_energy / sigma2 + np.log(2.0 * np.pi * sigma2))
            )
            per_class.append(
                {
                    "scores": np.asarray(horizon_scores, dtype=np.float64),
                    "variance": variance,
                    "nll": float(
                        np.clip(nll, -self.config.clip_value, self.config.clip_value)
                    ),
                }
            )

        # Every configured horizon contributes through the mean and dispersion.
        # The final coordinate is a class-relative residual margin: positive
        # means this class fits the sequence better than its best competitor.
        mean_scores = np.asarray([np.mean(row["scores"]) for row in per_class])
        output = []
        for index, row in enumerate(per_class):
            competitors = np.delete(mean_scores, index)
            margin = (
                float(np.min(competitors) - mean_scores[index])
                if competitors.size
                else 0.0
            )
            output.extend(
                [
                    float(mean_scores[index]),
                    float(np.std(row["scores"])),
                    row["variance"],
                    row["nll"],
                    margin,
                ]
            )
        return np.asarray(output, dtype=np.float64)

    def row(self, latent, context_index=None):
        code, dispersion, soft, hard = self._project_windows(latent)
        bank = (
            self.full_operator_bank_
            if context_index is None
            else self._operator_bank(exclude_index=context_index)
        )
        return finite_matrix(
            np.concatenate(
                [
                    self._local_block(code, dispersion, soft),
                    self._transition_block(soft, hard),
                    self._operator_features(latent, bank),
                ]
            ),
            self.config.clip_value,
        )
