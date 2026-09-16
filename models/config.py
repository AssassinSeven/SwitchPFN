"""Default SwitchPFN settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    target_length: int = 128
    latent_rank: int = 12
    window_length: int = 8
    window_stride: int = 4
    n_states: int = 12
    operator_code_dim: int = 12
    operator_ridge: float = 1e-2
    operator_horizons: tuple[int, ...] = (1, 2, 4)
    transition_lags: tuple[int, ...] = (1, 2)
    path_dim: int = 4
    path_order: int = 3
    path_partitions: tuple[int, ...] = (1, 2, 4)
    path_views: tuple[str, ...] = ("latent", "velocity")
    segment_scales: tuple[int, ...] = (1, 2, 4, 8)
    spectral_bands: int = 12
    low_frequency_bins: int = 8
    autocorr_lags: tuple[int, ...] = (1, 2, 4, 8, 16)
    cross_lags: tuple[int, ...] = (1, 2, 4)
    feature_budget: int = 1024
    prefix_fraction: float = 0.25
    n_estimators: int = 8
    pca_fit_rows: int = 50_000
    codebook_fit_windows: int = 50_000
    random_state: int = 2027
    eps: float = 1e-8
    clip_value: float = 1e4
