"""Distortion (clip-like / noisy-high-band / low-bitrate) defect detector."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import AnalysisConfig, FrameFeatures
from .harsh_detector import normalise_metric


def _bandwidth_expansion(
    bandwidth: NDArray[np.floating],
    active_mask: NDArray[np.bool_],
) -> NDArray[np.floating]:
    """Bandwidth deviation above song's median baseline."""
    active_vals = bandwidth[active_mask]
    if len(active_vals) == 0:
        return np.zeros_like(bandwidth)
    return np.maximum(bandwidth - float(np.median(active_vals)), 0.0)


def _breathy_penalty(
    rms: NDArray[np.floating],
    flatness: NDArray[np.floating],
    active_mask: NDArray[np.bool_],
) -> NDArray[np.floating]:
    """Context penalty: high flatness + low RMS → ``(1 - rms_norm) × flat_norm``."""
    active_rms = rms[active_mask]
    active_flat = flatness[active_mask]
    if len(active_rms) == 0:
        return np.zeros_like(rms)

    rms_range = max(float(np.max(active_rms)) - float(np.min(active_rms)), 1e-10)
    norm_rms = np.clip((rms - float(np.min(active_rms))) / rms_range, 0.0, 1.0)

    flat_range = max(float(np.max(active_flat)) - float(np.min(active_flat)), 1e-10)
    norm_flat = np.clip((flatness - float(np.min(active_flat))) / flat_range, 0.0, 1.0)

    return (1.0 - norm_rms) * norm_flat


def compute_distortion_scores(
    features: FrameFeatures,
    active_mask: NDArray[np.bool_],
    config: AnalysisConfig,
) -> NDArray[np.floating]:
    """Weighted distortion score per frame — shape ``(n_frames,)``."""
    cap = config.norm_upper_cap
    eps = config.norm_eps

    raw_flatness = features.flatness_air
    raw_highband = features.highband_ratio
    raw_bw = _bandwidth_expansion(features.spectral_bandwidth, active_mask)
    raw_breathy = _breathy_penalty(features.rms_envelope, features.flatness_air, active_mask)

    n_f = normalise_metric(raw_flatness, active_mask, cap, eps)
    n_h = normalise_metric(raw_highband, active_mask, cap, eps)
    n_b = normalise_metric(raw_bw, active_mask, cap, eps)
    n_p = normalise_metric(raw_breathy, active_mask, cap, eps)

    score = (
        config.w_flatness_air * n_f
        + config.w_highband_ratio * n_h
        + config.w_bandwidth_expansion * n_b
        + config.w_breathy_penalty * n_p
    )

    return score / cap
