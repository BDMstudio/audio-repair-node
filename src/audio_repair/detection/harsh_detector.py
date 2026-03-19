"""Harsh (sibilant / fake-bright) defect detector.

Computes 5 sub-metrics, normalises each to the song's own baseline
(vocal-active frames only), and produces a weighted ``harsh_score`` per frame.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import AnalysisConfig, FrameFeatures, STFTResult
from ..core.stft_compute import band_energy


# ---------------------------------------------------------------------------
# Normalisation (shared with distortion_detector)
# ---------------------------------------------------------------------------

def normalise_metric(
    values: NDArray[np.floating],
    active_mask: NDArray[np.bool_],
    upper_cap: float = 3.0,
    eps: float = 1e-6,
) -> NDArray[np.floating]:
    """Song-level normalisation: ``clip((x - median) / max(IQR, eps), 0, cap)``."""
    active_vals = values[active_mask]
    if len(active_vals) == 0:
        return np.zeros_like(values)

    median = float(np.median(active_vals))
    q1 = float(np.percentile(active_vals, 25))
    q3 = float(np.percentile(active_vals, 75))
    iqr = max(q3 - q1, eps)

    normed = (values - median) / iqr
    return np.clip(normed, 0.0, upper_cap)


# ---------------------------------------------------------------------------
# Sub-metrics
# ---------------------------------------------------------------------------

def _persistence_score(
    harsh_band_ratio: NDArray[np.floating],
    active_mask: NDArray[np.bool_],
    window: int = 15,
) -> NDArray[np.floating]:
    """Fraction of frames in a sliding window where ratio exceeds 75th-pctl."""
    active_vals = harsh_band_ratio[active_mask]
    if len(active_vals) == 0:
        return np.zeros_like(harsh_band_ratio)

    threshold = float(np.percentile(active_vals, 75))
    exceed = (harsh_band_ratio > threshold).astype(np.float64)

    half_w = window // 2
    padded = np.pad(exceed, (half_w, half_w), mode="constant", constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.mean(windows, axis=1)[: len(harsh_band_ratio)]


def _collision_score(
    vocal_stft: STFTResult,
    instr_stft: STFTResult,
    harsh_band: tuple[float, float],
) -> NDArray[np.floating]:
    """Overlap between vocal and instrumental high-frequency energy."""
    eps = 1e-10
    v_e = band_energy(vocal_stft.S, vocal_stft.freqs, *harsh_band)
    i_e = band_energy(instr_stft.S, instr_stft.freqs, *harsh_band)
    n = min(len(v_e), len(i_e))
    v_e, i_e = v_e[:n], i_e[:n]
    collision = np.minimum(v_e, i_e) / np.maximum(v_e, eps)
    result = np.zeros(vocal_stft.n_frames, dtype=np.float64)
    result[:n] = collision
    return result


def _centroid_drift(
    centroid: NDArray[np.floating],
    active_mask: NDArray[np.bool_],
) -> NDArray[np.floating]:
    """How far the spectral centroid drifts above the song's median."""
    active_vals = centroid[active_mask]
    if len(active_vals) == 0:
        return np.zeros_like(centroid)
    baseline = float(np.median(active_vals))
    return np.maximum(centroid - baseline, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_harsh_scores(
    features: FrameFeatures,
    active_mask: NDArray[np.bool_],
    vocal_stft: STFTResult,
    instr_stft: STFTResult,
    config: AnalysisConfig,
) -> NDArray[np.floating]:
    """Weighted harsh score per frame — shape ``(n_frames,)``."""
    cap = config.norm_upper_cap
    eps = config.norm_eps

    # Sub-metrics (raw)
    raw_main = features.harsh_band_ratio_main
    raw_wide = features.harsh_band_ratio_wide
    raw_persist = _persistence_score(raw_main, active_mask, config.persistence_window_frames)
    raw_collision = _collision_score(vocal_stft, instr_stft, config.harsh_band_main)
    raw_drift = _centroid_drift(features.spectral_centroid, active_mask)

    # Normalise
    n_main = normalise_metric(raw_main, active_mask, cap, eps)
    n_wide = normalise_metric(raw_wide, active_mask, cap, eps)
    n_persist = normalise_metric(raw_persist, active_mask, cap, eps)
    n_collision = normalise_metric(raw_collision, active_mask, cap, eps)
    n_drift = normalise_metric(raw_drift, active_mask, cap, eps)

    # Weighted sum
    score = (
        config.w_harsh_band_ratio_main * n_main
        + config.w_harsh_band_ratio_wide * n_wide
        + config.w_persistence_score * n_persist
        + config.w_collision_score * n_collision
        + config.w_centroid_drift * n_drift
    )

    # Scale to [0, 1] range
    return score / cap
