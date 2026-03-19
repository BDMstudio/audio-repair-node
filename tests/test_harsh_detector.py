"""Tests for harsh_detector."""

import numpy as np

from audio_repair.core.stft_compute import compute_stft
from audio_repair.detection.feature_extractor import extract_features
from audio_repair.detection.harsh_detector import compute_harsh_scores
from audio_repair.types import AnalysisConfig


def test_harsh_signal_high_score(harsh_signal, clean_vocal):
    """A harsh signal should score higher than a clean vocal."""
    config = AnalysisConfig()

    # Harsh
    y_h, sr_h = harsh_signal
    stft_h = compute_stft(y_h, sr_h)
    feat_h = extract_features(stft_h, config)
    active_h = np.ones(stft_h.n_frames, dtype=bool)
    scores_h = compute_harsh_scores(feat_h, active_h, stft_h, stft_h, config)

    # Clean
    y_c, sr_c = clean_vocal
    stft_c = compute_stft(y_c, sr_c)
    feat_c = extract_features(stft_c, config)
    active_c = np.ones(stft_c.n_frames, dtype=bool)
    scores_c = compute_harsh_scores(feat_c, active_c, stft_c, stft_c, config)

    assert np.mean(scores_h) > np.mean(scores_c), (
        f"Harsh signal ({np.mean(scores_h):.3f}) should score higher "
        f"than clean ({np.mean(scores_c):.3f})"
    )


def test_no_nan_on_edge_cases(silence_signal):
    """Scores should not produce NaN even on silence."""
    y, sr = silence_signal
    config = AnalysisConfig()
    stft = compute_stft(y, sr)
    feat = extract_features(stft, config)
    active = np.zeros(stft.n_frames, dtype=bool)
    scores = compute_harsh_scores(feat, active, stft, stft, config)
    assert not np.any(np.isnan(scores))
