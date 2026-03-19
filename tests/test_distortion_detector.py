"""Tests for distortion_detector."""

import numpy as np

from audio_repair.core.stft_compute import compute_stft
from audio_repair.detection.feature_extractor import extract_features
from audio_repair.detection.distortion_detector import compute_distortion_scores
from audio_repair.types import AnalysisConfig


def test_noise_higher_than_clean(white_noise, clean_vocal):
    """White noise (high flatness) should score higher than clean vocal."""
    config = AnalysisConfig()

    # Noise
    _, y_n, sr_n = white_noise
    stft_n = compute_stft(y_n, sr_n)
    feat_n = extract_features(stft_n, config)
    active_n = np.ones(stft_n.n_frames, dtype=bool)
    scores_n = compute_distortion_scores(feat_n, active_n, config)

    # Clean
    y_c, sr_c = clean_vocal
    stft_c = compute_stft(y_c, sr_c)
    feat_c = extract_features(stft_c, config)
    active_c = np.ones(stft_c.n_frames, dtype=bool)
    scores_c = compute_distortion_scores(feat_c, active_c, config)

    assert np.mean(scores_n) > np.mean(scores_c), (
        f"Noise ({np.mean(scores_n):.3f}) should score higher "
        f"than clean ({np.mean(scores_c):.3f})"
    )


def test_no_nan_on_silence(silence_signal):
    """Silence should produce zero scores, not NaN."""
    y, sr = silence_signal
    config = AnalysisConfig()
    stft = compute_stft(y, sr)
    feat = extract_features(stft, config)
    active = np.zeros(stft.n_frames, dtype=bool)
    scores = compute_distortion_scores(feat, active, config)
    assert not np.any(np.isnan(scores))
    assert np.all(scores == 0)
