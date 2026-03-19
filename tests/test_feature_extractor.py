"""Tests for feature_extractor."""

import numpy as np
import pytest

from audio_repair.core.audio_loader import load_audio
from audio_repair.core.stft_compute import compute_stft
from audio_repair.detection.feature_extractor import extract_features


class TestFeatureExtractor:
    def test_output_shapes(self, sine_440_path) -> None:
        """All 8 feature vectors should have shape (n_frames,)."""
        y, sr = load_audio(sine_440_path)
        stft = compute_stft(y, sr)
        feats = extract_features(stft)

        expected_len = stft.n_frames
        assert feats.spectral_centroid.shape == (expected_len,)
        assert feats.spectral_flatness.shape == (expected_len,)
        assert feats.spectral_bandwidth.shape == (expected_len,)
        assert feats.harsh_band_ratio_main.shape == (expected_len,)
        assert feats.harsh_band_ratio_wide.shape == (expected_len,)
        assert feats.highband_ratio.shape == (expected_len,)
        assert feats.flatness_air.shape == (expected_len,)
        assert feats.rms_envelope.shape == (expected_len,)

    def test_no_nan(self, white_noise_path) -> None:
        """No feature should contain NaN values."""
        y, sr = load_audio(white_noise_path)
        stft = compute_stft(y, sr)
        feats = extract_features(stft)

        for name in (
            "spectral_centroid", "spectral_flatness", "spectral_bandwidth",
            "harsh_band_ratio_main", "harsh_band_ratio_wide", "highband_ratio",
            "flatness_air", "rms_envelope",
        ):
            arr = getattr(feats, name)
            assert not np.any(np.isnan(arr)), f"NaN in {name}"

    def test_rms_non_negative(self, sine_440_path) -> None:
        y, sr = load_audio(sine_440_path)
        stft = compute_stft(y, sr)
        feats = extract_features(stft)
        assert np.all(feats.rms_envelope >= 0)

    def test_centroid_reasonable_for_440(self, sine_440_path) -> None:
        """A 440 Hz sine should have centroid near 440 Hz."""
        y, sr = load_audio(sine_440_path)
        stft = compute_stft(y, sr)
        feats = extract_features(stft)

        mean_centroid = float(np.mean(feats.spectral_centroid))
        assert 300 < mean_centroid < 600
