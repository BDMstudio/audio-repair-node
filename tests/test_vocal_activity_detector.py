"""Tests for vocal_activity_detector."""

import numpy as np

from audio_repair.core.stft_compute import compute_stft
from audio_repair.core.vocal_activity_detector import detect_vocal_activity
from audio_repair.types import AnalysisConfig


def test_silence_no_activity(silence_signal):
    """Silent input should produce no active frames."""
    y, sr = silence_signal
    stft = compute_stft(y, sr)
    config = AnalysisConfig()
    mask = detect_vocal_activity(stft, config)
    assert not np.any(mask), "Silence should have no active frames"


def test_output_shape_and_dtype(harsh_signal):
    """Mask should be bool with correct shape."""
    y, sr = harsh_signal
    stft = compute_stft(y, sr)
    config = AnalysisConfig()
    mask = detect_vocal_activity(stft, config)
    assert mask.shape == (stft.n_frames,)
    assert mask.dtype == bool


def test_no_crash_on_noise(white_noise):
    """White noise should not crash the VAD."""
    _, y, sr = white_noise
    stft = compute_stft(y, sr)
    config = AnalysisConfig()
    mask = detect_vocal_activity(stft, config)
    assert mask.shape == (stft.n_frames,)
    assert mask.dtype == bool


def test_higher_enter_ratio_fewer_active(harsh_signal):
    """Increasing enter_ratio should produce equal or fewer active frames."""
    y, sr = harsh_signal
    stft = compute_stft(y, sr)

    config_low = AnalysisConfig()
    config_low.vad_enter_ratio = 1.05

    config_high = AnalysisConfig()
    config_high.vad_enter_ratio = 2.0

    mask_low = detect_vocal_activity(stft, config_low)
    mask_high = detect_vocal_activity(stft, config_high)

    assert mask_high.sum() <= mask_low.sum(), (
        f"Higher enter_ratio should give fewer active frames: "
        f"low={mask_low.sum()}, high={mask_high.sum()}"
    )
