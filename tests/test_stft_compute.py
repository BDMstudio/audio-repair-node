"""Tests for stft_compute and band utility functions."""

import numpy as np

from audio_repair.core.audio_loader import load_audio
from audio_repair.core.stft_compute import (
    compute_stft,
    band_energy,
    band_ratio,
    band_flatness,
)


class TestComputeSTFT:
    def test_output_shape(self, sine_440_path) -> None:
        y, sr = load_audio(sine_440_path)
        result = compute_stft(y, sr)
        assert result.S.ndim == 2
        assert result.S.shape[0] == result.n_bins
        assert result.S.shape[1] == result.n_frames
        assert len(result.freqs) == result.n_bins
        assert len(result.times) == result.n_frames

    def test_sine_440_energy_in_body_band(self, sine_440_path) -> None:
        """440 Hz sine should concentrate energy in the body band."""
        y, sr = load_audio(sine_440_path)
        result = compute_stft(y, sr)
        body_e = band_energy(result.S, result.freqs, 200, 4000)
        high_e = band_energy(result.S, result.freqs, 5000, 8000)
        assert np.mean(body_e) > np.mean(high_e) * 100


class TestBandEnergy:
    def test_energy_non_negative(self, white_noise_path) -> None:
        y, sr = load_audio(white_noise_path)
        result = compute_stft(y, sr)
        e = band_energy(result.S, result.freqs, 200, 4000)
        assert np.all(e >= 0)


class TestBandRatio:
    def test_ratio_positive(self, sine_440_path) -> None:
        y, sr = load_audio(sine_440_path)
        result = compute_stft(y, sr)
        r = band_ratio(result.S, result.freqs, (5000, 8000), (200, 4000))
        assert np.all(r >= 0)

    def test_ratio_high_for_bright_signal(self, tmp_wav_dir) -> None:
        """A signal dominated by 6 kHz should have high harsh_band_ratio."""
        import soundfile as sf
        sr = 44100
        t = np.arange(sr * 2) / sr
        bright = (0.8 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
        path = tmp_wav_dir / "bright_6k.wav"
        sf.write(str(path), bright, sr)

        y, sr = load_audio(path)
        result = compute_stft(y, sr)
        r = band_ratio(result.S, result.freqs, (5000, 8000), (200, 4000))
        assert np.mean(r) > 1.0


class TestBandFlatness:
    def test_white_noise_flatness_high(self, white_noise_path) -> None:
        y, sr = load_audio(white_noise_path)
        result = compute_stft(y, sr)
        f = band_flatness(result.S, result.freqs, 4000, 12000)
        assert np.mean(f) > 0.5

    def test_sine_flatness_low(self, sine_440_path) -> None:
        y, sr = load_audio(sine_440_path)
        result = compute_stft(y, sr)
        f = band_flatness(result.S, result.freqs, 200, 4000)
        assert np.mean(f) < 0.3

    def test_no_nan(self, white_noise_path) -> None:
        y, sr = load_audio(white_noise_path)
        result = compute_stft(y, sr)
        f = band_flatness(result.S, result.freqs, 4000, 12000)
        assert not np.any(np.isnan(f))

    def test_empty_band_returns_zeros(self, sine_440_path) -> None:
        """A band above Nyquist should return all zeros."""
        y, sr = load_audio(sine_440_path)
        result = compute_stft(y, sr)
        # Use a band completely above Nyquist (22050 Hz at 44100 sr)
        f = band_flatness(result.S, result.freqs, 23000, 25000)
        assert np.all(f == 0)
