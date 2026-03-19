"""Tests for audio_loader."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_repair.core.audio_loader import load_audio


class TestLoadAudio:
    def test_loads_mono_wav(self, sine_440_path: Path) -> None:
        y, sr = load_audio(sine_440_path)
        assert sr == 44100
        assert y.ndim == 1
        assert y.dtype == np.float32
        assert len(y) > 0

    def test_converts_stereo_to_mono(self, tmp_path: Path) -> None:
        sr = 44100
        stereo = np.stack([
            np.sin(2 * np.pi * 440 * np.arange(sr) / sr),
            np.sin(2 * np.pi * 880 * np.arange(sr) / sr),
        ], axis=1).astype(np.float32)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, sr)

        y, out_sr = load_audio(path)
        assert out_sr == sr
        assert y.ndim == 1
        assert len(y) == sr

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/audio.wav")

    def test_warns_on_unusual_sr(self, tmp_path: Path) -> None:
        sr = 22050
        y = np.zeros(sr, dtype=np.float32)
        path = tmp_path / "low_sr.wav"
        sf.write(str(path), y, sr)

        with pytest.warns(UserWarning, match="22050"):
            load_audio(path)
