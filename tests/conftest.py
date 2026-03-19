"""Shared test fixtures — both file-path and in-memory fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path


# ---------------------------------------------------------------------------
# In-memory fixtures (used by unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rate() -> int:
    return 44100


@pytest.fixture
def duration_s() -> float:
    return 1.0


@pytest.fixture
def sine_440(tmp_path, sample_rate, duration_s):
    """1-second 440 Hz sine wave WAV file — returns (path, y, sr)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "sine_440.wav"
    sf.write(str(path), y, sample_rate)
    return path, y, sample_rate


@pytest.fixture
def white_noise(tmp_path, sample_rate, duration_s):
    """1-second white noise WAV file — returns (path, y, sr)."""
    rng = np.random.default_rng(42)
    y = (0.3 * rng.standard_normal(int(sample_rate * duration_s))).astype(np.float32)
    path = tmp_path / "white_noise.wav"
    sf.write(str(path), y, sample_rate)
    return path, y, sample_rate


@pytest.fixture
def harsh_signal(sample_rate, duration_s):
    """Synthetic signal with heavy 5-8 kHz energy — returns (y, sr)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    body = 0.3 * np.sin(2 * np.pi * 300 * t)
    harsh = 0.5 * np.sin(2 * np.pi * 6000 * t) + 0.4 * np.sin(2 * np.pi * 7000 * t)
    return (body + harsh).astype(np.float32), sample_rate


@pytest.fixture
def clean_vocal(sample_rate, duration_s):
    """Synthetic clean vocal (body only) — returns (y, sr)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    y = (
        0.5 * np.sin(2 * np.pi * 300 * t)
        + 0.3 * np.sin(2 * np.pi * 600 * t)
        + 0.15 * np.sin(2 * np.pi * 1200 * t)
    ).astype(np.float32)
    return y, sample_rate


@pytest.fixture
def silence_signal(sample_rate, duration_s):
    """1-second silence — returns (y, sr)."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32), sample_rate


# ---------------------------------------------------------------------------
# File-path fixtures (used by stft / e2e tests)
# ---------------------------------------------------------------------------

def _sine(freq: float, duration: float, sr: int = 44100) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _white_noise(duration: float, sr: int = 44100, amplitude: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (amplitude * rng.standard_normal(int(sr * duration))).astype(np.float32)


@pytest.fixture
def tmp_wav_dir(tmp_path: Path) -> Path:
    sr = 44100
    dur = 2.0
    files = {
        "sine_440.wav": _sine(440, dur, sr),
        "white_noise.wav": _white_noise(dur, sr),
    }
    for name, data in files.items():
        sf.write(str(tmp_path / name), data, sr)
    return tmp_path


@pytest.fixture
def sine_440_path(tmp_wav_dir: Path) -> Path:
    return tmp_wav_dir / "sine_440.wav"


@pytest.fixture
def white_noise_path(tmp_wav_dir: Path) -> Path:
    return tmp_wav_dir / "white_noise.wav"
