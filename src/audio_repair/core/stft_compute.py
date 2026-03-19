"""Unified STFT computation and frequency-band utility functions."""

from __future__ import annotations

import numpy as np
import librosa
from numpy.typing import NDArray

from ..types import STFTResult


def compute_stft(
    y: NDArray[np.floating],
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> STFTResult:
    """Compute the magnitude STFT and return an STFTResult.

    Parameters
    ----------
    y : ndarray, shape (n_samples,)
    sr : int
    n_fft : int
    hop_length : int

    Returns
    -------
    STFTResult
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(S.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft,
    )
    return STFTResult(S=S, freqs=freqs, times=times, sr=sr, hop_length=hop_length)


# ---------------------------------------------------------------------------
# Frequency-band helpers
# ---------------------------------------------------------------------------

def _band_mask(freqs: NDArray[np.floating], low: float, high: float) -> NDArray[np.bool_]:
    """Boolean mask selecting frequency bins within [low, high]."""
    return (freqs >= low) & (freqs <= high)


def _slice_band(
    S: NDArray[np.floating],
    freqs: NDArray[np.floating],
    low: float,
    high: float,
) -> NDArray[np.floating]:
    """Extract the sub-spectrogram for frequency bins in [low, high]."""
    mask = _band_mask(freqs, low, high)
    return S[mask]


def band_energy(
    S: NDArray[np.floating],
    freqs: NDArray[np.floating],
    low: float,
    high: float,
) -> NDArray[np.floating]:
    """Sum of squared magnitudes per frame in the band [low, high].

    Returns shape (n_frames,).
    """
    band = _slice_band(S, freqs, low, high)
    return np.sum(band ** 2, axis=0)


def band_ratio(
    S: NDArray[np.floating],
    freqs: NDArray[np.floating],
    target_band: tuple[float, float],
    ref_band: tuple[float, float],
    eps: float = 1e-10,
) -> NDArray[np.floating]:
    """Ratio of energy in *target_band* to energy in *ref_band*.

    ref_band is always explicitly specified (V1: body = 200–4000 Hz).
    Returns shape (n_frames,).
    """
    target_e = band_energy(S, freqs, *target_band)
    ref_e = band_energy(S, freqs, *ref_band)
    return target_e / np.maximum(ref_e, eps)


def band_flatness(
    S: NDArray[np.floating],
    freqs: NDArray[np.floating],
    low: float,
    high: float,
    eps: float = 1e-10,
    min_energy_gate: float = 1e-8,
) -> NDArray[np.floating]:
    """Spectral flatness within a frequency band.

    Geometric-mean / arithmetic-mean ratio, with numerical safety:
    - All values floored to *eps* before log.
    - Frames with total band energy below *min_energy_gate* → 0.
    - Output passed through ``np.nan_to_num``.

    Returns shape (n_frames,).
    """
    band = _slice_band(S, freqs, low, high)
    if band.shape[0] == 0:
        return np.zeros(S.shape[1], dtype=np.float64)

    band = np.maximum(band, eps)
    gm = np.exp(np.mean(np.log(band), axis=0))
    am = np.mean(band, axis=0)
    out = gm / np.maximum(am, eps)

    # Gate: silence / near-silence → 0
    frame_energy = np.sum(band ** 2, axis=0)
    out = np.where(frame_energy < min_energy_gate, 0.0, out)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
