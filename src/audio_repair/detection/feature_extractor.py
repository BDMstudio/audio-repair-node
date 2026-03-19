"""Feature extraction — 8 raw feature vectors from a shared STFTResult."""

from __future__ import annotations

import numpy as np
import librosa
from numpy.typing import NDArray

from ..types import STFTResult, FrameFeatures, AnalysisConfig
from ..core.stft_compute import band_ratio, band_flatness


def extract_features(
    stft_result: STFTResult,
    config: AnalysisConfig | None = None,
) -> FrameFeatures:
    """Extract all 8 per-frame feature vectors from a magnitude spectrogram.

    Features
    --------
    1. spectral_centroid
    2. spectral_flatness (full-band)
    3. spectral_bandwidth
    4. harsh_band_ratio_main  — 5–8 kHz / body (200–4000)
    5. harsh_band_ratio_wide  — 4–10 kHz / body
    6. highband_ratio          — 4–12 kHz / body
    7. flatness_air            — spectral flatness in 4–12 kHz band
    8. rms_envelope

    Parameters
    ----------
    stft_result : STFTResult
    config : AnalysisConfig or None

    Returns
    -------
    FrameFeatures
    """
    if config is None:
        config = AnalysisConfig()

    S = stft_result.S
    sr = stft_result.sr
    freqs = stft_result.freqs

    # --- librosa spectral features (accept S= to reuse the pre-computed STFT) ---
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rms_envelope = librosa.feature.rms(S=S)[0]

    # --- band ratios ---
    body = config.body_band

    harsh_band_ratio_main = band_ratio(S, freqs, config.harsh_band_main, body)
    harsh_band_ratio_wide = band_ratio(S, freqs, config.harsh_band_wide, body)
    highband_ratio_vec = band_ratio(S, freqs, config.air_noise_band, body)

    # --- band flatness for air/noise region ---
    flatness_air_vec = band_flatness(S, freqs, *config.air_noise_band)

    return FrameFeatures(
        spectral_centroid=spectral_centroid,
        spectral_flatness=spectral_flatness,
        spectral_bandwidth=spectral_bandwidth,
        harsh_band_ratio_main=harsh_band_ratio_main,
        harsh_band_ratio_wide=harsh_band_ratio_wide,
        highband_ratio=highband_ratio_vec,
        flatness_air=flatness_air_vec,
        rms_envelope=rms_envelope,
    )
