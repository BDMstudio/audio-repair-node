"""Audio file loading — WAV/FLAC/OGG → mono float32."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_EXPECTED_SAMPLE_RATES = {44100, 48000}


def load_audio(path: str | Path) -> tuple[NDArray[np.floating], int]:
    """Load an audio file and return (y, sr) as mono float32.

    Parameters
    ----------
    path : str or Path
        Path to a WAV, FLAC, or OGG file.

    Returns
    -------
    y : ndarray, shape (n_samples,)
        Audio time-series, mono, float32.
    sr : int
        Sample rate.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If the file cannot be read by soundfile.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio file {path}: {exc}") from exc

    # Convert to mono by averaging channels
    if data.shape[1] > 1:
        y: NDArray[np.floating] = np.mean(data, axis=1)
    else:
        y = data[:, 0]

    # Warn if sample rate is unexpected
    if sr not in _EXPECTED_SAMPLE_RATES:
        warnings.warn(
            f"Sample rate {sr} Hz is not in {_EXPECTED_SAMPLE_RATES}. "
            "Analysis parameters are tuned for 44.1/48 kHz.",
            stacklevel=2,
        )

    logger.info("Loaded %s — %d samples, %d Hz, mono", path.name, len(y), sr)
    return y.astype(np.float32, copy=False), int(sr)
