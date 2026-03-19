"""Vocal Activity Detector — adaptive hysteresis gate on RMS energy."""

from __future__ import annotations

import numpy as np
import librosa
from numpy.typing import NDArray

from ..types import STFTResult, AnalysisConfig


def detect_vocal_activity(
    stft_result: STFTResult,
    config: AnalysisConfig | None = None,
) -> NDArray[np.bool_]:
    """Return a per-frame boolean mask indicating vocal-active frames.

    Algorithm
    ---------
    1. Compute RMS from the magnitude spectrogram.
    2. Compute a **global** median RMS baseline over the entire track.
    3. Apply *hysteresis* gating:
       - Enter active when RMS > global_median × ``enter_ratio``
       - Exit active  when RMS < global_median × ``exit_ratio``
    4. Additional floor gate: frames with RMS < global_RMS_percentile(15)
       are always considered inactive.
    5. Discard active segments shorter than ``min_active_ms``.

    Parameters
    ----------
    stft_result : STFTResult
    config : AnalysisConfig or None

    Returns
    -------
    active_mask : ndarray[bool], shape (n_frames,)
    """
    if config is None:
        config = AnalysisConfig()

    S = stft_result.S
    sr = stft_result.sr
    hop = stft_result.hop_length

    # --- 1. Frame-level RMS from spectrogram ---
    rms = librosa.feature.rms(S=S)[0]  # shape (n_frames,)
    n_frames = len(rms)

    # --- 2. Global median baseline ---
    # For vocal stems, the global median is typically near silence
    # which makes it easy to detect active vocal frames.
    global_median = float(np.median(rms))

    # Floor gate: frames below the 15th percentile are always silent
    floor_threshold = float(np.percentile(rms, 15))

    # --- 3. Hysteresis gating with global baseline ---
    enter_thresh = max(global_median * config.vad_enter_ratio, floor_threshold)
    exit_thresh = max(global_median * config.vad_exit_ratio, floor_threshold * 0.8)

    active = np.zeros(n_frames, dtype=bool)
    in_active = False
    for i in range(n_frames):
        if in_active:
            if rms[i] < exit_thresh:
                in_active = False
            else:
                active[i] = True
        else:
            if rms[i] > enter_thresh:
                in_active = True
                active[i] = True

    # --- 4. Minimum duration filter ---
    min_frames = int(config.vad_min_active_ms * sr / (1000.0 * hop))
    if min_frames < 1:
        min_frames = 1

    active = _filter_short_segments(active, min_frames)

    return active


def _filter_short_segments(mask: NDArray[np.bool_], min_len: int) -> NDArray[np.bool_]:
    """Remove contiguous True runs shorter than *min_len* frames."""
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            start = i
            while i < n and out[i]:
                i += 1
            if (i - start) < min_len:
                out[start:i] = False
        else:
            i += 1
    return out
