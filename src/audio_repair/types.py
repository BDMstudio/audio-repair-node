"""Core data types for audio-repair pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class STFTResult:
    """Result of a Short-Time Fourier Transform computation."""

    S: NDArray[np.floating]          # magnitude spectrogram (n_bins, n_frames)
    freqs: NDArray[np.floating]      # frequency bin centres   (n_bins,)
    times: NDArray[np.floating]      # frame centre times      (n_frames,)
    sr: int
    hop_length: int

    @property
    def n_frames(self) -> int:
        return int(self.S.shape[1])

    @property
    def n_bins(self) -> int:
        return int(self.S.shape[0])


@dataclass
class FrameFeatures:
    """Per-frame feature vectors extracted from an STFTResult."""

    spectral_centroid: NDArray[np.floating]     # (T,)
    spectral_flatness: NDArray[np.floating]     # (T,)
    spectral_bandwidth: NDArray[np.floating]    # (T,)
    harsh_band_ratio_main: NDArray[np.floating] # (T,)  5-8k / body
    harsh_band_ratio_wide: NDArray[np.floating] # (T,)  4-10k / body
    highband_ratio: NDArray[np.floating]        # (T,)  4-12k / body
    flatness_air: NDArray[np.floating]          # (T,)  4-12k flatness
    rms_envelope: NDArray[np.floating]          # (T,)


@dataclass
class DefectSegment:
    """A contiguous segment flagged as defective."""

    segment_id: str
    start_ms: float
    end_ms: float
    primary_class: str                # "harsh" | "distortion" | "both"
    harsh_score: float
    distortion_score: float
    modifiers: list[str] = field(default_factory=list)
    confidence: float = 0.0
    review_needed: bool = False


@dataclass
class RepairStep:
    """A single repair action within a repair plan."""

    tool: str        # e.g. "rx", "pure_deess"
    action: str      # e.g. "deess_light", "declip_light"


@dataclass
class RepairPlan:
    """Repair plan for one defect segment."""

    segment_id: str
    start_ms: float
    end_ms: float
    primary_class: str
    harsh_score: float
    distortion_score: float
    modifiers: list[str] = field(default_factory=list)
    recommended_route: list[RepairStep] = field(default_factory=list)
    confidence: float = 0.0
    review_needed: bool = False

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "primary_class": self.primary_class,
            "harsh_score": round(self.harsh_score, 4),
            "distortion_score": round(self.distortion_score, 4),
            "modifiers": self.modifiers,
            "recommended_route": [
                {"tool": s.tool, "action": s.action} for s in self.recommended_route
            ],
            "confidence": round(self.confidence, 4),
            "review_needed": self.review_needed,
        }


@dataclass
class AnalysisConfig:
    """Full pipeline configuration with sensible defaults."""

    # STFT
    n_fft: int = 2048
    hop_length: int = 512

    # Frequency bands
    body_band: tuple[float, float] = (200.0, 4000.0)
    harsh_band_main: tuple[float, float] = (5000.0, 8000.0)
    harsh_band_wide: tuple[float, float] = (4000.0, 10000.0)
    air_noise_band: tuple[float, float] = (4000.0, 12000.0)

    # VAD
    vad_enter_ratio: float = 1.25
    vad_exit_ratio: float = 1.10
    vad_median_window_frames: int = 31
    vad_min_active_ms: float = 120.0

    # Harsh detector weights
    w_harsh_band_ratio_main: float = 0.35
    w_harsh_band_ratio_wide: float = 0.20
    w_persistence_score: float = 0.20
    w_collision_score: float = 0.15
    w_centroid_drift: float = 0.10

    # Distortion detector weights
    w_flatness_air: float = 0.40
    w_highband_ratio: float = 0.25
    w_bandwidth_expansion: float = 0.20
    w_breathy_penalty: float = 0.15

    # Thresholds
    harsh_high: float = 0.60
    harsh_low: float = 0.45
    distortion_high: float = 0.58
    distortion_low: float = 0.42

    # Segment merger
    min_harsh_duration_ms: float = 220.0
    min_distortion_duration_ms: float = 180.0
    merge_gap_ms_harsh: float = 120.0
    merge_gap_ms_distortion: float = 100.0

    # Normalisation
    norm_upper_cap: float = 3.0
    norm_eps: float = 1e-6

    # Persistence
    persistence_window_frames: int = 21
