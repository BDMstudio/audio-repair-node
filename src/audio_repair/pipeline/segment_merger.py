"""Merge contiguous defect frames into segments and discard short ones."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import AnalysisConfig, DefectSegment, STFTResult


def merge_segments(
    harsh_scores: NDArray[np.floating],
    distortion_scores: NDArray[np.floating],
    active_mask: NDArray[np.bool_],
    stft_result: STFTResult,
    config: AnalysisConfig,
) -> list[DefectSegment]:
    """Identify defect regions, merge nearby ones, and filter short segments."""
    n_frames = len(harsh_scores)
    ms_per_frame = (stft_result.hop_length / stft_result.sr) * 1000.0

    harsh_flag = harsh_scores >= config.harsh_high
    distort_flag = distortion_scores >= config.distortion_high
    defect_flag = (harsh_flag | distort_flag) & active_mask

    if not np.any(defect_flag):
        return []

    raw_segs = _find_runs(defect_flag)
    merge_gap = int(min(config.merge_gap_ms_harsh, config.merge_gap_ms_distortion) / ms_per_frame)
    merged = _merge_nearby(raw_segs, merge_gap)

    results: list[DefectSegment] = []
    for idx, (start, end) in enumerate(merged):
        seg_h = float(np.mean(harsh_scores[start:end]))
        seg_d = float(np.mean(distortion_scores[start:end]))

        is_harsh = seg_h >= config.harsh_high
        is_distort = seg_d >= config.distortion_high
        if is_harsh and is_distort:
            primary = "both"
            min_dur = min(config.min_harsh_duration_ms, config.min_distortion_duration_ms)
        elif is_harsh:
            primary = "harsh"
            min_dur = config.min_harsh_duration_ms
        elif is_distort:
            primary = "distortion"
            min_dur = config.min_distortion_duration_ms
        else:
            primary = "harsh" if seg_h > seg_d else "distortion"
            min_dur = min(config.min_harsh_duration_ms, config.min_distortion_duration_ms)

        duration_ms = (end - start) * ms_per_frame
        if duration_ms < min_dur:
            continue

        start_ms = start * ms_per_frame
        end_ms = end * ms_per_frame

        # Confidence: how far above threshold, scaled to [0.5, 1.0]
        h_excess = max(0.0, seg_h - config.harsh_high) / max(config.harsh_high, 1e-6)
        d_excess = max(0.0, seg_d - config.distortion_high) / max(config.distortion_high, 1e-6)
        confidence = round(0.5 + 0.5 * min(max(h_excess, d_excess), 1.0), 4)

        # Review needed: both in grey zone, or short but extreme
        h_grey = config.harsh_low <= seg_h < config.harsh_high
        d_grey = config.distortion_low <= seg_d < config.distortion_high
        short_extreme = duration_ms < 150 and (seg_h > config.harsh_high or seg_d > config.distortion_high)
        review = (h_grey and d_grey) or short_extreme

        results.append(DefectSegment(
            segment_id=f"seg_{idx:04d}",
            start_ms=round(start_ms, 1),
            end_ms=round(end_ms, 1),
            primary_class=primary,
            harsh_score=round(seg_h, 4),
            distortion_score=round(seg_d, 4),
            confidence=confidence,
            review_needed=review,
        ))

    return results


def _find_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Find contiguous True runs → ``(start, end_exclusive)``."""
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    return list(zip(starts.tolist(), ends.tolist()))


def _merge_nearby(
    segments: list[tuple[int, int]], gap: int,
) -> list[tuple[int, int]]:
    """Merge segments whose gap is ≤ *gap* frames."""
    if not segments:
        return []

    merged: list[tuple[int, int]] = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged
