"""Tests for segment_merger."""

import numpy as np

from audio_repair.pipeline.segment_merger import merge_segments, _find_runs, _merge_nearby
from audio_repair.core.stft_compute import compute_stft
from audio_repair.types import AnalysisConfig


class TestFindRuns:
    def test_empty(self):
        assert _find_runs(np.array([False, False])) == []

    def test_single_run(self):
        assert _find_runs(np.array([False, True, True, False])) == [(1, 3)]

    def test_multiple_runs(self):
        mask = np.array([True, True, False, False, True, False])
        assert _find_runs(mask) == [(0, 2), (4, 5)]

    def test_all_true(self):
        assert _find_runs(np.array([True, True, True])) == [(0, 3)]


class TestMergeNearby:
    def test_no_merge_when_far(self):
        segs = [(0, 5), (20, 25)]
        assert _merge_nearby(segs, gap=3) == [(0, 5), (20, 25)]

    def test_merge_when_close(self):
        segs = [(0, 5), (7, 12)]
        assert _merge_nearby(segs, gap=3) == [(0, 12)]

    def test_chain_merge(self):
        segs = [(0, 5), (7, 10), (12, 15)]
        assert _merge_nearby(segs, gap=3) == [(0, 15)]


class TestMergeSegments:
    def test_no_defects_returns_empty(self, clean_vocal):
        y, sr = clean_vocal
        stft = compute_stft(y, sr)
        config = AnalysisConfig()
        n = stft.n_frames
        harsh = np.full(n, 0.1)
        distort = np.full(n, 0.1)
        active = np.ones(n, dtype=bool)
        result = merge_segments(harsh, distort, active, stft, config)
        assert result == []

    def test_short_segments_filtered(self, harsh_signal):
        y, sr = harsh_signal
        stft = compute_stft(y, sr)
        config = AnalysisConfig()
        config.min_harsh_duration_ms = 99999  # Very high
        n = stft.n_frames
        harsh = np.full(n, 0.8)
        distort = np.full(n, 0.1)
        active = np.ones(n, dtype=bool)
        result = merge_segments(harsh, distort, active, stft, config)
        assert result == []
