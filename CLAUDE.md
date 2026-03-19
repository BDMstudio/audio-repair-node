# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

audio-repair-node is a Python CLI tool that detects **harsh** and **distortion** defects in AI-generated vocals by analyzing spectral features, then outputs repair routing plans (JSON) mapping to external tools (iZotope RX 11, pure:deess). V0.1 scope is intentionally limited to these two defect classes.

## Commands

```bash
# Install (uses uv package manager)
uv sync

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_harsh_detector.py

# Run with coverage
pytest --cov=src/audio_repair tests/

# CLI usage
audio-repair analyze --vocal <path> --instrumental <path> [--config configs/repair_config.json] [--output-dir <dir>] [-v]
```

## Architecture

**Pipeline flow** (each stage feeds the next):

```
vocal + instrumental WAV
  → audio_loader (load to mono float32)
  → stft_compute (STFT magnitude spectrograms)
  → vocal_activity_detector (per-frame VAD mask via adaptive hysteresis)
  → feature_extractor (8 per-frame spectral features)
  → harsh_detector (5 weighted sub-metrics → harsh score)
  → distortion_detector (4 weighted sub-metrics → distortion score)
  → segment_merger (merge contiguous defect frames → DefectSegment list)
  → repair_router (map segments to repair plans → RepairPlan list)
  → JSON output (defect_segments.json + repair_plan.json)
```

**Module layout** — `src/audio_repair/`:

| Layer | Modules | Responsibility |
|-------|---------|----------------|
| `core/` | `audio_loader`, `stft_compute`, `vocal_activity_detector` | I/O, signal transforms, VAD |
| `detection/` | `feature_extractor`, `harsh_detector`, `distortion_detector` | Per-frame feature extraction and scoring |
| `pipeline/` | `segment_merger`, `repair_router`, `analyzer` | Merge frames→segments, route repairs, orchestrate |
| root | `types`, `config`, `cli` | Data types, config loading, CLI entry |

**Orchestrator**: `pipeline/analyzer.py` wires all modules together. The CLI calls `analyzer.run_analysis()`.

## Key Data Types (`types.py`)

All pipeline stage boundaries use frozen/typed dataclasses:

- `STFTResult` — magnitude spectrogram + freq/time axes
- `FrameFeatures` — 8 per-frame feature vectors (centroid, flatness, bandwidth, band ratios, RMS)
- `DefectSegment` — merged segment with scores, class (`harsh`/`distortion`/`both`), confidence
- `RepairPlan` — segment + `recommended_route` list of `RepairStep(tool, action)`
- `AnalysisConfig` — ~30 tunable parameters with sensible defaults

## Scoring System

**Harsh score** = weighted sum of: `harsh_band_ratio_main` (0.35), `harsh_band_ratio_wide` (0.20), `persistence_score` (0.20), `collision_score` (0.15), `centroid_drift` (0.10).

**Distortion score** = weighted sum of: `flatness_air` (0.40), `highband_ratio` (0.25), `bandwidth_expansion` (0.20), `breathy_penalty` (0.15).

All features use **per-song baseline normalization**: `(x - median) / IQR`, clipped to `[0, cap]`, computed on VAD-active frames only.

## Routing Rules

Thresholds: `harsh_high=0.60`, `harsh_low=0.45`, `distortion_high=0.58`, `distortion_low=0.42`.

1. Harsh only → `deess_light`
2. Distortion only → `declip_light` (+ optional denoise)
3. Both high → `declip_light` → `voice_denoise_light?` → `deess_light`
4. Below threshold → `skip`
5. Grey zone → dominant class route at light level

## Configuration

`AnalysisConfig` provides defaults. JSON config files (see `configs/repair_config.json`) can override any subset of fields. Nested keys (`vad`, `bands`, `weights`, `thresholds`) are auto-flattened by `config.py`. Unknown keys are silently ignored.

## Testing Conventions

- **Fixtures** in `conftest.py`: synthetic signals (sine, white noise, harsh_signal, clean_vocal, silence) + temp WAV files via `tmp_path`
- Each module has a corresponding `test_<module>.py`
- `test_analyzer_e2e.py` runs the full pipeline and validates JSON output structure
