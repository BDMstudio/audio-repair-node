"""Configuration loading — defaults + optional JSON override."""

from __future__ import annotations

import json
from pathlib import Path

from .types import AnalysisConfig

# Re-export for convenience
DEFAULT_CONFIG = AnalysisConfig()


def load_config(path: str | Path | None = None) -> AnalysisConfig:
    """Return an AnalysisConfig, optionally overriding fields from a JSON file.

    The JSON file may contain *any subset* of AnalysisConfig fields.
    Nested ``vad`` / ``bands`` keys are flattened automatically.
    """
    if path is None:
        return AnalysisConfig()

    raw: dict = json.loads(Path(path).read_text(encoding="utf-8"))
    flat: dict = {}

    # Flatten nested keys used in the config template
    vad = raw.pop("vad", {})
    if vad:
        flat["vad_enter_ratio"] = vad.get("enter_ratio", DEFAULT_CONFIG.vad_enter_ratio)
        flat["vad_exit_ratio"] = vad.get("exit_ratio", DEFAULT_CONFIG.vad_exit_ratio)
        flat["vad_median_window_frames"] = vad.get(
            "median_window_frames", DEFAULT_CONFIG.vad_median_window_frames
        )
        flat["vad_min_active_ms"] = vad.get("min_active_ms", DEFAULT_CONFIG.vad_min_active_ms)

    bands = raw.pop("bands", {})
    if bands:
        for key in ("body_band", "harsh_band_main", "harsh_band_wide", "air_noise_band"):
            if key in bands:
                flat[key] = tuple(bands[key])

    # Flatten thresholds if nested
    thresholds = raw.pop("thresholds", {})
    if thresholds:
        flat.update(thresholds)

    # Flatten weights if nested
    weights = raw.pop("weights", {})
    if weights:
        for group_key, group_val in weights.items():
            if isinstance(group_val, dict):
                for sub_key, sub_val in group_val.items():
                    flat[f"w_{sub_key}"] = sub_val

    # Remaining top-level keys map directly
    flat.update(raw)

    # Remove keys not present in AnalysisConfig
    valid_fields = {f.name for f in AnalysisConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in flat.items() if k in valid_fields}

    return AnalysisConfig(**filtered)
