"""Main pipeline orchestrator — wires all modules together."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from ..config import load_config
from ..core.audio_loader import load_audio
from ..core.stft_compute import compute_stft
from ..core.vocal_activity_detector import detect_vocal_activity
from ..detection.feature_extractor import extract_features
from ..detection.harsh_detector import compute_harsh_scores
from ..detection.distortion_detector import compute_distortion_scores
from ..pipeline.segment_merger import merge_segments
from ..pipeline.repair_router import route_repairs
from ..types import AnalysisConfig

logger = logging.getLogger(__name__)


def analyze(
    vocal_path: str | Path,
    instrumental_path: str | Path,
    config: AnalysisConfig | None = None,
    output_dir: str | Path = "outputs",
) -> dict:
    """Run the full detection + routing pipeline.

    Returns dict with keys ``"defect_segments"`` and ``"repair_plan"``.
    """
    if config is None:
        config = AnalysisConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load audio
    logger.info("Loading audio files...")
    y_vocal, sr_vocal = load_audio(vocal_path)
    y_instr, sr_instr = load_audio(instrumental_path)

    if sr_vocal != sr_instr:
        logger.warning(
            "Sample rate mismatch: vocal=%d, instrumental=%d. Using vocal sr.",
            sr_vocal, sr_instr,
        )

    # 2. Unified STFT
    logger.info("Computing STFT...")
    vocal_stft = compute_stft(y_vocal, sr_vocal, config.n_fft, config.hop_length)
    instr_stft = compute_stft(y_instr, sr_instr, config.n_fft, config.hop_length)

    # 3. Vocal activity detection
    logger.info("Detecting vocal activity...")
    active_mask = detect_vocal_activity(vocal_stft, config)
    active_pct = active_mask.sum() / len(active_mask) * 100
    logger.info("Vocal-active frames: %.1f%%", active_pct)

    # 4. Feature extraction
    logger.info("Extracting features...")
    features = extract_features(vocal_stft, config)

    # 5. Harsh + distortion scoring
    logger.info("Computing defect scores...")
    harsh_scores = compute_harsh_scores(
        features, active_mask, vocal_stft, instr_stft, config,
    )
    distortion_scores = compute_distortion_scores(
        features, active_mask, config,
    )

    # 6. Segment merging
    logger.info("Merging defect segments...")
    segments = merge_segments(
        harsh_scores, distortion_scores, active_mask, vocal_stft, config,
    )
    logger.info("Found %d defect segments.", len(segments))

    # 7. Repair routing
    logger.info("Routing repairs...")
    plan = route_repairs(segments, config)

    # 8. Write outputs
    segments_data = [asdict(s) for s in segments]
    plan_data = [asdict(p) for p in plan]

    seg_path = output_dir / "defect_segments.json"
    plan_path = output_dir / "repair_plan.json"

    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, indent=2, ensure_ascii=False)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_data, f, indent=2, ensure_ascii=False)

    logger.info("Outputs written to %s", output_dir)

    return {
        "defect_segments": segments_data,
        "repair_plan": plan_data,
    }
