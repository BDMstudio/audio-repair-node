"""End-to-end test: full pipeline from WAV files to JSON output."""

import json

import numpy as np
import soundfile as sf

from audio_repair.pipeline.analyzer import analyze
from audio_repair.types import AnalysisConfig


def test_e2e_produces_valid_json(tmp_path, sample_rate):
    """Pipeline should run end-to-end and produce valid JSON outputs."""
    sr = sample_rate
    duration = 2.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Build a signal with a harsh section in the middle
    vocal = np.zeros(n, dtype=np.float32)
    # Body throughout
    vocal += 0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    # Harsh burst in second half
    mid = n // 2
    vocal[mid:] += (
        0.5 * np.sin(2 * np.pi * 6000 * t[mid:])
        + 0.4 * np.sin(2 * np.pi * 7000 * t[mid:])
    ).astype(np.float32)

    # Simple instrumental
    instr = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    vocal_path = tmp_path / "vocal.wav"
    instr_path = tmp_path / "instrumental.wav"
    sf.write(str(vocal_path), vocal, sr)
    sf.write(str(instr_path), instr, sr)

    output_dir = tmp_path / "outputs"
    config = AnalysisConfig()

    result = analyze(
        vocal_path=vocal_path,
        instrumental_path=instr_path,
        config=config,
        output_dir=output_dir,
    )

    # JSON files exist and are valid
    seg_path = output_dir / "defect_segments.json"
    plan_path = output_dir / "repair_plan.json"
    assert seg_path.exists()
    assert plan_path.exists()

    with open(seg_path) as f:
        segs = json.load(f)
    with open(plan_path) as f:
        plan = json.load(f)

    assert isinstance(segs, list)
    assert isinstance(plan, list)
    assert len(segs) == len(plan)

    # Each segment has required fields
    for seg in segs:
        assert "segment_id" in seg
        assert "start_ms" in seg
        assert "end_ms" in seg
        assert "primary_class" in seg
        assert seg["primary_class"] in ("harsh", "distortion", "both")

    # Each plan entry has route
    for entry in plan:
        assert "recommended_route" in entry
        assert isinstance(entry["recommended_route"], list)
        assert len(entry["recommended_route"]) > 0
