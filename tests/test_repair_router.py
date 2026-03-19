"""Tests for repair_router."""

from audio_repair.pipeline.repair_router import route_repairs
from audio_repair.types import AnalysisConfig, DefectSegment


def _seg(harsh: float, distortion: float) -> DefectSegment:
    return DefectSegment(
        segment_id="test",
        start_ms=0,
        end_ms=1000,
        primary_class="harsh" if harsh > distortion else "distortion",
        harsh_score=harsh,
        distortion_score=distortion,
    )


def test_harsh_only_route():
    """High harsh, low distortion → deess_light."""
    config = AnalysisConfig()
    segs = [_seg(harsh=0.75, distortion=0.20)]
    plan = route_repairs(segs, config)
    actions = [s.action for s in plan[0].recommended_route]
    assert "deess_light" in actions


def test_distortion_only_route():
    """High distortion, low harsh → declip_light."""
    config = AnalysisConfig()
    segs = [_seg(harsh=0.20, distortion=0.70)]
    plan = route_repairs(segs, config)
    actions = [s.action for s in plan[0].recommended_route]
    assert "declip_light" in actions


def test_both_high_route():
    """Both high → full chain: declip + denoise + deess."""
    config = AnalysisConfig()
    segs = [_seg(harsh=0.75, distortion=0.70)]
    plan = route_repairs(segs, config)
    actions = [s.action for s in plan[0].recommended_route]
    assert "declip_light" in actions
    assert "deess_light" in actions


def test_skip_route():
    """Both low → skip."""
    config = AnalysisConfig()
    segs = [_seg(harsh=0.20, distortion=0.20)]
    plan = route_repairs(segs, config)
    actions = [s.action for s in plan[0].recommended_route]
    assert "skip" in actions


def test_serialisable():
    """RepairPlan.to_dict() produces valid dict."""
    config = AnalysisConfig()
    segs = [_seg(harsh=0.75, distortion=0.70)]
    plan = route_repairs(segs, config)
    d = plan[0].to_dict()
    assert "segment_id" in d
    assert isinstance(d["recommended_route"], list)
    assert isinstance(d["recommended_route"][0], dict)
