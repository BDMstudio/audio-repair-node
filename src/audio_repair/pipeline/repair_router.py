"""Repair router — 4 routing rules mapping defect segments to repair plans."""

from __future__ import annotations

from ..types import DefectSegment, RepairPlan, RepairStep, AnalysisConfig


def route_repairs(
    segments: list[DefectSegment],
    config: AnalysisConfig | None = None,
) -> list[RepairPlan]:
    """Apply routing rules to each DefectSegment and produce RepairPlan objects.

    Routing Rules (from PRD)
    ------------------------
    Rule 1: harsh only  → ["deess_light"]
    Rule 2: distortion only → ["declip_light"], optionally + ["voice_denoise_light"]
    Rule 3: both high   → ["declip_light", "voice_denoise_light?", "deess_light"]
    Rule 4: below threshold → ["skip"]
    """
    if config is None:
        config = AnalysisConfig()

    plans: list[RepairPlan] = []
    for seg in segments:
        route, modifiers = _route_single(seg, config)
        plans.append(RepairPlan(
            segment_id=seg.segment_id,
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            primary_class=seg.primary_class,
            harsh_score=seg.harsh_score,
            distortion_score=seg.distortion_score,
            modifiers=seg.modifiers + modifiers,
            recommended_route=route,
            confidence=seg.confidence,
            review_needed=seg.review_needed,
        ))

    return plans


def _route_single(
    seg: DefectSegment, config: AnalysisConfig
) -> tuple[list[RepairStep], list[str]]:
    """Determine the repair route for a single segment."""
    h = seg.harsh_score
    d = seg.distortion_score
    extra_mods: list[str] = []

    # Rule 3: both high
    if h >= config.harsh_high and d >= config.distortion_high:
        route = [
            RepairStep(tool="rx", action="declip_light"),
            RepairStep(tool="rx", action="voice_denoise_light"),
            RepairStep(tool="pure_deess", action="deess_light"),
        ]
        return route, extra_mods

    # Rule 1: harsh only
    if h >= config.harsh_high and d < config.distortion_low:
        route = [RepairStep(tool="pure_deess", action="deess_light")]
        return route, extra_mods

    # Rule 2: distortion only
    if d >= config.distortion_high and h < config.harsh_low:
        route: list[RepairStep] = [RepairStep(tool="rx", action="declip_light")]
        # Noise modifier: add denoise if distortion is high
        if _has_noise_modifier(seg):
            route.append(RepairStep(tool="rx", action="voice_denoise_light"))
            extra_mods.append("noise_high")
        return route, extra_mods

    # Rule 4: below threshold → skip
    if h < config.harsh_low and d < config.distortion_low:
        route = [RepairStep(tool="skip", action="skip")]
        return route, extra_mods

    # Grey zone — partial overlap between thresholds
    # Apply the dominant class route at light level
    if h >= d:
        route = [RepairStep(tool="pure_deess", action="deess_light")]
    else:
        route = [RepairStep(tool="rx", action="declip_light")]
        if _has_noise_modifier(seg):
            route.append(RepairStep(tool="rx", action="voice_denoise_light"))
            extra_mods.append("noise_high")

    return route, extra_mods


def _has_noise_modifier(seg: DefectSegment) -> bool:
    """Check if the segment has a noise-related modifier."""
    return (
        "noise_high" in seg.modifiers
        or "distortion_peak" in seg.modifiers
        or seg.distortion_score > 0.70
    )
