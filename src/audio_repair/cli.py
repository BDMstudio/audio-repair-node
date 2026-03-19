"""Command-line interface for audio-repair."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .pipeline.analyzer import analyze


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``audio-repair analyze``."""
    parser = argparse.ArgumentParser(
        prog="audio-repair",
        description="AI Vocal De-artifact & Cleanup — defect detection and repair routing.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- analyze sub-command ---
    analyze_p = sub.add_parser("analyze", help="Analyze vocal/instrumental stems")
    analyze_p.add_argument(
        "--vocal", required=True, type=str, help="Path to vocal WAV/FLAC/OGG",
    )
    analyze_p.add_argument(
        "--instrumental", required=True, type=str, help="Path to instrumental WAV/FLAC/OGG",
    )
    analyze_p.add_argument(
        "--config", type=str, default=None, help="Path to JSON config override",
    )
    analyze_p.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory (default: outputs/)",
    )
    analyze_p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "analyze":
        config = load_config(args.config)
        result = analyze(
            vocal_path=args.vocal,
            instrumental_path=args.instrumental,
            config=config,
            output_dir=args.output_dir,
        )
        segments = result.get("defect_segments", [])
        plan = result.get("repair_plan", [])
        print(f"\n✅ Analysis complete — {len(segments)} defect segment(s), "
              f"{len(plan)} repair plan(s).")
        if plan:
            for p in plan:
                route = p.get("recommended_route", [])
                actions = " → ".join(
                    s["action"] if isinstance(s, dict) else str(s) for s in route
                )
                flag = " ⚠️  review" if p.get("review_needed") else ""
                sid = p.get("segment_id", "?")
                start = p.get("start_ms", 0)
                end = p.get("end_ms", 0)
                cls = p.get("primary_class", "?")
                conf = p.get("confidence", 0)
                print(
                    f"  {sid}: {start:.0f}–{end:.0f}ms "
                    f"[{cls}] → {actions} "
                    f"(confidence={conf:.2f}){flag}"
                )
