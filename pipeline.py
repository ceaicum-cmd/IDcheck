"""Command-line and Python API for the generalized body measurement pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from vision_analyzer_v6.scripts.vision_analyzer_v6 import FullVisualAnalysis, VisionAnalyzerV6


def analyze_image(
    image_path: str,
    real_height_cm: Optional[float] = None,
    character_name: str = "Character",
    profile_mode: bool = False,
    use_gpu: bool = False,
    analysis_preset: Optional[str] = None,
) -> FullVisualAnalysis:
    """Analyze any new character image and return structured 33-metric results."""

    analyzer = VisionAnalyzerV6(
        real_height_cm=real_height_cm,
        profile_mode=profile_mode,
        use_gpu=use_gpu,
        analysis_preset=analysis_preset,
    )
    return analyzer.get_full_visual_analysis(image_path, character_name=character_name)


def _normalize_codex_adv_args(argv: list[str]) -> list[str]:
    """Allow the literal command prefix `@codex adv` before normal CLI args."""

    if len(argv) >= 2 and argv[0].lower() == "@codex" and argv[1].lower() == "adv":
        return ["--adv", *argv[2:]]
    if argv and argv[0].lower() == "@codex adv":
        return ["--adv", *argv[1:]]
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized 33-metric body analysis pipeline")
    parser.add_argument("image", help="Path to an image containing a visible person")
    parser.add_argument("--height", type=float, help="Known real height in centimeters")
    parser.add_argument("--name", default="Character", help="Character/person label for the report")
    parser.add_argument("--profile-mode", action="store_true", help="Lower confidence threshold for side/profile images")
    parser.add_argument("--gpu", action="store_true", help="Request MediaPipe GPU delegate")
    parser.add_argument("--preset", default="standard", help="Analysis preset: standard, advanced, or literal @codex adv")
    parser.add_argument("--adv", action="store_true", help="Shortcut for --preset advanced (@codex adv mode)")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown report")
    parser.add_argument("--output", help="Optional output file for JSON or Markdown")
    args = parser.parse_args(_normalize_codex_adv_args(sys.argv[1:]))

    result = analyze_image(
        args.image,
        real_height_cm=args.height,
        character_name=args.name,
        profile_mode=args.profile_mode,
        use_gpu=args.gpu,
        analysis_preset="advanced" if args.adv else args.preset,
    )
    payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False) if args.json else result.markdown_report
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
