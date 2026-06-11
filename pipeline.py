"""Command-line and Python API for the generalized body measurement pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from vision_analyzer_v6.scripts.vision_analyzer_v6 import FullVisualAnalysis, VisionAnalyzerV6


def analyze_image(
    image_path: str,
    real_height_cm: Optional[float] = None,
    character_name: str = "Character",
    profile_mode: bool = False,
    use_gpu: bool = False,
) -> FullVisualAnalysis:
    """Analyze any new character image and return structured 33-metric results."""

    analyzer = VisionAnalyzerV6(
        real_height_cm=real_height_cm,
        profile_mode=profile_mode,
        use_gpu=use_gpu,
    )
    return analyzer.get_full_visual_analysis(image_path, character_name=character_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized 33-metric body analysis pipeline")
    parser.add_argument("image", help="Path to an image containing a visible person")
    parser.add_argument("--height", type=float, help="Known real height in centimeters")
    parser.add_argument("--name", default="Character", help="Character/person label for the report")
    parser.add_argument("--profile-mode", action="store_true", help="Lower confidence threshold for side/profile images")
    parser.add_argument("--gpu", action="store_true", help="Request MediaPipe GPU delegate")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown report")
    parser.add_argument("--output", help="Optional output file for JSON or Markdown")
    args = parser.parse_args()

    result = analyze_image(
        args.image,
        real_height_cm=args.height,
        character_name=args.name,
        profile_mode=args.profile_mode,
        use_gpu=args.gpu,
    )
    payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False) if args.json else result.markdown_report
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
