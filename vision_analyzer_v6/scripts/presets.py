"""Analysis presets for the generalized body measurement pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass(frozen=True)
class AnalysisPreset:
    """Runtime knobs that trade speed for landmark recall and report detail."""

    name: str
    min_detection_confidence: Optional[float]
    profile_mode: bool
    enable_segmentation: bool
    quality_audit: bool
    description: str


PRESETS: Dict[str, AnalysisPreset] = {
    "standard": AnalysisPreset(
        name="standard",
        min_detection_confidence=None,
        profile_mode=False,
        enable_segmentation=True,
        quality_audit=True,
        description="Balanced single-image analysis for clear full-body photos.",
    ),
    "advanced": AnalysisPreset(
        name="advanced",
        min_detection_confidence=0.20,
        profile_mode=True,
        enable_segmentation=True,
        quality_audit=True,
        description="@codex adv high-recall mode for harder poses, side/profile images, and maximum validation metadata.",
    ),
}


ALIASES = {
    "default": "standard",
    "std": "standard",
    "adv": "advanced",
    "@codex adv": "advanced",
    "codex adv": "advanced",
}


def resolve_preset(value: Optional[Union[str, AnalysisPreset]]) -> AnalysisPreset:
    """Resolve CLI/API preset names and the literal '@codex adv' command."""

    if isinstance(value, AnalysisPreset):
        return value
    normalized = (value or "standard").strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in PRESETS:
        allowed = ", ".join(sorted([*PRESETS, "@codex adv"]))
        raise ValueError(f"Unknown analysis preset '{value}'. Use one of: {allowed}.")
    return PRESETS[normalized]
