"""TrueLock v2.3 consistency checks for generalized 33-metric analyses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from vision_analyzer_v6.scripts.vision_analyzer_v6 import VisionAnalyzerV6


class TrueLock:
    def __init__(self, canonical_state_path: str = "canonical_state.json", real_height_cm: Optional[float] = None):
        self.canonical = self._load_canonical(canonical_state_path)
        self.analyzer = VisionAnalyzerV6(canonical_state_path, real_height_cm=real_height_cm)
        self.version = "2.3"

    def _load_canonical(self, path: str) -> Dict[str, Any]:
        candidate = Path(path)
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
        return {}

    def check_true_lock(self, image_path: str, character_name: str = "Character") -> Dict[str, Any]:
        analysis = self.analyzer.get_full_visual_analysis(image_path, character_name=character_name)
        canonical_metrics = self.canonical.get("curve_metrics_33") or self.canonical.get("body_geometry_canonical", {}).get("measurements", {})
        current_metrics = analysis.curve_metrics_33
        body_deviation = self._calculate_body_deviation(current_metrics, canonical_metrics)
        locked = body_deviation < 0.15
        return {
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "locked": locked,
            "overall_deviation": round(body_deviation, 4),
            "body_deviation": round(body_deviation, 4),
            "recommendation": "ACCEPT" if locked else "REVIEW or REGENERATE",
            "current_metrics": current_metrics,
            "status": analysis.status,
        }

    def _calculate_body_deviation(self, current: Dict[str, Any], canonical: Dict[str, Any]) -> float:
        if not current or not canonical:
            return 1.0
        weighted_keys = {
            "height_cm": 0.16,
            "shoulder_width_cm": 0.12,
            "waist_circumference_cm": 0.18,
            "full_hip_circumference_cm": 0.18,
            "waist_to_hip_ratio": 0.20,
            "max_glute_projection_cm": 0.10,
            "overall_curve_balance_score": 0.06,
        }
        score = 0.0
        weight_used = 0.0
        for key, weight in weighted_keys.items():
            if key not in current or key not in canonical:
                continue
            base = max(abs(float(canonical[key])), 1.0)
            score += min(abs(float(current[key]) - float(canonical[key])) / base, 1.0) * weight
            weight_used += weight
        return score / weight_used if weight_used else 1.0
