# truelock/scripts/truelock.py
"""
TrueLock v2.2
Enforces character consistency by comparing analysis against canonical state.
"""

from typing import Dict, Any
import json
from datetime import datetime

from ..vision_analyzer_v6.scripts.vision_analyzer_v6 import VisionAnalyzerV6


class TrueLock:
    def __init__(self, canonical_state_path: str = "canonical_state.json"):
        self.canonical = self._load_canonical(canonical_state_path)
        self.analyzer = VisionAnalyzerV6(canonical_state_path)
        self.version = "2.2"

    def _load_canonical(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def check_true_lock(self, image_path: str) -> Dict[str, Any]:
        analysis = self.analyzer.get_full_visual_analysis(image_path)

        canonical_body = self.canonical.get("body_geometry_canonical", {})
        current_body = analysis.body_geometry_canonical

        body_deviation = self._calculate_body_deviation(current_body, canonical_body)
        overall_score = body_deviation

        locked = overall_score < 0.15

        return {
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "locked": locked,
            "overall_deviation": round(overall_score, 4),
            "body_deviation": round(body_deviation, 4),
            "recommendation": "ACCEPT" if locked else "REVIEW or REGENERATE",
            "current_metrics": current_body,
        }

    def _calculate_body_deviation(self, current: Dict, canonical: Dict) -> float:
        if not current or not canonical:
            return 1.0

        current_whr = current.get("measurements", {}).get("waist_to_hip_ratio", 0)
        canonical_whr = canonical.get("measurements", {}).get("waist_to_hip_ratio", 0.665)

        whr_diff = abs(current_whr - canonical_whr)
        return min(whr_diff * 5, 1.0)