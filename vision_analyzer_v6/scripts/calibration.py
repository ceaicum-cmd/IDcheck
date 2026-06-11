"""Calibration helpers for 3D world landmarks."""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def dist_3d(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt(
        (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a.get("z", 0.0) - b.get("z", 0.0)) ** 2
    )


def get_scale_factor(world_landmarks: List[Dict[str, float]], real_height_cm: Optional[float] = None) -> Dict[str, float | str]:
    target_height = float(real_height_cm or 165.0)
    if not world_landmarks or len(world_landmarks) < 33:
        return {"cm_per_world_meter": 100.0, "source": "default_world_meter", "height_cm": target_height}
    nose = world_landmarks[0]
    ankle = max((world_landmarks[27], world_landmarks[28]), key=lambda p: dist_3d(nose, p))
    proxy_height_m = dist_3d(nose, ankle) * 1.08
    if proxy_height_m <= 0:
        return {"cm_per_world_meter": 100.0, "source": "default_world_meter", "height_cm": target_height}
    return {"cm_per_world_meter": target_height / proxy_height_m, "source": "height_calibrated", "height_cm": target_height}
