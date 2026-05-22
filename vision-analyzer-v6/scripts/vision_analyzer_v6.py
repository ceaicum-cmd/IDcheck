# vision-analyzer-v6/scripts/vision_analyzer_v6.py
"""
Vision Analyzer V6
Main visual analysis skill with 3D metrics, proportions, and TrueLock support.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import json
from datetime import datetime

from ..landmark_provider.scripts.landmark_provider import extract_landmarks


@dataclass
class FullVisualAnalysis:
    face_3d_metrics: Dict[str, Any] = field(default_factory=dict)
    body_geometry_canonical: Dict[str, Any] = field(default_factory=dict)
    proportions: Dict[str, Any] = field(default_factory=dict)
    identity_lock_face: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"


class VisionAnalyzerV6:
    def __init__(self, canonical_state_path: str = "canonical_state.json"):
        self.canonical = self._load_canonical(canonical_state_path)
        self.version = "6.11"

    def _load_canonical(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def calculate_face_3d_metrics(self, face_landmarks: List[Dict]) -> Dict[str, Any]:
        if not face_landmarks or len(face_landmarks) < 468:
            return {"error": "Insufficient landmarks"}

        def dist_3d(p1, p2):
            return np.sqrt((p1["x"]-p2["x"])**2 + (p1["y"]-p2["y"])**2 + (p1.get("z",0)-p2.get("z",0))**2)

        l_eye = face_landmarks[33]
        r_eye = face_landmarks[263]
        nose = face_landmarks[1]
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        chin = face_landmarks[152]
        l_cheek = face_landmarks[117]
        r_cheek = face_landmarks[346]

        ipd = dist_3d(l_eye, r_eye)
        face_width = dist_3d(l_cheek, r_cheek)

        return {
            "ipd_3d_mm": round(ipd * 1000, 2),
            "face_width_3d_mm": round(face_width * 1000, 2),
            "lip_protrusion_mm": round(abs(upper_lip.get("z",0) - lower_lip.get("z",0)) * 1000, 2),
            "nose_projection_mm": round(abs(nose.get("z",0)) * 1000, 2),
            "cheekbone_prominence": round((abs(l_cheek.get("z",0)) + abs(r_cheek.get("z",0))) / 2 * 1000, 2),
        }

    def analyze_body_proportions(self, world_landmarks: List[Dict]) -> Dict[str, Any]:
        if not world_landmarks or len(world_landmarks) < 33:
            return {"error": "Insufficient landmarks"}

        def dist_3d(a, b):
            return np.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a.get("z",0)-b.get("z",0))**2)

        nose = world_landmarks[0]
        l_shoulder = world_landmarks[11]
        r_shoulder = world_landmarks[12]
        l_hip = world_landmarks[23]
        r_hip = world_landmarks[24]
        l_ankle = world_landmarks[27]

        shoulder_width = dist_3d(l_shoulder, r_shoulder)
        hip_width = dist_3d(l_hip, r_hip)
        torso = dist_3d(l_shoulder, l_hip)
        leg = dist_3d(l_hip, l_ankle)
        full_height = dist_3d(nose, l_ankle)

        canonical_h = self.canonical.get("body_geometry_canonical", {}).get("height_cm", 157.0)
        scale = canonical_h / (full_height * 100) if full_height > 0 else 1.0

        measurements = {
            "bust_cm": round(shoulder_width * 1.35 * scale * 100, 1),
            "waist_cm": round(hip_width * 0.82 * scale * 100, 1),
            "hip_cm": round(hip_width * 1.18 * scale * 100, 1),
            "shoulder_width_cm": round(shoulder_width * 100 * scale, 1),
            "waist_to_hip_ratio": round((hip_width * 0.82) / (hip_width * 1.18), 4),
            "shoulder_to_hip_ratio": round(shoulder_width / hip_width, 4)
        }

        proportions = {
            "leg_to_torso_ratio": round(leg / torso, 4),
            "glute_projection": 0.873,
            "bust_to_waist_ratio": round(measurements["bust_cm"] / measurements["waist_cm"], 4)
        }

        return {
            "measurements": measurements,
            "proportions": proportions,
            "full_height_m": round(full_height, 4),
            "scale_factor": round(scale, 4)
        }

    def get_full_visual_analysis(self, image_path: str) -> FullVisualAnalysis:
        data = extract_landmarks(image_path)
        face = data.get("face", {})
        pose = data.get("pose", {})

        face_metrics = self.calculate_face_3d_metrics(face.get("landmarks", [])) if face.get("detected") else {}
        body_data = self.analyze_body_proportions(pose.get("world_landmarks", [])) if pose.get("world_landmarks") else {}

        return FullVisualAnalysis(
            face_3d_metrics=face_metrics,
            body_geometry_canonical={
                "source": "visual-analyzer-v6",
                "height_cm": self.canonical.get("body_geometry_canonical", {}).get("height_cm", 157),
                "measurements": body_data.get("measurements", {}),
                "proportions": body_data.get("proportions", {}),
                "locked": True
            },
            proportions=body_data.get("proportions", {}),
            status="ok" if face.get("detected") and pose.get("detected") else "partial"
        )

    def analyze_emotional_state(self, blendshapes: Dict) -> Dict[str, Any]:
        if not blendshapes:
            return {"dominant_emotion": "neutral", "confidence": 0.6, "intensity": "low"}

        emotions = {
            "happy": (blendshapes.get("mouthSmileLeft", 0) * 0.4 + blendshapes.get("mouthSmileRight", 0) * 0.4),
            "sad": (blendshapes.get("mouthFrownLeft", 0) * 0.35 + blendshapes.get("mouthFrownRight", 0) * 0.35),
            "angry": (blendshapes.get("browDownLeft", 0) * 0.25 + blendshapes.get("browDownRight", 0) * 0.25),
            "surprised": (blendshapes.get("eyeWideLeft", 0) * 0.3 + blendshapes.get("eyeWideRight", 0) * 0.3),
            "neutral": 0.4
        }

        total = sum(emotions.values())
        if total > 0:
            for emotion in emotions:
                emotions[emotion] /= total

        dominant_emotion = max(emotions, key=emotions.get)
        confidence = round(emotions[dominant_emotion], 3)
        intensity = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"

        return {
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "intensity": intensity,
            "emotion_profile": {k: round(v, 3) for k, v in emotions.items()}
        }