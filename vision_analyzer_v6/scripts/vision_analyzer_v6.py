"""Vision Analyzer V6: generalized body metrics and 33-metric reporting."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from landmark_provider.scripts.landmark_provider import extract_landmarks

Point = Dict[str, float]


@dataclass
class FullVisualAnalysis:
    face_3d_metrics: Dict[str, Any] = field(default_factory=dict)
    body_geometry_canonical: Dict[str, Any] = field(default_factory=dict)
    proportions: Dict[str, Any] = field(default_factory=dict)
    curve_metrics_33: Dict[str, Any] = field(default_factory=dict)
    body_dna: Dict[str, Any] = field(default_factory=dict)
    identity_lock_face: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    markdown_report: str = ""
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisionAnalyzerV6:
    """Character-agnostic analyzer for new images and new subjects."""

    def __init__(
        self,
        canonical_state_path: str = "canonical_state.json",
        real_height_cm: Optional[float] = None,
        use_gpu: bool = False,
        profile_mode: bool = False,
        enable_segmentation: bool = True,
    ) -> None:
        self.canonical = self._load_canonical(canonical_state_path)
        self.real_height_cm = real_height_cm
        self.use_gpu = use_gpu
        self.profile_mode = profile_mode
        self.enable_segmentation = enable_segmentation
        self.version = "6.12-generalized"

    def _load_canonical(self, path: str) -> Dict[str, Any]:
        candidate = Path(path)
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _dist_3d(a: Point, b: Point) -> float:
        return float(
            math.sqrt(
                (a["x"] - b["x"]) ** 2
                + (a["y"] - b["y"]) ** 2
                + (a.get("z", 0.0) - b.get("z", 0.0)) ** 2
            )
        )

    @staticmethod
    def _dist_2d_px(a: Point, b: Point, image_meta: Dict[str, int]) -> float:
        dx = (a["x"] - b["x"]) * image_meta.get("width_px", 1)
        dy = (a["y"] - b["y"]) * image_meta.get("height_px", 1)
        return float(math.sqrt(dx * dx + dy * dy))

    @staticmethod
    def _midpoint(a: Point, b: Point) -> Point:
        return {
            "x": (a["x"] + b["x"]) / 2.0,
            "y": (a["y"] + b["y"]) / 2.0,
            "z": (a.get("z", 0.0) + b.get("z", 0.0)) / 2.0,
        }

    @staticmethod
    def _angle_degrees(a: Point, b: Point, c: Point) -> float:
        ba = np.array([a["x"] - b["x"], a["y"] - b["y"], a.get("z", 0.0) - b.get("z", 0.0)])
        bc = np.array([c["x"] - b["x"], c["y"] - b["y"], c.get("z", 0.0) - b.get("z", 0.0)])
        denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
        if denom == 0:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(float(np.dot(ba, bc)) / denom, -1.0, 1.0))))

    def _scale_factor_cm_per_meter(self, world_landmarks: List[Point]) -> Tuple[float, str, float]:
        canonical_height = self.canonical.get("body_geometry_canonical", {}).get("height_cm")
        target_height = self.real_height_cm or canonical_height or 165.0
        if len(world_landmarks) < 33:
            return 100.0, "default_world_meter", float(target_height)

        nose = world_landmarks[0]
        left_ankle, right_ankle = world_landmarks[27], world_landmarks[28]
        ankle = left_ankle if self._dist_3d(nose, left_ankle) >= self._dist_3d(nose, right_ankle) else right_ankle
        measured_head_to_ankle_m = self._dist_3d(nose, ankle)
        full_height_proxy_m = measured_head_to_ankle_m * 1.08
        if full_height_proxy_m <= 0:
            return 100.0, "default_world_meter", float(target_height)
        return float(target_height) / full_height_proxy_m, "height_calibrated", float(target_height)

    def calculate_face_3d_metrics(self, face_landmarks: List[Point]) -> Dict[str, Any]:
        if not face_landmarks or len(face_landmarks) < 468:
            return {"detected": False, "error": "insufficient_landmarks"}

        l_eye, r_eye = face_landmarks[33], face_landmarks[263]
        nose, upper_lip, lower_lip = face_landmarks[1], face_landmarks[13], face_landmarks[14]
        chin, forehead = face_landmarks[152], face_landmarks[10]
        l_cheek, r_cheek = face_landmarks[117], face_landmarks[346]
        ipd = self._dist_3d(l_eye, r_eye)
        face_width = self._dist_3d(l_cheek, r_cheek)
        face_height = self._dist_3d(forehead, chin)

        return {
            "detected": True,
            "ipd_3d_mm": round(ipd * 1000, 2),
            "face_width_3d_mm": round(face_width * 1000, 2),
            "face_height_3d_mm": round(face_height * 1000, 2),
            "fwhr": round(face_width / face_height, 4) if face_height else None,
            "lip_ap_separation_mm": round(abs(upper_lip.get("z", 0.0) - lower_lip.get("z", 0.0)) * 1000, 2),
            "nose_projection_mm": round(abs(nose.get("z", 0.0)) * 1000, 2),
        }

    def analyze_body_from_image_landmarks(
        self,
        image_landmarks: List[Point],
        image_meta: Dict[str, int],
    ) -> Dict[str, Any]:
        """Build a centimeter estimate from normalized 2D landmarks when world landmarks are absent."""

        if not image_landmarks or len(image_landmarks) < 33:
            return {"measurements": {}, "proportions": {}, "angles": {}, "error": "insufficient_2d_landmarks"}

        target_height = float(self.real_height_cm or self.canonical.get("body_geometry_canonical", {}).get("height_cm") or 165.0)
        width_px = max(int(image_meta.get("width_px", 1)), 1)
        height_px = max(int(image_meta.get("height_px", 1)), 1)

        def px(point: Point) -> Tuple[float, float, float]:
            return (point["x"] * width_px, point["y"] * height_px, point.get("z", 0.0) * width_px)

        def dist_px(a: Point, b: Point) -> float:
            ax, ay, az = px(a)
            bx, by, bz = px(b)
            return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

        nose = image_landmarks[0]
        ankle = max((image_landmarks[27], image_landmarks[28]), key=lambda item: dist_px(nose, item))
        body_height_px = max(dist_px(nose, ankle) * 1.08, 1.0)
        cm_per_px = target_height / body_height_px

        l_shoulder, r_shoulder = image_landmarks[11], image_landmarks[12]
        l_hip, r_hip = image_landmarks[23], image_landmarks[24]
        l_knee, r_knee = image_landmarks[25], image_landmarks[26]
        l_ankle, r_ankle = image_landmarks[27], image_landmarks[28]
        shoulder_mid = self._midpoint(l_shoulder, r_shoulder)
        hip_mid = self._midpoint(l_hip, r_hip)
        knee_mid = self._midpoint(l_knee, r_knee)
        ankle_mid = self._midpoint(l_ankle, r_ankle)

        shoulder_width_cm = dist_px(l_shoulder, r_shoulder) * cm_per_px
        hip_width_cm = dist_px(l_hip, r_hip) * cm_per_px
        torso_cm = dist_px(shoulder_mid, hip_mid) * cm_per_px
        leg_cm = dist_px(hip_mid, ankle_mid) * cm_per_px
        thigh_cm = dist_px(hip_mid, knee_mid) * cm_per_px

        z_values = [p.get("z", 0.0) for p in image_landmarks[:33]]
        normalized_depth_px = (max(z_values) - min(z_values)) * width_px if z_values else 0.0
        visual_depth_cm = normalized_depth_px * cm_per_px
        hip_depth_cm = max(visual_depth_cm * 0.58, hip_width_cm * 0.50)
        waist_width_cm = max(hip_width_cm * 0.74, shoulder_width_cm * 0.62)
        waist_depth_cm = max(hip_depth_cm * 0.66, waist_width_cm * 0.42)

        bust_circumference_cm = math.pi * math.sqrt(
            2.0 * ((shoulder_width_cm * 0.44) ** 2 + (hip_depth_cm * 0.42) ** 2)
        )
        waist_circumference_cm = math.pi * math.sqrt(2.0 * ((waist_width_cm / 2.0) ** 2 + (waist_depth_cm / 2.0) ** 2))
        hip_circumference_cm = math.pi * math.sqrt(2.0 * ((hip_width_cm / 2.0) ** 2 + (hip_depth_cm / 2.0) ** 2))
        glute_projection_cm = max(0.0, hip_depth_cm - waist_depth_cm) * 0.70

        measurements = {
            "height_cm": round(target_height, 1),
            "shoulder_width_cm": round(shoulder_width_cm, 1),
            "bust_circumference_cm": round(bust_circumference_cm, 1),
            "waist_circumference_cm": round(waist_circumference_cm, 1),
            "hip_width_cm": round(hip_width_cm, 1),
            "hip_circumference_cm": round(hip_circumference_cm, 1),
            "torso_length_cm": round(torso_cm, 1),
            "leg_length_cm": round(leg_cm, 1),
            "thigh_length_cm": round(thigh_cm, 1),
            "body_depth_cm": round(visual_depth_cm, 1),
            "glute_projection_cm": round(glute_projection_cm, 1),
        }
        proportions = {
            "waist_to_hip_ratio": round(waist_circumference_cm / hip_circumference_cm, 4) if hip_circumference_cm else 0.0,
            "shoulder_to_hip_ratio": round(shoulder_width_cm / hip_width_cm, 4) if hip_width_cm else 0.0,
            "leg_to_torso_ratio": round(leg_cm / torso_cm, 4) if torso_cm else 0.0,
            "bust_to_waist_ratio": round(bust_circumference_cm / waist_circumference_cm, 4) if waist_circumference_cm else 0.0,
            "glute_projection_to_height_ratio": round(glute_projection_cm / target_height, 4) if target_height else 0.0,
        }
        angles = {
            "hip_knee_ankle_angle_deg": round(self._angle_degrees(hip_mid, knee_mid, ankle_mid), 1),
            "shoulder_hip_knee_angle_deg": round(self._angle_degrees(shoulder_mid, hip_mid, knee_mid), 1),
            "pelvic_tilt_proxy_deg": round(abs(math.degrees(math.atan2(l_hip.get("z", 0.0) - r_hip.get("z", 0.0), max(abs(l_hip["x"] - r_hip["x"]), 1e-6)))), 1),
        }
        return {
            "measurements": measurements,
            "proportions": proportions,
            "angles": angles,
            "scale_factor_cm_per_pixel": round(cm_per_px, 6),
            "scale_source": "height_calibrated_2d",
        }

    def analyze_body_proportions(
        self,
        world_landmarks: List[Point],
        image_landmarks: Optional[List[Point]] = None,
        image_meta: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        if not world_landmarks or len(world_landmarks) < 33:
            if image_landmarks and image_meta:
                return self.analyze_body_from_image_landmarks(image_landmarks, image_meta)
            return {"measurements": {}, "proportions": {}, "angles": {}, "error": "insufficient_landmarks"}

        cm_per_world_meter, scale_source, calibrated_height = self._scale_factor_cm_per_meter(world_landmarks)
        l_shoulder, r_shoulder = world_landmarks[11], world_landmarks[12]
        l_hip, r_hip = world_landmarks[23], world_landmarks[24]
        l_knee, r_knee = world_landmarks[25], world_landmarks[26]
        l_ankle, r_ankle = world_landmarks[27], world_landmarks[28]
        shoulder_mid = self._midpoint(l_shoulder, r_shoulder)
        hip_mid = self._midpoint(l_hip, r_hip)
        knee_mid = self._midpoint(l_knee, r_knee)
        ankle_mid = self._midpoint(l_ankle, r_ankle)

        shoulder_width_cm = self._dist_3d(l_shoulder, r_shoulder) * cm_per_world_meter
        hip_width_cm = self._dist_3d(l_hip, r_hip) * cm_per_world_meter
        torso_cm = self._dist_3d(shoulder_mid, hip_mid) * cm_per_world_meter
        leg_cm = self._dist_3d(hip_mid, ankle_mid) * cm_per_world_meter
        thigh_cm = self._dist_3d(hip_mid, knee_mid) * cm_per_world_meter

        z_values = [p.get("z", 0.0) for p in world_landmarks]
        body_depth_cm = (max(z_values) - min(z_values)) * cm_per_world_meter if z_values else 0.0
        hip_depth_cm = max(body_depth_cm * 0.58, hip_width_cm * 0.48)
        waist_width_cm = max(hip_width_cm * 0.72, shoulder_width_cm * 0.62)
        waist_depth_cm = max(hip_depth_cm * 0.66, waist_width_cm * 0.42)

        bust_circumference_cm = math.pi * math.sqrt(
            2.0 * ((shoulder_width_cm * 0.44) ** 2 + (hip_depth_cm * 0.44) ** 2)
        )
        waist_circumference_cm = math.pi * math.sqrt(2.0 * ((waist_width_cm / 2) ** 2 + (waist_depth_cm / 2) ** 2))
        hip_circumference_cm = math.pi * math.sqrt(2.0 * ((hip_width_cm / 2) ** 2 + (hip_depth_cm / 2) ** 2))
        glute_projection_cm = max(0.0, hip_depth_cm - waist_depth_cm) * 0.72

        if image_landmarks and image_meta and len(image_landmarks) >= 33:
            # 2D normalized cue improves side/profile outputs without binding to one subject.
            shoulder_px = self._dist_2d_px(image_landmarks[11], image_landmarks[12], image_meta)
            hip_px = self._dist_2d_px(image_landmarks[23], image_landmarks[24], image_meta)
            if shoulder_px > 1 and hip_px > 1:
                visual_hip_dominance = hip_px / shoulder_px
                hip_circumference_cm *= min(max(visual_hip_dominance / max(hip_width_cm / max(shoulder_width_cm, 1), 0.1), 0.92), 1.12)
                waist_circumference_cm *= min(max(1.0 / max(visual_hip_dominance, 0.8), 0.92), 1.08)

        measurements = {
            "height_cm": round(calibrated_height, 1),
            "shoulder_width_cm": round(shoulder_width_cm, 1),
            "bust_circumference_cm": round(bust_circumference_cm, 1),
            "waist_circumference_cm": round(waist_circumference_cm, 1),
            "hip_width_cm": round(hip_width_cm, 1),
            "hip_circumference_cm": round(hip_circumference_cm, 1),
            "torso_length_cm": round(torso_cm, 1),
            "leg_length_cm": round(leg_cm, 1),
            "thigh_length_cm": round(thigh_cm, 1),
            "body_depth_cm": round(body_depth_cm, 1),
            "glute_projection_cm": round(glute_projection_cm, 1),
        }
        whr = waist_circumference_cm / hip_circumference_cm if hip_circumference_cm else 0.0
        proportions = {
            "waist_to_hip_ratio": round(whr, 4),
            "shoulder_to_hip_ratio": round(shoulder_width_cm / hip_width_cm, 4) if hip_width_cm else 0.0,
            "leg_to_torso_ratio": round(leg_cm / torso_cm, 4) if torso_cm else 0.0,
            "bust_to_waist_ratio": round(bust_circumference_cm / waist_circumference_cm, 4) if waist_circumference_cm else 0.0,
            "glute_projection_to_height_ratio": round(glute_projection_cm / calibrated_height, 4) if calibrated_height else 0.0,
        }

        angles = {
            "hip_knee_ankle_angle_deg": round(self._angle_degrees(hip_mid, knee_mid, ankle_mid), 1),
            "shoulder_hip_knee_angle_deg": round(self._angle_degrees(shoulder_mid, hip_mid, knee_mid), 1),
            "pelvic_tilt_proxy_deg": round(abs(math.degrees(math.atan2(l_hip.get("z", 0) - r_hip.get("z", 0), max(self._dist_3d(l_hip, r_hip), 1e-6)))), 1),
        }
        return {
            "measurements": measurements,
            "proportions": proportions,
            "angles": angles,
            "scale_factor_cm_per_world_meter": round(cm_per_world_meter, 4),
            "scale_source": scale_source,
        }

    def build_33_curve_metrics(self, body_data: Dict[str, Any]) -> Dict[str, Any]:
        m = body_data.get("measurements", {})
        p = body_data.get("proportions", {})
        a = body_data.get("angles", {})
        h = float(m.get("height_cm", self.real_height_cm or 165.0))
        shoulder = float(m.get("shoulder_width_cm", h * 0.235))
        bust = float(m.get("bust_circumference_cm", h * 0.52))
        waist = float(m.get("waist_circumference_cm", h * 0.43))
        hip = float(m.get("hip_circumference_cm", h * 0.57))
        hip_width = float(m.get("hip_width_cm", h * 0.205))
        glute = float(m.get("glute_projection_cm", max(hip - waist, 0) * 0.11))
        whr = float(p.get("waist_to_hip_ratio", waist / hip if hip else 0.0))
        leg_to_torso = float(p.get("leg_to_torso_ratio", 1.45))
        scale = h / 165.0

        metrics = {
            "height_cm": round(h, 1),
            "shoulder_width_cm": round(shoulder, 1),
            "bust_circumference_cm": round(bust, 1),
            "underbust_circumference_cm": round(bust * 0.84, 1),
            "waist_circumference_cm": round(waist, 1),
            "high_hip_circumference_cm": round(hip * 0.93, 1),
            "full_hip_circumference_cm": round(hip, 1),
            "glute_max_circumference_cm": round(hip * (1.01 + min(glute / max(h, 1), 0.12)), 1),
            "upper_thigh_circumference_cm": round(max(hip * 0.54, h * 0.29), 1),
            "mid_thigh_circumference_cm": round(max(hip * 0.48, h * 0.255), 1),
            "waist_depth_ap_cm": round(max(waist * 0.27, 1.0), 1),
            "high_hip_depth_ap_cm": round(max(hip * 0.245, 1.0), 1),
            "glute_shelf_projection_cm": round(glute * 0.92, 1),
            "lower_glute_projection_cm": round(glute * 0.78, 1),
            "max_glute_projection_cm": round(glute, 1),
            "hip_depth_ap_cm": round(max(hip * 0.255, glute + waist * 0.18), 1),
            "thigh_depth_ap_cm": round(max(hip * 0.185, 15.0 * scale), 1),
            "lateral_hip_flare_cm": round(max((hip_width - shoulder * 0.72) / 2.0, 0.0), 1),
            "alpha_upper_glute_slope_deg": round(28.0 + min(glute * 1.15, 18.0), 1),
            "beta_lower_glute_projection_deg": round(24.0 + min(glute * 1.35, 20.0), 1),
            "glute_roundness_shelf_angle_deg": round(72.0 + min(glute * 1.8, 24.0), 1),
            "lumbar_s_curve_index": round(35.0 + (1.0 - min(whr, 1.0)) * 45.0 + glute * 1.25, 1),
            "pelvic_tilt_angle_deg": round(float(a.get("pelvic_tilt_proxy_deg", 11.0)), 1),
            "waist_cinch_severity_deg": round(18.0 + max(0.0, (hip - waist) / max(hip, 1.0)) * 42.0, 1),
            "glute_ham_tie_in_angle_deg": round(16.0 + min(glute * 0.55, 9.0), 1),
            "lateral_hip_flare_angle_deg": round(18.0 + max(0.0, hip_width - shoulder * 0.72) * 0.8, 1),
            "waist_to_hip_ratio": round(whr, 3),
            "bust_to_waist_ratio": round(bust / waist, 2) if waist else 0.0,
            "glute_projection_to_height_ratio": round(glute / h * 100.0, 2) if h else 0.0,
            "hourglass_curvature_index": round(max(0.0, (hip / max(waist, 1.0) - 1.0) * 100.0 + glute * 1.8), 1),
            "lower_body_dominance_score": round(50.0 + max(0.0, hip / max(bust, 1.0) - 1.0) * 80.0 + glute, 1),
            "cinch_to_curve_ratio": round((hip - waist) / max(glute, 1.0), 2),
            "overall_curve_balance_score": round(min(100.0, 55.0 + (1.0 - min(whr, 1.0)) * 40.0 + leg_to_torso * 4.0 + glute), 1),
        }
        metrics["body_shape"] = self._classify_shape(metrics)
        metrics["frame_size"] = self._classify_frame(h, shoulder)
        return metrics

    @staticmethod
    def _classify_frame(height_cm: float, shoulder_width_cm: float) -> str:
        frame_index = shoulder_width_cm / max(height_cm, 1.0)
        if height_cm < 160 and frame_index < 0.235:
            return "petite narrow frame"
        if frame_index >= 0.255:
            return "athletic broad frame"
        if height_cm >= 172:
            return "tall balanced frame"
        return "balanced medium frame"

    @staticmethod
    def _classify_shape(metrics: Dict[str, Any]) -> str:
        whr = float(metrics.get("waist_to_hip_ratio", 0.8))
        bust_waist = float(metrics.get("bust_to_waist_ratio", 1.1))
        lower = float(metrics.get("lower_body_dominance_score", 50.0))
        if whr <= 0.72 and bust_waist >= 1.15:
            return "hourglass"
        if lower >= 58 and whr <= 0.78:
            return "lower-body dominant hourglass"
        if whr >= 0.84:
            return "straight athletic"
        return "balanced proportional"

    def build_body_dna(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "shape_family": metrics.get("body_shape"),
            "frame": metrics.get("frame_size"),
            "waist_cinch": "high" if metrics.get("waist_to_hip_ratio", 1) <= 0.72 else "moderate",
            "s_curve_strength": "high" if metrics.get("lumbar_s_curve_index", 0) >= 58 else "moderate",
            "dominant_visual_cues": [
                f"WHR {metrics.get('waist_to_hip_ratio')}",
                f"hip {metrics.get('full_hip_circumference_cm')} cm",
                f"shoulder {metrics.get('shoulder_width_cm')} cm",
                f"curve score {metrics.get('overall_curve_balance_score')}",
            ],
        }

    def render_33_metric_report(self, character_name: str, metrics: Dict[str, Any], body_dna: Dict[str, Any]) -> str:
        tables: Iterable[Tuple[str, List[Tuple[str, str, str]]]] = [
            (
                "Core Linear Curve Metrics (10)",
                [
                    ("Height", "height_cm", "calibrated full-body scale"),
                    ("Shoulder Width", "shoulder_width_cm", "upper-frame anchor"),
                    ("Bust Circumference", "bust_circumference_cm", "upper-torso circumference"),
                    ("Underbust Circumference", "underbust_circumference_cm", "ribcage anchor"),
                    ("Waist Circumference", "waist_circumference_cm", "minimum waist line"),
                    ("High Hip Circumference", "high_hip_circumference_cm", "upper pelvis transition"),
                    ("Full Hip Circumference", "full_hip_circumference_cm", "widest hip line"),
                    ("Glute Max Circumference", "glute_max_circumference_cm", "posterior hip maximum"),
                    ("Upper Thigh Circumference", "upper_thigh_circumference_cm", "upper leg support"),
                    ("Mid Thigh Circumference", "mid_thigh_circumference_cm", "leg taper continuity"),
                ],
            ),
            (
                "Depth & Projection Metrics (8)",
                [
                    ("Waist Depth A-P", "waist_depth_ap_cm", "side-profile waist depth"),
                    ("High Hip Depth A-P", "high_hip_depth_ap_cm", "upper hip depth"),
                    ("Glute Shelf Projection", "glute_shelf_projection_cm", "upper posterior projection"),
                    ("Lower Glute Projection", "lower_glute_projection_cm", "lower posterior projection"),
                    ("Maximum Glute Projection", "max_glute_projection_cm", "peak posterior projection"),
                    ("Hip Depth A-P", "hip_depth_ap_cm", "full pelvis depth"),
                    ("Thigh Depth A-P", "thigh_depth_ap_cm", "upper leg depth"),
                    ("Lateral Hip Flare", "lateral_hip_flare_cm", "side flare from torso"),
                ],
            ),
            (
                "Angle & Curvature Metrics (8)",
                [
                    ("Upper Glute Slope", "alpha_upper_glute_slope_deg", "upper curve slope"),
                    ("Lower Glute Projection", "beta_lower_glute_projection_deg", "lower curve angle"),
                    ("Glute Roundness Shelf", "glute_roundness_shelf_angle_deg", "roundness angle"),
                    ("Lumbar S-Curve Index", "lumbar_s_curve_index", "spine-to-hip curve strength"),
                    ("Pelvic Tilt", "pelvic_tilt_angle_deg", "pelvis rotation proxy"),
                    ("Waist Cinch Severity", "waist_cinch_severity_deg", "waist-to-hip transition"),
                    ("Glute-Ham Tie-In", "glute_ham_tie_in_angle_deg", "posterior thigh transition"),
                    ("Lateral Hip Flare Angle", "lateral_hip_flare_angle_deg", "frontal hip flare"),
                ],
            ),
            (
                "Ratio & Shape Index Metrics (7)",
                [
                    ("Waist-to-Hip Ratio", "waist_to_hip_ratio", "primary proportion index"),
                    ("Bust-to-Waist Ratio", "bust_to_waist_ratio", "upper cinch index"),
                    ("Glute Projection / Height", "glute_projection_to_height_ratio", "projection normalized to height"),
                    ("Hourglass Curvature Index", "hourglass_curvature_index", "combined waist/hip curvature"),
                    ("Lower-Body Dominance Score", "lower_body_dominance_score", "lower-vs-upper balance"),
                    ("Cinch-to-Curve Ratio", "cinch_to_curve_ratio", "waist reduction vs projection"),
                    ("Overall Curve Balance Score", "overall_curve_balance_score", "aggregate curve index"),
                ],
            ),
        ]
        lines = [
            "# Combined Body Analysis Report — 33 Curve-Focused Metrics",
            "## MK2 + MetricForge v2 | Real-Life cm Measurements",
            "",
            f"**Character name:** {character_name}",
            f"**Height:** {metrics['height_cm']} cm",
            f"**Frame:** {metrics['frame_size']}",
            f"**Generated:** {datetime.now(timezone.utc).date().isoformat()} UTC",
            "",
        ]
        metric_number = 1
        for title, rows in tables:
            lines.extend([f"## {title}", "| # | Metric | Value | Readout |", "|---:|---|---:|---|"])
            for label, key, note in rows:
                value = metrics[key]
                unit = " cm" if key.endswith("_cm") else "°" if key.endswith("_deg") else ""
                lines.append(f"| {metric_number} | {label} | {value}{unit} | {note} |")
                metric_number += 1
            lines.append("")
        lines.extend(
            [
                "## Body DNA",
                f"* **Shape family:** {body_dna['shape_family']}",
                f"* **Frame:** {body_dna['frame']}",
                f"* **Waist cinch:** {body_dna['waist_cinch']}",
                f"* **S-curve strength:** {body_dna['s_curve_strength']}",
                f"* **Dominant visual cues:** {', '.join(body_dna['dominant_visual_cues'])}",
                "",
                "## Signature Details (locked values)",
                f"* Height {metrics['height_cm']} cm; shoulder {metrics['shoulder_width_cm']} cm; waist {metrics['waist_circumference_cm']} cm; hip {metrics['full_hip_circumference_cm']} cm; WHR {metrics['waist_to_hip_ratio']}; curve score {metrics['overall_curve_balance_score']}.",
                "",
                "## Enhanced Identity Lock Block (copy-paste ready)",
                "```",
                f"{character_name}: height {metrics['height_cm']} cm, {metrics['frame_size']}, {metrics['body_shape']}; shoulder width {metrics['shoulder_width_cm']} cm; bust {metrics['bust_circumference_cm']} cm; waist {metrics['waist_circumference_cm']} cm; full hip {metrics['full_hip_circumference_cm']} cm; glute max {metrics['glute_max_circumference_cm']} cm; max posterior projection {metrics['max_glute_projection_cm']} cm; WHR {metrics['waist_to_hip_ratio']}; bust-to-waist {metrics['bust_to_waist_ratio']}; lumbar S-curve index {metrics['lumbar_s_curve_index']}; waist cinch severity {metrics['waist_cinch_severity_deg']} degrees; lateral hip flare {metrics['lateral_hip_flare_cm']} cm; overall curve balance score {metrics['overall_curve_balance_score']}.",
                "```",
            ]
        )
        return "\n".join(lines)

    def get_full_visual_analysis(self, image_path: str, character_name: str = "Character") -> FullVisualAnalysis:
        data = extract_landmarks(
            image_path,
            use_gpu=self.use_gpu,
            profile_mode=self.profile_mode,
            enable_segmentation=self.enable_segmentation,
        )
        face = data.get("face", {})
        pose = data.get("pose", {})
        face_metrics = self.calculate_face_3d_metrics(face.get("landmarks", [])) if face.get("detected") else {}
        body_data = self.analyze_body_proportions(
            pose.get("world_landmarks", []),
            image_landmarks=pose.get("landmarks", []),
            image_meta=data.get("image", {}),
        ) if pose.get("detected") else {"measurements": {}, "proportions": {}, "angles": {}, "error": "pose_not_detected"}
        curve_metrics = self.build_33_curve_metrics(body_data)
        body_dna = self.build_body_dna(curve_metrics)
        markdown = self.render_33_metric_report(character_name, curve_metrics, body_dna)
        detected = bool(face.get("detected")) + bool(pose.get("detected"))
        return FullVisualAnalysis(
            face_3d_metrics=face_metrics,
            body_geometry_canonical={
                "source": "vision-analyzer-v6",
                "version": self.version,
                "height_cm": curve_metrics.get("height_cm"),
                "measurements": body_data.get("measurements", {}),
                "angles": body_data.get("angles", {}),
                "proportions": body_data.get("proportions", {}),
                "scale_source": body_data.get("scale_source"),
                "locked": True,
            },
            proportions=body_data.get("proportions", {}),
            curve_metrics_33=curve_metrics,
            body_dna=body_dna,
            identity_lock_face={"face_detected": bool(face.get("detected")), "metrics": face_metrics},
            validation={
                "image": data.get("image", {}),
                "segmentation": data.get("segmentation", {}),
                "profile_mode_used": data.get("profile_mode_used"),
                "inference_device": data.get("inference_device"),
                "body_scale_source": body_data.get("scale_source"),
                "landmark_quality": {
                    "face_detected": bool(face.get("detected")),
                    "pose_detected": bool(pose.get("detected")),
                    "has_world_landmarks": bool(pose.get("world_landmarks")),
                    "pose_landmark_count": pose.get("landmark_count", 0),
                },
            },
            markdown_report=markdown,
            status="ok" if detected == 2 else "partial" if detected else "no_landmarks",
        )

    def analyze_emotional_state(self, blendshapes: Dict[str, float]) -> Dict[str, Any]:
        if not blendshapes:
            return {"dominant_emotion": "neutral", "confidence": 0.6, "intensity": "low"}
        emotions = {
            "happy": blendshapes.get("mouthSmileLeft", 0.0) * 0.4 + blendshapes.get("mouthSmileRight", 0.0) * 0.4,
            "sad": blendshapes.get("mouthFrownLeft", 0.0) * 0.35 + blendshapes.get("mouthFrownRight", 0.0) * 0.35,
            "angry": blendshapes.get("browDownLeft", 0.0) * 0.25 + blendshapes.get("browDownRight", 0.0) * 0.25,
            "surprised": blendshapes.get("eyeWideLeft", 0.0) * 0.3 + blendshapes.get("eyeWideRight", 0.0) * 0.3,
            "neutral": 0.4,
        }
        total = sum(emotions.values()) or 1.0
        normalized = {emotion: score / total for emotion, score in emotions.items()}
        dominant = max(normalized, key=normalized.get)
        confidence = round(normalized[dominant], 3)
        return {
            "dominant_emotion": dominant,
            "confidence": confidence,
            "intensity": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
            "emotion_profile": {k: round(v, 3) for k, v in normalized.items()},
        }
