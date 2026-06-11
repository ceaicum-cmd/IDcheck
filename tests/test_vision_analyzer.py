from __future__ import annotations

from landmark_provider.scripts.landmark_provider import _confidence, _runtime_dependency_error
from vision_analyzer_v6.scripts.vision_analyzer_v6 import VisionAnalyzerV6


def _blank_landmarks():
    return [{"x": 0.5, "y": 0.1 + i * 0.001, "z": 0.0, "visibility": 1.0} for i in range(33)]


def _synthetic_pose_2d():
    points = _blank_landmarks()
    points[0] = {"x": 0.50, "y": 0.08, "z": 0.0}
    points[11] = {"x": 0.40, "y": 0.26, "z": -0.02}
    points[12] = {"x": 0.60, "y": 0.26, "z": -0.02}
    points[23] = {"x": 0.38, "y": 0.55, "z": 0.02}
    points[24] = {"x": 0.62, "y": 0.55, "z": 0.02}
    points[25] = {"x": 0.43, "y": 0.75, "z": 0.01}
    points[26] = {"x": 0.57, "y": 0.75, "z": 0.01}
    points[27] = {"x": 0.45, "y": 0.92, "z": 0.0}
    points[28] = {"x": 0.55, "y": 0.92, "z": 0.0}
    return points


def _synthetic_pose_world():
    points = _blank_landmarks()
    points[0] = {"x": 0.0, "y": 0.95, "z": 0.0}
    points[11] = {"x": -0.18, "y": 0.55, "z": -0.02}
    points[12] = {"x": 0.18, "y": 0.55, "z": -0.02}
    points[23] = {"x": -0.19, "y": 0.05, "z": 0.05}
    points[24] = {"x": 0.19, "y": 0.05, "z": 0.05}
    points[25] = {"x": -0.12, "y": -0.40, "z": 0.02}
    points[26] = {"x": 0.12, "y": -0.40, "z": 0.02}
    points[27] = {"x": -0.10, "y": -0.95, "z": 0.0}
    points[28] = {"x": 0.10, "y": -0.95, "z": 0.0}
    return points


def test_2d_fallback_generates_calibrated_metrics():
    analyzer = VisionAnalyzerV6(real_height_cm=170)
    body = analyzer.analyze_body_proportions(
        [],
        image_landmarks=_synthetic_pose_2d(),
        image_meta={"width_px": 1000, "height_px": 1800},
    )
    metrics = analyzer.build_33_curve_metrics(body)

    assert body["scale_source"] == "height_calibrated_2d"
    assert metrics["height_cm"] == 170.0
    assert metrics["full_hip_circumference_cm"] > metrics["waist_circumference_cm"]
    assert "Overall Curve Balance Score" in analyzer.render_33_metric_report("Test", metrics, analyzer.build_body_dna(metrics))


def test_world_landmarks_preferred_over_2d_fallback():
    analyzer = VisionAnalyzerV6(real_height_cm=168)
    body = analyzer.analyze_body_proportions(
        _synthetic_pose_world(),
        image_landmarks=_synthetic_pose_2d(),
        image_meta={"width_px": 1000, "height_px": 1800},
    )

    assert body["scale_source"] == "height_calibrated"
    assert body["measurements"]["height_cm"] == 168.0
    assert body["measurements"]["shoulder_width_cm"] > 0


def test_landmark_confidence_and_runtime_error_message():
    assert _confidence(profile_mode=True, min_detection_confidence=None) == 0.25
    assert _confidence(profile_mode=False, min_detection_confidence=4.0) == 0.99
    error = _runtime_dependency_error(OSError("libGLESv2.so.2: cannot open shared object file"))
    assert "libGLESv2.so.2" in str(error)
    assert "libgles2" in str(error)
