"""Measurement consistency helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class MeasurementProfile:
    height_cm: float
    shoulder_width_cm: float
    waist_circumference_cm: float
    full_hip_circumference_cm: float
    waist_to_hip_ratio: float


class MeasurementConsistency:
    def create_reference_profile(self, measurements: Dict[str, float]) -> MeasurementProfile:
        return MeasurementProfile(
            height_cm=float(measurements.get("height_cm", 0.0)),
            shoulder_width_cm=float(measurements.get("shoulder_width_cm", 0.0)),
            waist_circumference_cm=float(measurements.get("waist_circumference_cm", 0.0)),
            full_hip_circumference_cm=float(measurements.get("full_hip_circumference_cm", measurements.get("hip_circumference_cm", 0.0))),
            waist_to_hip_ratio=float(measurements.get("waist_to_hip_ratio", 0.0)),
        )

    def compare_with_reference(self, current: Dict[str, float], reference: MeasurementProfile) -> Dict[str, float | bool]:
        ref = asdict(reference)
        deviations = []
        for key, ref_value in ref.items():
            if ref_value == 0 or key not in current:
                continue
            deviations.append(min(abs(float(current[key]) - ref_value) / abs(ref_value), 1.0))
        overall_deviation = sum(deviations) / len(deviations) if deviations else 1.0
        return {"overall_score": round((1.0 - overall_deviation) * 100.0, 2), "is_consistent": overall_deviation < 0.15}
