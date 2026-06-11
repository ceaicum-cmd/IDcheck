"""
Generalized MediaPipe landmark extraction.

The provider is intentionally character-agnostic: every call accepts an arbitrary
image path and returns normalized image landmarks, 3D world landmarks when
MediaPipe provides them, optional segmentation metadata, and image metadata for
centimeter calibration. Model paths are bundled by default and can be overridden
with environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACE_MODEL_PATH = PACKAGE_ROOT / "models" / "face_landmarker.task"
DEFAULT_POSE_MODEL_PATH = PACKAGE_ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_DETECTION_CONFIDENCE = 0.5
PROFILE_DETECTION_CONFIDENCE = 0.25

_face_landmarker_cache: Dict[Tuple[bool, float], Any] = {}
_pose_landmarker_cache: Dict[Tuple[bool, bool, float], Any] = {}


def _resolve_model_path(env_name: str, default_path: Path) -> str:
    candidate = Path(os.environ.get(env_name, default_path)).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Required MediaPipe model not found: {candidate}. "
            f"Set {env_name} or place the model at {default_path}."
        )
    return str(candidate)


def _runtime_dependency_error(error: BaseException) -> RuntimeError:
    message = str(error)
    if "libGLESv2" in message:
        hint = (
            "MediaPipe Tasks could not load the OpenGL ES runtime library "
            "libGLESv2.so.2. Install the OS package that provides it "
            "(for example libgles2 on Debian/Ubuntu containers) and retry."
        )
    elif "libGL" in message:
        hint = (
            "OpenCV/MediaPipe could not load libGL. Install the OS package that "
            "provides libGL or use a compatible headless runtime."
        )
    else:
        hint = "MediaPipe/OpenCV runtime dependency failed to load."
    return RuntimeError(f"{hint} Original error: {message}")


def _load_runtime_modules():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import BaseOptions, vision
    except (ImportError, OSError) as exc:
        raise _runtime_dependency_error(exc) from exc

    return mp, python, BaseOptions, vision


def _delegate(use_gpu: bool, base_options: Any) -> Any:
    return base_options.Delegate.GPU if use_gpu else base_options.Delegate.CPU


def _confidence(profile_mode: bool, min_detection_confidence: Optional[float]) -> float:
    value = (
        min_detection_confidence
        if min_detection_confidence is not None
        else PROFILE_DETECTION_CONFIDENCE if profile_mode else DEFAULT_DETECTION_CONFIDENCE
    )
    return float(max(0.01, min(0.99, value)))


def _get_face_landmarker(use_gpu: bool, min_detection_confidence: float):
    key = (use_gpu, min_detection_confidence)
    if key not in _face_landmarker_cache:
        _, python, BaseOptions, vision = _load_runtime_modules()
        base_options = python.BaseOptions(
            model_asset_path=_resolve_model_path("FACE_LANDMARKER_MODEL", DEFAULT_FACE_MODEL_PATH),
            delegate=_delegate(use_gpu, BaseOptions),
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=min_detection_confidence,
        )
        try:
            _face_landmarker_cache[key] = vision.FaceLandmarker.create_from_options(options)
        except (ImportError, OSError) as exc:
            raise _runtime_dependency_error(exc) from exc
    return _face_landmarker_cache[key]


def _get_pose_landmarker(use_gpu: bool, enable_segmentation: bool, min_detection_confidence: float):
    key = (use_gpu, enable_segmentation, min_detection_confidence)
    if key not in _pose_landmarker_cache:
        _, python, BaseOptions, vision = _load_runtime_modules()
        base_options = python.BaseOptions(
            model_asset_path=_resolve_model_path("POSE_LANDMARKER_MODEL", DEFAULT_POSE_MODEL_PATH),
            delegate=_delegate(use_gpu, BaseOptions),
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            output_segmentation_masks=enable_segmentation,
        )
        try:
            _pose_landmarker_cache[key] = vision.PoseLandmarker.create_from_options(options)
        except (ImportError, OSError) as exc:
            raise _runtime_dependency_error(exc) from exc
    return _pose_landmarker_cache[key]


def _load_image_fast(image_path: str):
    mp, _, _, _ = _load_runtime_modules()
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        from PIL import Image

        with Image.open(path) as source:
            img_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Could not read image: {image_path}") from exc

    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    try:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    except (ImportError, OSError) as exc:
        raise _runtime_dependency_error(exc) from exc
    return image, {"width_px": int(img_rgb.shape[1]), "height_px": int(img_rgb.shape[0])}


def _landmark_to_dict(landmark: Any, include_visibility: bool = False) -> Dict[str, float]:
    item = {"x": float(landmark.x), "y": float(landmark.y), "z": float(landmark.z)}
    if include_visibility and hasattr(landmark, "visibility"):
        item["visibility"] = float(landmark.visibility)
    if include_visibility and hasattr(landmark, "presence"):
        item["presence"] = float(landmark.presence)
    return item


def _process_face_result(result: Any) -> Dict[str, Any]:
    if not getattr(result, "face_landmarks", None):
        return {"detected": False, "landmarks": [], "landmark_count": 0}

    blendshapes: Dict[str, float] = {}
    if getattr(result, "face_blendshapes", None):
        for category in result.face_blendshapes[0]:
            blendshapes[category.category_name] = float(category.score)

    return {
        "detected": True,
        "landmarks": [_landmark_to_dict(lm) for lm in result.face_landmarks[0]],
        "landmark_count": len(result.face_landmarks[0]),
        "blendshapes": blendshapes,
    }


def _process_pose_result(result: Any) -> Dict[str, Any]:
    if not getattr(result, "pose_landmarks", None):
        return {
            "detected": False,
            "landmarks": [],
            "world_landmarks": [],
            "has_world_landmarks": False,
            "landmark_count": 0,
        }

    world_landmarks = []
    if getattr(result, "pose_world_landmarks", None):
        world_landmarks = [_landmark_to_dict(lm) for lm in result.pose_world_landmarks[0]]

    return {
        "detected": True,
        "landmarks": [_landmark_to_dict(lm, include_visibility=True) for lm in result.pose_landmarks[0]],
        "world_landmarks": world_landmarks,
        "has_world_landmarks": bool(world_landmarks),
        "landmark_count": len(result.pose_landmarks[0]),
    }


def _process_segmentation(result: Any) -> Optional[Dict[str, Any]]:
    masks = getattr(result, "segmentation_masks", None)
    if not masks:
        return None
    mask_array = masks[0].numpy_view()
    person_pixels = int(np.sum(mask_array > 0.5))
    return {
        "mask_shape": tuple(int(v) for v in mask_array.shape),
        "person_pixels": person_pixels,
        "total_pixels": int(mask_array.size),
        "person_ratio": round(float(person_pixels) / float(mask_array.size), 4),
    }


def extract_landmarks(
    image_path: str,
    use_gpu: bool = False,
    profile_mode: bool = False,
    enable_segmentation: bool = True,
    min_detection_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract face, pose, world-pose, and optional segmentation metadata."""

    confidence = _confidence(profile_mode, min_detection_confidence)
    image, image_meta = _load_image_fast(image_path)
    face_result = _get_face_landmarker(use_gpu=use_gpu, min_detection_confidence=confidence).detect(image)
    pose_result = _get_pose_landmarker(
        use_gpu=use_gpu,
        enable_segmentation=enable_segmentation,
        min_detection_confidence=confidence,
    ).detect(image)
    segmentation = _process_segmentation(pose_result) if enable_segmentation else None

    payload: Dict[str, Any] = {
        "face": _process_face_result(face_result),
        "pose": _process_pose_result(pose_result),
        "image": image_meta,
        "inference_device": "GPU" if use_gpu else "CPU",
        "profile_mode_used": profile_mode,
        "min_detection_confidence": confidence,
    }
    if segmentation is not None:
        payload["segmentation"] = segmentation
    return payload


def clear_landmarkers() -> None:
    """Release cached MediaPipe task instances."""

    _face_landmarker_cache.clear()
    _pose_landmarker_cache.clear()
