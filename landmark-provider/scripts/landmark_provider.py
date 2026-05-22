# landmark-provider/scripts/landmark_provider.py
"""
Landmark Provider v2.2
Provides optimized MediaPipe landmark extraction for face and pose.
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision, BaseOptions
import numpy as np
from typing import Dict, Any, Optional
import cv2
import os

# ====================== CONFIG ======================
FACE_MODEL_PATH = "/home/workdir/attachments/face_landmarker.task"
POSE_MODEL_PATH = "/home/workdir/attachments/pose_landmarker_full.task"

USE_GPU = True
MIN_DETECTION_CONFIDENCE = 0.5

# ====================== SINGLETON LANDMARKERS ======================
_face_landmarker = None
_pose_landmarker = None

def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        delegate = BaseOptions.Delegate.GPU if USE_GPU else BaseOptions.Delegate.CPU
        base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH, delegate=delegate)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
        )
        _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker

def _get_pose_landmarker():
    global _pose_landmarker
    if _pose_landmarker is None:
        delegate = BaseOptions.Delegate.GPU if USE_GPU else BaseOptions.Delegate.CPU
        base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH, delegate=delegate)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        )
        _pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker

# ====================== FAST IMAGE LOADING ======================
def _load_image_fast(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

# ====================== PUBLIC API ======================
def extract_landmarks(image_path: str, use_gpu: bool = True, profile_mode: bool = False) -> Dict[str, Any]:
    """
    Extract face and pose landmarks with world coordinates.
    
    Args:
        profile_mode: If True, uses lower confidence thresholds for better side-profile detection.
    """
    global USE_GPU, MIN_DETECTION_CONFIDENCE
    USE_GPU = use_gpu

    if profile_mode:
        MIN_DETECTION_CONFIDENCE = 0.25

    image = _load_image_fast(image_path)

    face_result = _get_face_landmarker().detect(image)
    pose_result = _get_pose_landmarker().detect(image)

    return {
        "face": _process_face_result(face_result),
        "pose": _process_pose_result(pose_result),
        "inference_device": "GPU" if USE_GPU else "CPU",
        "profile_mode_used": profile_mode
    }

def _process_face_result(result) -> Dict[str, Any]:
    if not result.face_landmarks:
        return {"detected": False}
    return {
        "detected": True,
        "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in result.face_landmarks[0]],
        "landmark_count": len(result.face_landmarks[0])
    }

def _process_pose_result(result) -> Dict[str, Any]:
    if not result.pose_landmarks:
        return {"detected": False}

    world_landmarks = None
    if result.pose_world_landmarks:
        world_landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in result.pose_world_landmarks[0]]

    return {
        "detected": True,
        "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in result.pose_landmarks[0]],
        "world_landmarks": world_landmarks,
        "landmark_count": len(result.pose_landmarks[0])
    }

def clear_landmarkers():
    global _face_landmarker, _pose_landmarker
    _face_landmarker = None
    _pose_landmarker = None