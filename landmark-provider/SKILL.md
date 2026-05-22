---
name: landmark-provider
description: Provides MediaPipe Tasks API landmark extraction for face and body. Supports pose_world_landmarks in real 3D meters, face landmarks with depth, key point extraction and deviation scoring. Use for TrueLock consistency checks and visual analysis pipelines.
---

# Landmark Provider v2.2

## Overview
Extracts precise 3D anatomical landmarks from images using MediaPipe Tasks API (FaceLandmarker + PoseLandmarker). Returns both normalized face landmarks and real-world 3D pose world landmarks in meters.

## When to use
- When you need raw landmark data for further analysis
- For TrueLock pipelines that require 3D body and face data
- When building visual analysis or consistency checking systems

## Instructions
- Always reuse landmarker instances for performance
- Prefer GPU delegate when available
- Return both face landmarks (with z) and pose world landmarks
- Support graceful degradation when landmarks are not detected