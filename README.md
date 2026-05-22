# Body Measurement & Analysis Pipeline

A standalone pipeline for extracting 3D body landmarks, performing detailed visual analysis, and measuring proportions from any image containing human bodies.

## Overview

This project provides tools to:

- Extract precise 3D body landmarks using MediaPipe
- Calculate real-world measurements in centimeters (with height-based scaling)
- Analyze body proportions (WHR, shoulder-to-hip, leg-to-torso, etc.)
- Detect facial landmarks and emotional expressions when a face is visible
- Support side-profile images with enhanced detection modes
- Run custom TensorFlow Lite models directly

It is designed to be **character-agnostic** and can be used on any photo of a person.

## Skills Included

| Skill                  | Purpose                                           | Key Features |
|------------------------|---------------------------------------------------|--------------|
| `landmark-provider`    | MediaPipe landmark extraction                     | Pose World Landmarks (3D), Face landmarks, Profile mode, Hand support |
| `vision-analyzer-v6`   | 3D body & face analysis                           | Measurements in cm, proportions, IPD, emotional state |
| `truelock`             | Optional consistency checking                     | Deviation scoring (can be used or ignored) |

## Quick Start

### Installation

```bash
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from landmark_provider.scripts.landmark_provider import extract_landmarks
from vision_analyzer_v6.scripts.vision_analyzer_v6 import VisionAnalyzerV6

# Extract landmarks (supports side profiles)
data = extract_landmarks("your_image.jpg", profile_mode=True)

# Full analysis
analyzer = VisionAnalyzerV6()
result = analyzer.get_full_visual_analysis("your_image.jpg")

print(result.body_geometry_canonical)
print(result.proportions)
```

## TensorFlow Lite Support

The project includes a utility module for running custom TFLite models:

```python
from utils.tflite_utils import TFLiteModel

model = TFLiteModel("path/to/your/model.tflite")
output = model.predict(input_data)
```

## Features

- Real 3D world landmarks from MediaPipe
- Height-based scaling for accurate cm measurements
- Body proportion analysis (WHR, ratios, etc.)
- Facial metrics when available (IPD, symmetry, expressions)
- Emotional state detection from facial blendshapes
- Direct TensorFlow Lite inference support
- Works on front and side profile images

## Project Structure

```
.
├── pyproject.toml
├── requirements.txt
├── README.md
├── utils/
├── landmark-provider/
├── vision-analyzer-v6/
└── truelock/
```

## License

MIT