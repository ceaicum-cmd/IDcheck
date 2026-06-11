# Body Measurement & Analysis Pipeline

A generalized, character-agnostic pipeline for extracting MediaPipe face/body landmarks, calibrating body measurements in centimeters, and generating a production-ready 33-metric proportion report for any new visible person/character image.

## What changed in the generalized version

- Valid Python packages now use importable underscore names: `landmark_provider` and `vision_analyzer_v6`.
- The MediaPipe `.task` models from `mk2-metricforge-portable.zip` are bundled under `landmark_provider/models/` and can also be overridden with environment variables.
- `VisionAnalyzerV6(real_height_cm=...)` is wired correctly for height-based centimeter calibration.
- The pipeline produces a structured `curve_metrics_33` dictionary plus the requested four-table Markdown report.
- If MediaPipe world landmarks are unavailable, the analyzer now falls back to height-calibrated 2D pose landmarks instead of dropping to generic defaults.
- Landmark extraction is fully input-driven: there are no hardcoded character profiles.
- The CLI can emit Markdown or JSON, supports side/profile images, and recognizes `@codex adv` / `--adv` for high-recall advanced analysis.

## Installation

```bash
pip install -e .
```

If your environment has OpenCV GUI conflicts, reinstall the headless wheel:

```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install -U opencv-contrib-python-headless opencv-python-headless
```

MediaPipe Tasks may also require system OpenGL ES runtime libraries such as `libGLESv2.so.2` on minimal Linux containers. Install the OS package that provides those libraries before running image inference.

## Python usage

```python
from pipeline import analyze_image

result = analyze_image(
    "new_character.jpg",
    real_height_cm=168,
    character_name="New Character",
    profile_mode=True,
)

print(result.curve_metrics_33)
print(result.markdown_report)
```

## CLI usage

Markdown report:

```bash
python pipeline.py new_character.jpg --height 168 --name "New Character" --profile-mode
```

JSON payload:

```bash
python pipeline.py new_character.jpg --height 168 --name "New Character" --json --output analysis.json
```


## `@codex adv` advanced mode

Use the `@codex adv` preset when you want the highest-recall generalized analysis path for difficult inputs, side/profile poses, lower-contrast images, or production validation runs:

```bash
python pipeline.py new_character.jpg --height 168 --name "New Character" --preset "@codex adv"
# equivalent shortcut
python pipeline.py new_character.jpg --height 168 --name "New Character" --adv
```

In Python:

```python
result = analyze_image(
    "new_character.jpg",
    real_height_cm=168,
    character_name="New Character",
    analysis_preset="@codex adv",
)
```

Advanced mode enables profile-aware low-threshold landmark recall, segmentation-backed validation, and an expanded `validation.quality_audit` block with metric confidence, scale source, pose visibility, mask ratio, and review flags.

## Model configuration

By default, the repository uses bundled models:

- `landmark_provider/models/face_landmarker.task`
- `landmark_provider/models/pose_landmarker_full.task`

Override them when needed:

```bash
export FACE_LANDMARKER_MODEL=/path/to/face_landmarker.task
export POSE_LANDMARKER_MODEL=/path/to/pose_landmarker_full.task
```

## Output structure

`FullVisualAnalysis` includes:

- `face_3d_metrics`
- `body_geometry_canonical`
- `proportions`
- `curve_metrics_33`
- `body_dna`
- `identity_lock_face`
- `validation`
- `markdown_report`
- `status`

The Markdown report keeps the required professional structure:

1. Core Linear Curve Metrics (10)
2. Depth & Projection Metrics (8)
3. Angle & Curvature Metrics (8)
4. Ratio & Shape Index Metrics (7)
5. Body DNA
6. Signature Details
7. Enhanced Identity Lock Block

## Notes

For real image-based measurement quality, provide a clear full-body image and a known height when possible. The analyzer prefers MediaPipe world landmarks, then falls back to height-calibrated 2D pose landmarks, 2D visual cues, segmentation metadata, and height calibration to generate repeatable centimeter outputs for new subjects. Runtime dependency failures are surfaced as actionable errors that name the missing system library.
