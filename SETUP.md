# IDcheck setup

This repository is configured as a Python package for the generalized MK2 / RealMeasure body-measurement pipeline.

## Local setup

```bash
git clone https://github.com/ceaicum-cmd/IDcheck.git
cd IDcheck
bash scripts/setup_env.sh
```

The setup script creates `.venv`, upgrades packaging tools, removes GUI OpenCV wheels, installs the headless OpenCV wheels, installs the project with `pip install -e .`, and runs an import check for `pipeline.analyze_image`.

## Run an analysis

```bash
scripts/run_body_measure.sh image.jpg 157 "Subject" report.json
```

Equivalent direct command:

```bash
body-measure image.jpg --height 157 --name "Subject" --profile-mode --adv --json --output report.json
```

For a Markdown report instead of JSON, omit `--json`:

```bash
body-measure image.jpg --height 157 --name "Subject" --profile-mode --adv --output report.md
```

## Model files

The package expects MediaPipe task files under:

```text
landmark_provider/models/face_landmarker.task
landmark_provider/models/pose_landmarker_full.task
```

You can override them with environment variables:

```bash
export FACE_LANDMARKER_MODEL=/path/to/face_landmarker.task
export POSE_LANDMARKER_MODEL=/path/to/pose_landmarker_full.task
```

## Minimal Linux / Codex containers

If MediaPipe fails with `libGLESv2.so.2`, install the OS package that provides OpenGL ES runtime libraries. On Debian/Ubuntu-style containers this is commonly:

```bash
sudo apt-get update
sudo apt-get install -y libgles2 libglib2.0-0
```

Then rerun:

```bash
bash scripts/setup_env.sh
```

## Codex PR rule

Do not commit generated ZIPs, build folders, virtualenvs, model-cache folders, or binary outputs directly in Codex PRs. Keep PRs source/text-only. If a portable ZIP is needed, build it with the workflow or locally after merge.
