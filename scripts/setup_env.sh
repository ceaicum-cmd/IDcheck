#!/usr/bin/env bash
set -euo pipefail

# IDcheck / MK2 setup helper.
# Run from the repository root after cloning the repo.

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[setup] using Python: $($PYTHON_BIN --version 2>&1)"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Avoid OpenCV GUI wheel conflicts in headless/Codex/container environments.
python -m pip uninstall -y opencv-python opencv-contrib-python || true
python -m pip install --upgrade opencv-contrib-python-headless opencv-python-headless

# Editable install uses pyproject.toml and exposes the body-measure console command.
python -m pip install -e .

python - <<'PY'
from pipeline import analyze_image
print("[setup] import check OK: pipeline.analyze_image available")
PY

echo "[setup] complete"
echo "Example: body-measure image.jpg --height 157 --name Subject --profile-mode --json --output report.json"
