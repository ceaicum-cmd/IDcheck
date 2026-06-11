#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper around the body-measure CLI.
# Usage:
#   scripts/run_body_measure.sh image.jpg 157 "Subject" report.json

IMAGE_PATH="${1:?image path required}"
HEIGHT_CM="${2:-157}"
NAME="${3:-Subject}"
OUTPUT_PATH="${4:-analysis.json}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

body-measure "$IMAGE_PATH" \
  --height "$HEIGHT_CM" \
  --name "$NAME" \
  --profile-mode \
  --adv \
  --json \
  --output "$OUTPUT_PATH"

echo "[run] wrote $OUTPUT_PATH"
