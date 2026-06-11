#!/usr/bin/env bash
set -euo pipefail

# Build a source-only portable archive without committing binary artifacts.
# Usage:
#   scripts/build_portable_zip.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/dist"
ZIP_NAME="${ZIP_NAME:-IDcheck-portable.zip}"
ZIP_PATH="$OUT_DIR/$ZIP_NAME"

mkdir -p "$OUT_DIR"
rm -f "$ZIP_PATH"

cd "$ROOT_DIR"

python - <<'PY'
from pathlib import Path
required = [
    Path("pipeline.py"),
    Path("pyproject.toml"),
    Path("README.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing required files: {missing}")
print("[zip] required files present")
PY

zip -r "$ZIP_PATH" . \
  -x ".git/*" \
  -x ".venv/*" \
  -x "venv/*" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x ".pytest_cache/*" \
  -x ".mypy_cache/*" \
  -x ".ruff_cache/*" \
  -x "dist/*" \
  -x "build/*" \
  -x "*.egg-info/*" \
  -x "*.zip" \
  -x "*.pyc"

echo "[zip] wrote $ZIP_PATH"
