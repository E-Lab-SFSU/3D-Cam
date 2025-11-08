#!/usr/bin/env bash

# Simple launcher for the Picamera2 CLI capture tool.
# Ensures the script runs from the repository root with the proper Python path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "venv/bin/activate" ]; then
  echo "[INFO] Activating local virtual environment..."
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

python apps/capture_raspi_cli.py "$@"

