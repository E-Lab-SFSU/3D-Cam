#!/usr/bin/env bash

# Setup script to provision Picamera2 stack and project virtual environment.
# Safe to re-run; uses apt for system packages and recreates the Python venv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[1/4] Updating package index..."
sudo apt update

echo "[2/4] Installing Picamera2 / libcamera dependencies via apt..."
sudo apt install -y \
  python3-picamera2 \
  python3-libcamera \
  python3-libcamera-apps \
  python3-kms++ \
  python3-rpi.gpio \
  libcamera-tools \
  libcamera-apps \
  ffmpeg

echo "[3/4] Recreating project virtual environment with system packages..."
rm -rf venv
python3 -m venv --system-site-packages venv

echo "[4/4] Activating venv and installing Python requirements..."
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Activate the environment with: source venv/bin/activate"
echo "Run the capture CLI via: ./capture_raspi_cli.sh"

