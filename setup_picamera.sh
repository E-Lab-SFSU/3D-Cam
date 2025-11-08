#!/usr/bin/env bash

# Setup script to provision Picamera2 stack and project virtual environment.
# Safe to re-run; uses apt for system packages and recreates the Python venv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[1/4] Updating package index..."
sudo apt update

echo "[2/4] Installing Picamera2 / libcamera dependencies via apt..."
APT_PACKAGES=(
  python3-picamera2
  python3-libcamera
  libcamera-tools
  libcamera-apps
  python3-kms++
  python3-rpi.gpio
  python3-opengl
  python3-pyqt6
  python3-pyqt6.qtquick
  python3-pyqt6.qtopengl
  python3-simplejpeg
  ffmpeg
)

for pkg in "${APT_PACKAGES[@]}"; do
  if apt-cache show "$pkg" >/dev/null 2>&1; then
    echo "  - Installing ${pkg}..."
    sudo apt install -y "$pkg"
  else
    echo "  - [WARN] ${pkg} not found in repositories, skipping."
  fi
done

echo "[3/4] Recreating project virtual environment with system packages..."
rm -rf venv
python3 -m venv --system-site-packages venv

echo "[4/4] Activating venv and installing Python requirements..."
cat <<'EOF' > venv/setup_env.sh
#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/bin/activate"
pip install --upgrade pip
pip install -r "$(dirname "${BASH_SOURCE[0]}")/../requirements.raspi.txt"
EOF
chmod +x venv/setup_env.sh
./venv/setup_env.sh

echo "Setup complete. Activate the environment with: source venv/bin/activate"
echo "Run the capture CLI via: ./capture_raspi_cli.sh"

