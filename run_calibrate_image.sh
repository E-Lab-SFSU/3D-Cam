#!/bin/bash
# Run script for calibrate_image.py on Linux/Raspberry Pi

set -e

if [ ! -d "venv" ]; then
    ./setup_venv.sh
fi

source venv/bin/activate
python calibrate_image.py

