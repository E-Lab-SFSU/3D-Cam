#!/bin/bash
# Run script for smooth_tracks.py on Linux/Raspberry Pi

set -e

if [ ! -d "venv" ]; then
    ./setup_venv.sh
fi

source venv/bin/activate
export PYTHONPATH=$(pwd)
python apps/smooth_tracks.py

