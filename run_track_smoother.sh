#!/bin/bash
# Run script for track_smoother.py on Linux/Raspberry Pi

set -e

if [ ! -d "venv" ]; then
    ./setup_venv.sh
fi

source venv/bin/activate
python track_smoother.py

