#!/bin/bash
# Run script for z_histogram.py on Linux/Raspberry Pi

set -e

if [ ! -d "venv" ]; then
    ./setup_venv.sh
fi

source venv/bin/activate
python z_histogram.py

