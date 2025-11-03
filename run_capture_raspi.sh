#!/bin/bash
# Run script for capture_raspi.py on Linux/Raspberry Pi

set -e

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup_venv.sh
fi

source venv/bin/activate
python capture_raspi.py

