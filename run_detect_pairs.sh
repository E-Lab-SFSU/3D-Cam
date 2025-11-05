#!/bin/bash
# Run script for detect_pairs.py on Linux/Raspberry Pi

set -e

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup_venv.sh
fi

source venv/bin/activate
export PYTHONPATH=$(pwd)
python apps/detect_pairs.py

