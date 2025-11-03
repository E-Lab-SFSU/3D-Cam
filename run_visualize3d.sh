#!/bin/bash
# Run script for visualize3d.py on Linux/Raspberry Pi
# Automatically sets up venv if needed and runs the program

set -e

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup_venv.sh
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Running visualize3d.py..."
python visualize3d.py

