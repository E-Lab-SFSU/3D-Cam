#!/bin/bash
# Quick fix script for macOS - installs essential packages manually

echo "Fixing macOS installation..."

# Activate virtual environment
source venv/bin/activate

echo "Installing numpy..."
pip install "numpy>=2.0.0,<3.0.0"

echo ""
echo "Installing opencv-python..."
pip install "opencv-python>=4.8.0"

echo ""
echo "Installing matplotlib..."
pip install "matplotlib>=3.9.0"

echo ""
echo "Attempting SciPy (may fail - that's OK)..."
pip install "scipy>=1.11.0" || echo "SciPy failed - will install via conda"

echo ""
echo "✓ Essential packages installed!"
echo ""
echo "If SciPy failed, install it via conda:"
echo "  conda install scipy -c conda-forge"
echo ""
echo "Then try: ./run_visualize3d.sh"

