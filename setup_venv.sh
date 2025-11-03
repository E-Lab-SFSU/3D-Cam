#!/bin/bash
# Setup script for 3D-Cam project
# Creates a virtual environment and installs dependencies

set -e  # Exit on error

echo "Setting up 3D-Cam virtual environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install it first."
    exit 1
fi

# Check if venv module is available
if ! python3 -m venv --help &> /dev/null; then
    echo "Error: python3-venv is not installed."
    echo "Please install it with: sudo apt install python3-venv python3-full"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "You can now run your scripts while the virtual environment is active:"
echo "  python visualize3d.py"

