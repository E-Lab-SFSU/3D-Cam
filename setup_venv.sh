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

# Detect macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS..."
    echo "Note: SciPy may require special handling on macOS."
    echo "If SciPy installation fails, try installing it via conda:"
    echo "  conda install scipy -c conda-forge"
    echo ""
fi

# Try to install dependencies
if pip install -r requirements.txt; then
    echo ""
    echo "✓ All dependencies installed successfully!"
else
    echo ""
    echo "⚠ Some dependencies failed to install."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo ""
        echo "For macOS, try installing SciPy via conda:"
        echo "  source venv/bin/activate"
        echo "  conda install scipy -c conda-forge"
        echo "  pip install numpy \"numpy<3.0.0,>=2.0.0\" opencv-python matplotlib"
        echo ""
        echo "Or see setup_macos.md for more solutions."
    fi
    exit 1
fi

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

