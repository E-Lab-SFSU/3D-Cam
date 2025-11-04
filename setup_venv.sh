#!/bin/bash
# Setup script for 3D-Cam project
# Creates a virtual environment and installs dependencies

# Don't exit on error - we'll handle failures gracefully for each package
set +e

echo "Setting up 3D-Cam virtual environment..."

# Check if Python 3 is available (critical - exit on error)
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install it first."
    exit 1
fi

# Check if venv module is available (critical - exit on error)
if ! python3 -m venv --help &> /dev/null 2>&1; then
    echo "Error: python3-venv is not installed."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "On macOS, venv should be included. Try: python3 -m ensurepip --upgrade"
    else
        echo "Please install it with: sudo apt install python3-venv python3-full"
    fi
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
IS_MACOS=false
if [[ "$OSTYPE" == "darwin"* ]]; then
    IS_MACOS=true
    echo "Detected macOS..."
    echo "Note: SciPy may require special handling on macOS."
    echo ""
fi

# Install dependencies one by one to handle failures gracefully
INSTALLED_ALL=true

# Install numpy first (required by others)
echo "Installing numpy..."
if pip install "numpy>=2.0.0,<3.0.0"; then
    echo "✓ numpy installed"
else
    echo "✗ numpy installation failed"
    INSTALLED_ALL=false
fi

# Install opencv-python
echo "Installing opencv-python..."
if pip install "opencv-python>=4.8.0"; then
    echo "✓ opencv-python installed"
else
    echo "✗ opencv-python installation failed"
    INSTALLED_ALL=false
fi

# Install matplotlib
echo "Installing matplotlib..."
if pip install "matplotlib>=3.9.0"; then
    echo "✓ matplotlib installed"
else
    echo "✗ matplotlib installation failed"
    INSTALLED_ALL=false
fi

# Install scipy (may fail on macOS with old clang)
echo "Installing scipy..."
if pip install "scipy>=1.11.0"; then
    echo "✓ scipy installed"
else
    echo "✗ scipy installation failed"
    if [[ "$IS_MACOS" == true ]]; then
        echo ""
        echo "⚠ SciPy failed to install (likely needs clang 15.0+)."
        echo "This is expected on macOS with older Xcode Command Line Tools."
        echo ""
        echo "To fix SciPy on macOS, run:"
        echo "  source venv/bin/activate"
        echo "  conda install scipy -c conda-forge"
        echo ""
        echo "Or see setup_macos.md for other solutions."
        echo ""
        echo "Note: The program may still work if SciPy is only needed for"
        echo "      the Hungarian algorithm (optional pairing method)."
        INSTALLED_ALL=false
    else
        INSTALLED_ALL=false
    fi
fi

echo ""
if [[ "$INSTALLED_ALL" == true ]]; then
    echo "✓ All dependencies installed successfully!"
else
    echo "⚠ Some dependencies had issues (see above)."
    if [[ "$IS_MACOS" == true ]]; then
        echo "If SciPy failed, install it via conda as shown above."
    fi
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

