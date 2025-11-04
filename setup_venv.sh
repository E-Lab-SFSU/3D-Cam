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

# Install dependencies one by one (to handle failures gracefully)
echo "Installing dependencies..."

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

# Install scipy
echo "Installing scipy..."
if pip install "scipy>=1.11.0"; then
    echo "✓ scipy installed"
else
    echo "✗ scipy installation failed"
    INSTALLED_ALL=false
fi

echo ""
if [[ "$INSTALLED_ALL" == true ]]; then
    echo "✓ All dependencies installed successfully!"
else
    echo "⚠ Some dependencies had issues (see above)."
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
echo "  python apps/visualize_3d.py"

