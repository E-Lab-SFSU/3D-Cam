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

# Install dependencies via requirements.txt when available
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    echo "Using requirements.txt..."
    if pip install -r requirements.txt; then
        echo "✓ requirements.txt installed successfully!"
    else
        echo "✗ Failed to install from requirements.txt"
        echo "Attempting fallback installation of core packages..."
        FALLBACK_PACKAGES=("numpy>=2.0.0,<3.0.0" "opencv-python>=4.8.0" "matplotlib>=3.9.0" "scipy>=1.11.0" "scikit-image==0.24.0")
        for pkg in "${FALLBACK_PACKAGES[@]}"; do
            echo "Installing ${pkg}..."
            if pip install "${pkg}"; then
                echo "  ✓ ${pkg}"
            else
                echo "  ✗ ${pkg} installation failed"
            fi
        done
    fi
else
    echo "requirements.txt not found. Installing core packages individually..."
    PACKAGES=("numpy>=2.0.0,<3.0.0" "opencv-python>=4.8.0" "matplotlib>=3.9.0" "scipy>=1.11.0" "scikit-image==0.24.0")
    for pkg in "${PACKAGES[@]}"; do
        echo "Installing ${pkg}..."
        if pip install "${pkg}"; then
            echo "  ✓ ${pkg}"
        else
            echo "  ✗ ${pkg} installation failed"
        fi
    done
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

