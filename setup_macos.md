# macOS Setup Instructions

If you encounter issues installing SciPy on macOS (especially on Apple Silicon), use one of these solutions:

## Solution 1: Use Conda (Recommended for macOS)

Since you're already using conda, install SciPy via conda which has pre-built binaries:

```bash
# Activate your virtual environment first
source venv/bin/activate

# Install scipy via conda (pre-built, no compilation needed)
conda install scipy -c conda-forge

# Then install the rest via pip
pip install -r requirements.txt
```

## Solution 2: Update Xcode Command Line Tools

SciPy 1.16+ requires clang 15.0+. Update your Xcode Command Line Tools:

```bash
# Update Xcode Command Line Tools
xcode-select --install

# If that doesn't work, try:
softwareupdate --all --install --force

# Then try installing again
pip install -r requirements.txt
```

## Solution 3: Use Homebrew to Install SciPy Dependencies

```bash
# Install via Homebrew (if you have it)
brew install scipy

# Or install build dependencies
brew install gfortran
```

## Solution 4: Use Pre-built Wheels Only

Force pip to use only pre-built wheels (no building from source):

```bash
pip install --only-binary=all -r requirements.txt
```

If SciPy can't find a pre-built wheel, it will fail, but you can then use Solution 1 (conda).

## Quick Fix for Your Current Situation

Since you're already in a conda environment and SciPy failed to build:

```bash
# Make sure venv is activated
source venv/bin/activate

# Install SciPy via conda (has pre-built binaries)
conda install scipy -c conda-forge

# Install the rest
pip install numpy "numpy<3.0.0,>=2.0.0" opencv-python matplotlib
```

## Why This Happens

- macOS often has older clang versions
- SciPy 1.16+ requires clang 15.0+ to build from source
- Pre-built wheels avoid compilation issues
- Conda provides pre-built binaries for macOS

