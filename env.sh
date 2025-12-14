#!/bin/bash

# THEMAP Environment Setup Script
# This script sets up a Python virtual environment using uv and installs all dependencies
#
# Usage:
#   source env.sh    (recommended - keeps environment activated)
#   bash env.sh      (requires manual activation after)

set -e  # Exit on error

echo "=========================================="
echo "THEMAP Environment Setup"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"

    echo "✅ uv installed successfully"
    echo ""
else
    echo "✅ uv is already installed"
    echo ""
fi

# Check Python version
echo "🐍 Checking Python version..."
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
else
    PYTHON_CMD="python3"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "   Using Python: $PYTHON_VERSION"
echo ""

# Create virtual environment with uv
echo "🔨 Creating virtual environment in .venv..."
uv venv .venv --python $PYTHON_CMD
echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip and install build tools
echo "📦 Installing build tools..."
uv pip install --upgrade pip setuptools wheel
echo "✅ Build tools installed"
echo ""

# Install THEMAP with all dependencies
echo "📚 Installing THEMAP with all dependencies..."
echo "   This includes: core, ml, otdd, protein, dev, and test dependencies"
echo ""

# Install the package in editable mode with all optional dependencies
uv pip install -e ".[all,dev,test]"

echo ""
echo "✅ Installation complete!"
echo ""

# Minimal verification
echo "🔍 Verifying installation..."
python -c "import themap; from themap.data import MoleculeDataset; print(f'✅ THEMAP v{themap.__version__} imported successfully')"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "🎉 Setup Complete!"
    echo "=========================================="
    echo ""

    # Check if script was sourced
    if [ -n "$BASH_SOURCE" ] && [ "$0" != "$BASH_SOURCE" ]; then
        echo "✅ Environment is ACTIVE in your current shell"
        echo ""
        echo "Next steps:"
        echo "   python run_tests.py              # Run tests"
        echo "   python examples/basic/molecule_datasets_demo.py  # Try an example"
        echo ""
    else
        echo "To activate the environment, run:"
        echo "   source .venv/bin/activate"
        echo ""
        echo "Or run this script with 'source' to auto-activate:"
        echo "   source env.sh"
        echo ""
    fi
else
    echo "❌ Verification failed. Installation may be incomplete."
    exit 1
fi
