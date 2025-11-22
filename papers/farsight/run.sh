#!/bin/bash
# run.sh - FarSight MLLM Setup and Execution Script

set -e  # Exit on error

echo "🚀 FarSight MLLM - Setup and Run Script"
echo "========================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ UV installed successfully!"
    echo "⚠️  Please restart your terminal or run: source ~/.zshrc"
    exit 0
fi

echo "✅ UV is installed"

# Create virtual environment with UV if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with UV..."
    uv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies using UV
echo "📥 Installing dependencies with UV..."
uv pip install -e .

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Available commands:"
echo "  python run_farsight_llava.py --config configs/default.yaml"
echo "  python scripts/preprocess_data.py"
echo "  python scripts/synthesize_hallucination.py"
echo "  pytest tests/"
echo ""

# If arguments are provided, run them
if [ $# -gt 0 ]; then
    echo "▶️  Running: $@"
    exec "$@"
else
    echo "💡 To run a specific command, use: ./run.sh <command>"
    echo "   Example: ./run.sh python run_farsight_llava.py --config configs/default.yaml"
fi
