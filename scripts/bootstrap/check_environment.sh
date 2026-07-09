#!/usr/bin/env bash
# check_environment.sh - Validates Python environment dependencies.

set -euo pipefail

echo "========================================="
echo "Checking local environment dependencies..."
echo "========================================="

# 1. Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH."
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python 3 found (version $PYTHON_VERSION)"

# 2. Check key libraries
echo -n "Checking PyYAML... "
if python3 -c "import yaml" &> /dev/null; then
    echo "✅ Installed"
else
    echo "❌ Missing (run: pip install pyyaml)"
fi

echo -n "Checking jsonschema... "
if python3 -c "import jsonschema" &> /dev/null; then
    echo "✅ Installed"
else
    echo "⚠️  Missing (jsonschema not found. Will use basic key-check fallbacks. Run: pip install jsonschema)"
fi

echo -n "Checking project package source... "
# Add current directory to pythonpath to verify
if PYTHONPATH="src" python3 -c "import finetuning_lifecycle" &> /dev/null; then
    echo "✅ Found and importable"
else
    echo "⚠️  Project package not in path. Run: export PYTHONPATH=\"\$(pwd)/src:\${PYTHONPATH:-}\" to load it locally."
fi

echo "========================================="
echo "Environment verification complete."
echo "========================================="
