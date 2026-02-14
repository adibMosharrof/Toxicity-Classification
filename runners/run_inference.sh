#!/bin/bash

# Inference Runner Script
# This script runs the inference pipeline using Hydra config

# Get the absolute path to the project root
PROJECT_ROOT="/u/siddique-d1/adib/jigsaw"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

# Use venv python
PYTHON_BIN=".venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Virtual environment not found at $PYTHON_BIN"
    exit 1
fi

echo "Using venv python to run inference..."
"$PYTHON_BIN" src/inference/inference_engine.py "$@"

echo "Inference completed!"
