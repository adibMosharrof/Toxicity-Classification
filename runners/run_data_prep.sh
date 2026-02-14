#!/bin/bash

# Data Preparation Runner Script
# This script runs the simple data preparation pipeline using Hydra config

# causes script o exit immediately if a command exits with a non-zero status
set -e

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

echo "Using venv python to run data preparation..."
"$PYTHON_BIN" src/dataprep/simple_data_prep.py "$@"

echo "Data preparation completed!"
