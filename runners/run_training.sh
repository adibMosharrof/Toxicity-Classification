#!/bin/bash

# Training Runner Script
# This script runs the training pipeline using Hydra config

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

# Use accelerate launch for distributed training on multiple GPUs
echo "Using accelerate launch for distributed training on 2 GPUs..."
"$PYTHON_BIN" -m accelerate.commands.launch --multi_gpu --num_processes 2 src/training/training_engine.py "$@"

echo "Training completed!"
