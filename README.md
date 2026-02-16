# Toxicity Classification with Custom Transformer

## Overview

This project implements a custom Transformer model from scratch (using PyTorch primitives) for binary toxicity classification on the Civil Comments dataset. 

## Quick Start

### Installation

This project uses **Poetry** for dependency management. All package specifications are in `pyproject.toml`.

**Requirements:**
- Python 3.11+
- Poetry >= 2.0.0

**Install with Poetry:**

```bash
# Install all dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Running the Project

### 1. Data Preparation

Prepare and clean the dataset before training:

```bash
./runners/run_data_prep.sh
```

**Output:**
```
outputs/data_prep/
├── 2026-02-15/16-46-17/
│   ├── train_split.csv      (1.26M samples)
│   ├── val_split.csv        (270K samples)
│   └── test_split.csv       (270K samples)
```

---

### 2. Model Training & Inference

Train the custom Pre-Layer Normalization Transformer (inference runs automatically upon completion):

```bash
./runners/run_training.sh
```

**Output:**
```
outputs/training/
├── 2026-02-15/23-13-53/
│   ├── final_model/                 # Best trained model
│   │   ├── pytorch_model.bin        # Model weights
│   │   ├── config.json
│   │   └── tokenizer.json
│   ├── checkpoint-30/               # Intermediate checkpoints
│   ├── checkpoint-60/
│   └── inference_results/           # Auto-generated after training
│       ├── predictions.csv          # Predictions on test set
│       ├── metrics.json             # F1, AUROC, precision, recall
│       └── results_summary.txt      # Summary report
```

**Training Configuration:**

Customize training in `config/training.yaml`:

```yaml
# Model and loss selection
defaults:
  - model: custom_transformer      # or: distilbert
  - loss: bce                        # or: focal, weighted_bce

# Training hyperparameters
training:
  num_epochs: 1
  learning_rate: 2e-5
  warmup_steps: 500
  eval_steps: 30
```

**Training Configuration Examples:**

Train with different model and loss combinations:

**Example 1: Custom Transformer with Focal Loss**
```bash
# Use custom Pre-LN Transformer with Focal Loss 
./runners/run_training.sh model=custom_transformer loss=focal
```

**Example 2: DistilBERT with Weighted BCE**
```bash
# Use pretrained DistilBERT with weighted loss adjustment
./runners/run_training.sh model=distilbert loss=weighted_bce
```

---
## Project Structure

```
├── README.md                          # This file
├── pyproject.toml                     # Package configuration
│
├── config/
│   ├── training.yaml                 # Training configuration
│   ├── model/
│   │   ├── custom_transformer.yaml   # Architecture hyperparameters
│   │   └── distilbert.yaml           # DistilBERT config (optional)
│   └── loss/
│       ├── bce.yaml                  # BCE loss config
│       ├── focal.yaml                # Focal loss config
│       └── weighted_bce.yaml         # Weighted BCE config
│
├── src/
│   ├── models/
│   │   ├── toxicity_classifier.py    # Main model wrapper
│   │   ├── model_factory.py          # Model instantiation
│   │   ├── loss_factory.py           # Loss function factory
│   │   ├── custom_transformer/       # Custom transformer 
│   │   │   ├── preln_transformer.py
│   │   │   └── __init__.py
│   │   ├── losses/                   # Loss implementations
│   │   │   ├── focal.py
│   │   │   └── __init__.py
│   │   └── backbones/                # Model backbones
│   │       ├── bert.py
│   │       ├── custom_transformer.py
│   │       └── __init__.py
│   │
│   ├── training/
│   │   └── training_engine.py        # Training loop (Trainer wrapper)
│   │
│   ├── inference/
│   │   ├── inference_engine.py       # Post-training inference
│   │   └── metrics.py                # Metrics calculation
│   │
│   ├── datamodule.py                 # Data loading pipeline
│   └── analysis/                     # Data analysis scripts
│       ├── data_exploration.py
│       ├── label_consistency.py
│       └── boundary_case_handler.py
│
├── runners/
│   └── run_data_prep.sh              # Data preparation script
│   └── run_training.sh               # Training script
│   └── run_inference.sh              # Inference script
│
├── docs/
│   ├── 1_data_analysis.md            # Data exploration results
│   ├── 4_handling_class_imbalance.md # Class imbalance strategies
│   ├── 5_preln_model.md              # Architecture deep-dive
│   └── 6_loss_design.md              # Loss function comparison
│
└── outputs/                          # Generated during execution
    ├── data_prep/                    # Preprocessed data splits
    ├── training/                     # Training results & checkpoints
    │   └── 2026-02-15/23-13-53/
    │       ├── final_model/
    │       ├── checkpoint-*/
    │       └── inference_results/
    └── data_exploration/             # Data analysis outputs
```
## References

- Civil Comments Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
