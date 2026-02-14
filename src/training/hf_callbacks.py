"""HuggingFace Trainer callbacks for training configuration."""

from transformers.trainer_callback import EarlyStoppingCallback


def setup_callbacks(early_stopping_patience: int = 3):
    """
    Setup callbacks for HuggingFace Trainer.

    Args:
        early_stopping_patience: Number of evaluations with no improvement after which to stop (default: 3)

    Returns:
        List of callbacks
    """
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=0.001,
        ),
    ]

    return callbacks
