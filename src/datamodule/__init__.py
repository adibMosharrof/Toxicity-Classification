"""Data modules for different tasks."""

from src.datamodule.base import BaseDataModule
from src.datamodule.inference_datamodule import InferenceDataModule

__all__ = ["BaseDataModule", "InferenceDataModule"]
