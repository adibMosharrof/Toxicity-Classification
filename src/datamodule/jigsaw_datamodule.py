"""Generic datamodule for loading and preparing Jigsaw classification data."""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datamodule.collators import JigsawCollator, InferenceCollator
from src.datamodule.datasets import JigsawDataset


class JigsawDataModule:
    """Generic data module for Jigsaw toxicity classification tasks (inference, training, validation)."""

    def __init__(
        self,
        tokenizer,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """
        Initialize JigsawDataModule.

        Args:
            tokenizer: Tokenizer for encoding text
            batch_size: Batch size for DataLoader (default: 32)
            max_length: Maximum sequence length (default: 512)
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            data_path: Path to CSV file

        Returns:
            Loaded dataframe

        Raises:
            FileNotFoundError: If file does not exist
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        self.logger.info(f"Loaded data from {data_path}: {len(df)} rows")

        return df

    def prepare_test_dataloader(self, test_path: str) -> Tuple[DataLoader, pd.DataFrame]:
        """
        Load test data and create dataloader (for inference).

        Args:
            test_path: Path to test CSV file

        Returns:
            Tuple of (dataloader, original_df)
        """
        self.logger.info("Loading test data for inference")
        df = self.load_data(test_path)
        dataset = JigsawDataset(df, self.tokenizer, self.max_length)
        
        collator = InferenceCollator(self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
        )
        
        return dataloader, df

    def prepare_train_val_datasets(
        self, train_path: str, val_path: str
    ) -> Tuple[JigsawDataset, JigsawDataset]:
        """
        Load train and validation data and create datasets (for training with HF Trainer).

        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        self.logger.info("Loading training data")
        train_df = self.load_data(train_path)
        train_dataset = JigsawDataset(train_df, self.tokenizer, self.max_length)
        self.logger.info(f"Created training dataset with {len(train_dataset)} samples")

        self.logger.info("Loading validation data")
        val_df = self.load_data(val_path)
        val_dataset = JigsawDataset(val_df, self.tokenizer, self.max_length)
        self.logger.info(f"Created validation dataset with {len(val_dataset)} samples")

        return train_dataset, val_dataset

        return train_dataloader, val_dataloader, train_df, val_df
