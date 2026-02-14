"""Generic datamodule for loading and preparing Jigsaw classification data."""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datamodule.collators import JigsawCollator
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

    def get_dataloader(self, df: pd.DataFrame, shuffle: bool = False) -> DataLoader:
        """
        Create a DataLoader from dataframe.

        Args:
            df: DataFrame with 'comment_text' and 'label' columns
            shuffle: Whether to shuffle the data (default: False)

        Returns:
            DataLoader
        """
        dataset = JigsawDataset(df, self.tokenizer, self.max_length)
        self.logger.info(f"Created JigsawDataset with {len(dataset)} samples")

        collator = JigsawCollator(self.tokenizer)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            pin_memory=True,
        )

        self.logger.info(f"Created DataLoader with batch size {self.batch_size}")
        return dataloader

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
        dataloader = self.get_dataloader(df, shuffle=False)
        return dataloader, df

    def prepare_train_val_dataloaders(
        self, train_path: str, val_path: str
    ) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
        """
        Load train and validation data and create dataloaders (for training).

        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file

        Returns:
            Tuple of (train_loader, val_loader, train_df, val_df)
        """
        self.logger.info("Loading training data")
        train_df = self.load_data(train_path)
        train_dataloader = self.get_dataloader(train_df, shuffle=True)

        self.logger.info("Loading validation data")
        val_df = self.load_data(val_path)
        val_dataloader = self.get_dataloader(val_df, shuffle=False)

        return train_dataloader, val_dataloader, train_df, val_df
