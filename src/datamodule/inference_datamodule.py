"""Inference datamodule for loading and preparing inference data."""

import logging
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datamodule.base import BaseDataModule
from src.datamodule.collators import InferenceCollator
from src.datamodule.datasets import JigsawDataset


class InferenceDataModule(BaseDataModule):
    """Data module for inference tasks."""

    def __init__(
        self,
        tokenizer,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """
        Initialize InferenceDataModule.

        Args:
            tokenizer: Tokenizer for encoding text
            batch_size: Batch size for DataLoader (default: 32)
            max_length: Maximum sequence length (default: 512)
        """
        super().__init__(batch_size=batch_size, max_length=max_length)
        self.tokenizer = tokenizer

    def create_dataloader(self, df: pd.DataFrame) -> DataLoader:
        """
        Create DataLoader from dataframe.

        Args:
            df: DataFrame with data

        Returns:
            DataLoader
        """
        logger = logging.getLogger(__name__)
        dataset = JigsawDataset(df, self.tokenizer, self.max_length)
        logger.info(f"Created JigsawDataset with {len(dataset)} samples")

        collator = InferenceCollator(self.tokenizer)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
        )

        logger.info(f"Created DataLoader with batch size {self.batch_size}")
        return dataloader

    def prepare_dataloader(self, data_path: str) -> Tuple[DataLoader, pd.DataFrame]:
        """
        Load data and create dataloader.

        Args:
            data_path: Path to CSV file

        Returns:
            Tuple of (dataloader, original_df)
        """
        logger = logging.getLogger(__name__)
        logger.info("Loading test data")
        df = self.load_data(data_path)

        logger.info("Creating dataloader")
        dataloader = self.create_dataloader(df)

        return dataloader, df
