"""Base datamodule for data handling."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd


class BaseDataModule:
    """Base class for data modules."""

    def __init__(
        self,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """
        Initialize BaseDataModule.

        Args:
            batch_size: Batch size for DataLoader (default: 32)
            max_length: Maximum sequence length (default: 512)
        """
        self.batch_size = batch_size
        self.max_length = max_length

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
        logger = logging.getLogger(__name__)
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}: {len(df)} rows")

        return df
