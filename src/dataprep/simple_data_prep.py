"""Simple data preparation module with train/val/test splitting using Hydra configuration."""

import logging
import sys
from pathlib import Path
from typing import Tuple

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


class SimpleDataPrep:
    """Class-based data splitter for creating train, val, and test splits."""

    def __init__(
        self,
        data_file: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        data_limit: int = -1,
    ):
        """
        Initialize SimpleDataPrep.

        Args:
            data_file: Path to the input CSV file
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.15)
            test_ratio: Proportion of data for testing (default: 0.15)
            random_seed: Random seed for reproducibility (default: 42)
            data_limit: Maximum number of rows to use. -1 means use all data (default: -1)
        """
        self.logger = logging.getLogger(__name__)
        self.data_file = Path(data_file)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.data_limit = data_limit

        # Validate ratios sum to 1
        ratio_sum = train_ratio + val_ratio + test_ratio
        if not (0.99 <= ratio_sum <= 1.01):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {ratio_sum}"
            )

        self.logger.info(f"Initialized SimpleDataPrep with seed={random_seed}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file with optional row limit.

        Returns:
            Loaded dataframe

        Raises:
            FileNotFoundError: If data file does not exist
        """
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        self.logger.info(f"Loading data from {self.data_file}")

        if self.data_limit > 0:
            df = pd.read_csv(self.data_file, nrows=self.data_limit)
            self.logger.info(f"Loaded {len(df)} rows (limited to {self.data_limit})")
        else:
            df = pd.read_csv(self.data_file)
            self.logger.info(f"Loaded {len(df)} rows (full dataset)")

        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process data: select relevant columns and create label column.

        Args:
            df: Input dataframe

        Returns:
            Processed dataframe with selected columns and label
        """
        # Select relevant columns
        required_cols = ['id', 'target', 'comment_text']
        df = df[required_cols].copy()

        # Create label column: 1 if target > 0.5, else 0
        df['label'] = (df['target'] > 0.5).astype(int)

        self.logger.info(f"Processed data: {len(df)} rows, columns: {list(df.columns)}")

        return df

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train, val, and test sets.

        Args:
            df: Input dataframe to split

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info(
            f"Splitting data with ratios - train: {self.train_ratio}, "
            f"val: {self.val_ratio}, test: {self.test_ratio}"
        )

        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_ratio,
            random_state=self.random_seed,
        )

        # Second split: train vs val
        # Adjust val_ratio relative to train+val size
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed,
        )

        self.logger.info(
            f"Split sizes - train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
        )

        return train_df, val_df, test_df

    def save_splits(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """
        Save split dataframes to CSV files with step names.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            output_dir: Path object for directory to save the CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_file = output_path / "train_split.csv"
        val_file = output_path / "val_split.csv"
        test_file = output_path / "test_split.csv"

        train_df.to_csv(train_file, index=False)
        self.logger.info(f"Saved train split to {train_file}")

        val_df.to_csv(val_file, index=False)
        self.logger.info(f"Saved val split to {val_file}")

        test_df.to_csv(test_file, index=False)
        self.logger.info(f"Saved test split to {test_file}")

    def run(self, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the full pipeline: load, process, split, and save.

        Args:
            output_dir: Path object for output directory to save the split CSVs

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = self.load_data()
        df = self.process_data(df)
        
        train_df, val_df, test_df = self.split(df)
        self.save_splits(train_df, val_df, test_df, output_dir)

        return train_df, val_df, test_df


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@hydra.main(version_base=None, config_path="../../config", config_name="data_prep")
def main(cfg: DictConfig) -> None:
    """
    Main function to execute data splitting using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting data preparation pipeline")
    logger.info(f"Configuration:\n{cfg}")

    # Initialize SimpleDataPrep
    prep = SimpleDataPrep(
        data_file=cfg.data.train_file,
        train_ratio=cfg.splits.train_ratio,
        val_ratio=cfg.splits.val_ratio,
        test_ratio=cfg.splits.test_ratio,
        random_seed=cfg.random_seed,
        data_limit=cfg.data_limit if cfg.data_limit != -1 else -1,
    )

    # Get the Hydra output directory
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Run the splitting pipeline
    train_df, val_df, test_df = prep.run(output_dir)

    logger.info("Data preparation completed successfully!")
    logger.info(f"Summary:")
    logger.info(f"  Train: {len(train_df)} rows")
    logger.info(f"  Val: {len(val_df)} rows")
    logger.info(f"  Test: {len(test_df)} rows")


if __name__ == "__main__":
    main()
