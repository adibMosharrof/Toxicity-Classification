"""PyTorch datasets for Jigsaw project."""

import torch
from torch.utils.data import Dataset
import pandas as pd


class JigsawDataset(Dataset):
    """PyTorch Dataset for Jigsaw toxicity classification."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        """
        Initialize JigsawDataset.

        Args:
            df: DataFrame with 'comment_text' and 'label' columns
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length (default: 512)
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dict with input_ids, attention_mask, label, and id
        """
        row = self.df.iloc[idx]

        # Tokenize text
        encoded = self.tokenizer(
            row["comment_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "id": row["id"],
        }
