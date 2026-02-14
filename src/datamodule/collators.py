"""Custom data collators."""

from typing import Dict, List

import torch


class InferenceCollator:
    """Custom collator for inference that preserves label and id fields."""

    def __init__(self, tokenizer):
        """
        Initialize collator.

        Args:
            tokenizer: Tokenizer for padding token
        """
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of data.

        Args:
            batch: List of samples from dataset

        Returns:
            Collated batch dict
        """
        # Extract fields
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["label"] for item in batch]
        ids = [item["id"] for item in batch]

        # Pad sequences
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
            "id": ids,
        }
