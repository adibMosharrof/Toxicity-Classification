"""Custom data collators."""

from typing import Dict, List

import torch


class JigsawCollator:
    """Custom collator for Jigsaw training datasets."""

    def __init__(self, tokenizer):
        """
        Initialize collator.

        Args:
            tokenizer: Tokenizer for padding token
        """
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of training data.

        Args:
            batch: List of samples from dataset

        Returns:
            Collated batch dict with input_ids, attention_mask, and labels
        """
        # Extract fields
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["label"] for item in batch]

        # Pad sequences
        input_ids_stacked = torch.stack(input_ids)
        attention_mask_stacked = torch.stack(attention_mask)
        labels_stacked = torch.stack(labels)

        result = {
            "input_ids": input_ids_stacked,
            "attention_mask": attention_mask_stacked,
            "labels": labels_stacked,  # HF Trainer expects "labels"
        }

        return result


class InferenceCollator:
    """Custom collator for Jigsaw inference datasets that preserves id field."""

    def __init__(self, tokenizer):
        """
        Initialize collator.

        Args:
            tokenizer: Tokenizer for padding token
        """
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of inference data, preserving sample IDs.

        Args:
            batch: List of samples from dataset

        Returns:
            Collated batch dict with input_ids, attention_mask, and id
        """
        # Extract fields
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        ids = [item["id"] for item in batch]

        # Pad sequences
        input_ids_stacked = torch.stack(input_ids)
        attention_mask_stacked = torch.stack(attention_mask)

        result = {
            "input_ids": input_ids_stacked,
            "attention_mask": attention_mask_stacked,
            "id": ids,
        }

        return result
