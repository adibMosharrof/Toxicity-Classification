"""
Binary toxicity classification model.

Simple wrapper around any backbone model (BERT, custom Transformer, etc.)
that standardizes the interface for binary classification.
"""
from collections import namedtuple
import torch
import torch.nn as nn


# Simple output container for training
ModelOutput = namedtuple('ModelOutput', ['loss', 'logits'])


class ToxicityClassifier(nn.Module):
    """
    Binary toxicity classifier wrapping a backbone model.
    
    Args:
        backbone: Any model that returns output with logits (e.g., HuggingFace
                 AutoModelForSequenceClassification or custom model)
        threshold: Probability threshold for binary classification (default: 0.5)
    
    Returns:
        preds: Binary predictions (0 or 1)
        probs: Sigmoid probabilities in [0, 1]
    """
    
    def __init__(self, backbone, threshold: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.threshold = threshold
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for binary toxicity classification.
        
        Args:
            input_ids: Token IDs from tokenizer, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)
            labels: Labels for training, shape (batch_size,) - optional
        
        Returns:
            For training (labels provided): SequenceClassifierOutput with loss
            For inference (labels=None): tuple of (preds, probs)
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract logits - handle both HuggingFace outputs and raw models
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # For training: compute cross-entropy loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return ModelOutput(loss=loss, logits=logits)
        
        # For inference: binary classification with threshold
        # Softmax -> take probability of toxic class (index 1)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = (probs >= self.threshold).int()
        
        return preds, probs
