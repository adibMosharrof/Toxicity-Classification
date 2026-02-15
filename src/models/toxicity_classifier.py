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
        loss_fn: Loss function for training (nn.Module)
        threshold: Probability threshold for binary classification (default: 0.5)
    
    Returns:
        preds: Binary predictions (0 or 1)
        probs: Sigmoid probabilities in [0, 1]
    """
    
    def __init__(self, backbone, loss_fn=None, threshold: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.threshold = threshold
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for binary toxicity classification.
        
        Args:
            input_ids: Token IDs from tokenizer, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)
            labels: Labels for training, shape (batch_size,) - optional (0 or 1)
        
        Returns:
            For training (labels provided): ModelOutput with loss and logits
            For inference (labels=None): tuple of (preds, probs)
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract logits - handle both HuggingFace outputs and raw models
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # For binary classification, extract single logit per sample
        # If model outputs (batch, 2), take the positive class logit (index 1)
        if logits.dim() == 2 and logits.shape[1] == 2:
            logits = logits[:, 1]  # Shape: (batch_size,)
        elif logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)  # Shape: (batch_size,)
        else:
            logits = logits.view(-1)  # Ensure shape is (batch_size,)
        
        # For training: compute loss
        if labels is not None:
            if self.loss_fn is None:
                raise RuntimeError(
                    "loss_fn is required for training. "
                    "Provide loss_config when creating the model via ModelFactory.create()"
                )
            loss = self.loss_fn(logits, labels.float())
            return ModelOutput(loss=loss, logits=logits)
        
        # For inference: sigmoid -> probability -> binary prediction
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).int()
        
        return preds, probs
