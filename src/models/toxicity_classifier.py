"""
Binary toxicity classification model.

Simple wrapper around any backbone model (BERT, custom Transformer, etc.)
that standardizes the interface for binary classification.
"""
import torch
import torch.nn as nn


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
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for binary toxicity classification.
        
        Args:
            input_ids: Token IDs from tokenizer, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)
        
        Returns:
            preds: Binary predictions (0 or 1), shape (batch_size,)
            probs: Probabilities for toxic class, shape (batch_size,)
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract logits - handle both HuggingFace outputs and raw models
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Binary classification: 2 outputs [non-toxic, toxic]
        # Softmax -> take probability of toxic class (index 1)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = (probs >= self.threshold).int()
        
        return preds, probs
