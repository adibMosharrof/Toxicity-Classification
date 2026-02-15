"""Custom Transformer backbone for toxicity classification."""

import torch.nn as nn
from src.models.custom_transformer import PreLNTransformer


class CustomTransformerBackbone(nn.Module):
    """
    Custom Pre-Layer Normalization Transformer backbone for toxicity classification.
    
    Wraps the PreLNTransformer model to provide a standard interface compatible
    with the ToxicityClassifier wrapper.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 1024,
        num_layers: int = 6,
        max_len: int = 256,
        dropout: float = 0.1,
        classification_hidden_dim: int = 512,
    ):
        """
        Initialize Custom Transformer backbone.
        
        Args:
            vocab_size: Vocabulary size (default: 30000)
            d_model: Embedding and hidden dimension (default: 256)
            num_heads: Number of attention heads (default: 4)
            d_ff: Feed-forward hidden dimension (default: 1024)
            num_layers: Number of transformer layers (default: 6)
            max_len: Maximum sequence length (default: 256)
            dropout: Dropout rate (default: 0.1)
            classification_hidden_dim: Hidden size for classification head (default: 512)
        """
        super().__init__()
        self.model = PreLNTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
            classification_hidden_dim=classification_hidden_dim,
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through Custom Transformer backbone.
        
        Args:
            input_ids: Token IDs from tokenizer, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
        
        Returns:
            ModelOutput: Object with .logits attribute for compatibility with ToxicityClassifier
        """
        logits = self.model(input_ids, attention_mask)
        
        # Create output object with .logits attribute to match HuggingFace interface
        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return ModelOutput(logits)
