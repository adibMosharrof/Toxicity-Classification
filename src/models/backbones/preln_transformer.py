"""Pre-Layer Normalization Transformer for toxicity classification."""

import math
import torch
import torch.nn as nn


class PreLNTransformer(nn.Module):
    """
    Custom Pre-Layer Normalization Transformer model for binary toxicity classification.
    
    Architecture:
    - Token embeddings (learnable)
    - Sinusoidal positional encoding (non-trainable)
    - Stack of Pre-LN transformer blocks
    - [CLS] token extraction for classification
    - Classification head
    
    Args:
        vocab_size: Size of vocabulary (default: 30000)
        d_model: Embedding and hidden dimension (default: 256)
        num_heads: Number of attention heads (default: 4)
        d_ff: Feed-forward hidden dimension (default: 1024)
        num_layers: Number of transformer layers (default: 6)
        max_len: Maximum sequence length for positional encoding (default: 256)
        dropout: Dropout rate (default: 0.1)
        classification_hidden_dim: Hidden size for classification head (default: 512)
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
        super().__init__()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Sinusoidal positional encoding (registered as buffer)
        self._create_sinusoidal_pos_encoding(d_model, max_len)
        
        # Dropout after embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Pre-LN Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            norm_first=True,  # Pre-LN
            batch_first=True,
        )
        
        # Stack with final layer norm for stability
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),  # Final norm after all layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classification_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classification_hidden_dim, 2),  # Binary classification
        )
    
    def _create_sinusoidal_pos_encoding(self, d_model: int, max_len: int) -> None:
        """
        Create sinusoidal positional encoding and register as buffer.
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer: (1, max_len, d_model) for broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for toxicity classification.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask (1=attend, 0=mask), shape (batch_size, seq_len)
        
        Returns:
            Logits tensor, shape (batch_size, 2)
        """
        seq_len = input_ids.shape[1]
        
        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids)
        x = x + self.pe[:, :seq_len, :]
        x = self.dropout(x)
        
        # Convert attention mask: TransformerEncoder expects True=mask, False=attend
        # Our mask: 1=attend, 0=mask, so invert it
        src_key_padding_mask = (attention_mask == 0)
        
        # Transformer encoder (applies all layers + final norm)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Extract [CLS] token (first token) for classification
        cls_output = x[:, 0, :]
        
        # Classification head
        logits = self.classifier(cls_output)
        
        return logits
