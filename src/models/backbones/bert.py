"""
BERT-based backbones for toxicity classification.
"""
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class BertBackbone(nn.Module):
    """
    BERT-based backbone for toxicity classification.
    
    Wraps HuggingFace AutoModelForSequenceClassification with binary classification setup.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize BERT backbone for binary toxicity classification.
        
        Args:
            model_name: HuggingFace model identifier (default: distilbert-base-uncased)
        """
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=True,
            trust_remote_code=True,  # Bypass torch.load security check for older torch versions
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT backbone.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask
        
        Returns:
            outputs: HuggingFace SequenceClassifierOutput
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
