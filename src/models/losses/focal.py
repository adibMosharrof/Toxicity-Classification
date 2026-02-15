"""Focal loss implementation for binary classification."""
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class FocalLossWrapper(nn.Module):
    """
    Wrapper around torchvision's sigmoid_focal_loss for binary classification.
    
    Focal loss down-weights easy examples and focuses training on hard examples.
    Useful for handling class imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize FocalLossWrapper.
        
        Args:
            alpha: Weighting factor in [0, 1] to balance positive examples (default: 0.25)
            gamma: Exponent of the modulating factor (1 - p)^gamma (default: 2.0)
            reduction: 'mean', 'sum', or 'none' (default: 'mean')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Raw logits from model, shape (batch_size,) or (batch_size, 1)
            targets: Binary target labels, shape (batch_size,) with values 0 or 1
        
        Returns:
            Scalar loss value (if reduction='mean' or 'sum')
        """
        # Flatten for binary classification
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        
        # Use torchvision's implementation
        loss = sigmoid_focal_loss(
            logits,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )
        
        return loss
