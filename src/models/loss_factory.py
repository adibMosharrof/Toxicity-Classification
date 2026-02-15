"""Factory for creating loss functions."""
import logging
import torch
import torch.nn as nn

from src.models.losses.focal import FocalLossWrapper

logger = logging.getLogger(__name__)


class LossFactory:
    """Factory for creating loss functions based on configuration."""
    
    LOSS_TYPES = {
        "bce": "BCEWithLogitsLoss",
        "focal": "FocalLoss",
        "weighted_bce": "WeightedBCEWithLogitsLoss",
    }
    
    @staticmethod
    def create(loss_config: dict) -> nn.Module:
        """
        Create a loss function from configuration.
        
        Args:
            loss_config: Dictionary with 'type' and 'params' keys
                        Example: {'type': 'focal', 'params': {'alpha': 0.25, 'gamma': 2.0}}
        
        Returns:
            nn.Module: Loss function instance
        
        Raises:
            ValueError: If loss_type is not supported
        """
        loss_type = loss_config.get('type', 'bce')
        params = loss_config.get('params', {})
        
        logger.info(f"Creating loss function: {loss_type}")
        
        if loss_type == 'bce':
            # Standard BCEWithLogitsLoss for binary classification
            logger.info("Using BCEWithLogitsLoss (binary cross-entropy)")
            return nn.BCEWithLogitsLoss(reduction=params.get('reduction', 'mean'))
        
        elif loss_type == 'focal':
            # Focal loss for handling class imbalance
            logger.info(f"Using FocalLoss with alpha={params.get('alpha', 0.25)}, gamma={params.get('gamma', 2.0)}")
            return FocalLossWrapper(
                alpha=params.get('alpha', 0.25),
                gamma=params.get('gamma', 2.0),
                reduction=params.get('reduction', 'mean')
            )
        
        elif loss_type == 'weighted_bce':
            # Weighted BCE for class imbalance
            pos_weight = params.get('pos_weight')
            if pos_weight is None:
                raise ValueError("pos_weight required for weighted_bce loss")
            logger.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight}")
            pos_weight_tensor = torch.tensor([pos_weight])
            return nn.BCEWithLogitsLoss(
                pos_weight=pos_weight_tensor,
                reduction=params.get('reduction', 'mean')
            )
        
        else:
            raise ValueError(
                f"Unsupported loss_type: {loss_type}. "
                f"Supported types: {list(LossFactory.LOSS_TYPES.keys())}"
            )
