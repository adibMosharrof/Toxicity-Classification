"""
Model factory for creating and configuring toxicity classifiers.

Provides a clean interface for instantiating toxicity classifiers with different backbones.
"""
import logging
from pathlib import Path
from typing import Literal, Optional, Any, Dict

from src.models.toxicity_classifier import ToxicityClassifier
from src.models.backbones import BertBackbone
from src.models.backbones.custom_transformer import CustomTransformerBackbone


logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating toxicity classifier models."""

    # Mapping of backbone types to their classes
    BACKBONES = {
        "bert": BertBackbone,
        "custom_transformer": CustomTransformerBackbone,
    }

    @staticmethod
    def create(
        model_name: str = "distilbert-base-uncased",
        backbone_type: Literal["bert", "custom_transformer"] = "bert",
        threshold: float = 0.5,
        path: Optional[str] = None,
        project_root: Optional[str] = None,
        architecture: Optional[Dict[str, Any]] = None,
    ) -> ToxicityClassifier:
        """
        Create a toxicity classifier model for binary classification.

        Args:
            model_name: Model identifier (e.g., "distilbert-base-uncased", "bert-base-uncased").
            backbone_type: Type of backbone to use (default: "bert").
                          Supported: "bert", "custom_transformer".
            threshold: Probability threshold for binary classification (default: 0.5).
            path: Optional path to local trained model directory (relative to project_root).
                  If provided, loads from this path instead of model_name.
            project_root: Project root directory for resolving relative paths.
            architecture: Architecture config dict for custom_transformer. Required if backbone_type="custom_transformer".
                         Contains: vocab_size, d_model, num_heads, d_ff, num_layers, dropout, classification_hidden_dim, max_len

        Returns:
            ToxicityClassifier instance with the specified backbone.

        Raises:
            ValueError: If backbone_type is not supported or config is invalid.
            FileNotFoundError: If path is provided but does not exist.

        Examples:
            # Load from HuggingFace (pretrained)
            model = ModelFactory.create(model_name="distilbert-base-uncased")

            # Load custom transformer
            model = ModelFactory.create(
                backbone_type="custom_transformer",
                threshold=0.5,
                architecture={"vocab_size": 30000, "d_model": 256, ...}
            )
        """
        # Get backbone class from registry
        if backbone_type not in ModelFactory.BACKBONES:
            raise ValueError(
                f"Unsupported backbone_type: {backbone_type}. "
                f"Supported types: {list(ModelFactory.BACKBONES.keys())}"
            )
        
        # For custom_transformer, use architecture config directly
        if backbone_type == "custom_transformer":
            if not architecture:
                raise ValueError(
                    "architecture config is required for custom_transformer backbone"
                )
            logger.info(f"Creating model with custom_transformer backbone")
            backbone = CustomTransformerBackbone(**architecture)
            
            # If path is provided, load the trained model weights
            if path:
                model_path = Path(project_root) / path if project_root else Path(path)
                checkpoint_path = model_path / "pytorch_model.bin"
                if checkpoint_path.exists():
                    logger.info(f"Loading custom transformer weights from: {checkpoint_path}")
                    import torch
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    backbone.model.load_state_dict(state_dict)
                    logger.info("Custom transformer weights loaded successfully")
                else:
                    logger.warning(f"Checkpoint not found at {checkpoint_path}, using random initialization")
            
            model = ToxicityClassifier(backbone, threshold=threshold)
            logger.info(f"Model created successfully with threshold={threshold}")
            return model
        
        # For other backbones (bert, etc.), use model_identifier
        # Determine which model identifier to use
        if path:
            # path is relative to project_root
            model_path = Path(project_root) / path if project_root else Path(path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            model_identifier = str(model_path)
            logger.info(f"Loading model from local path: {model_identifier}")
        else:
            model_identifier = model_name
            logger.info(f"Loading model from HuggingFace: {model_name}")
        
        backbone_class = ModelFactory.BACKBONES[backbone_type]
        logger.info(f"Creating model with {backbone_type} backbone")
        backbone = backbone_class(model_identifier)
        model = ToxicityClassifier(backbone, threshold=threshold)
        logger.info(f"Model created successfully with threshold={threshold}")
        return model
