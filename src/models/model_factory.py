"""
Model factory for creating and configuring toxicity classifiers.

Provides a clean interface for instantiating toxicity classifiers with different backbones.
"""
import logging
from pathlib import Path
from typing import Literal, Optional

from src.models.toxicity_classifier import ToxicityClassifier
from src.models.backbones import BertBackbone


logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating toxicity classifier models."""

    # Mapping of backbone types to their classes
    BACKBONES = {
        "bert": BertBackbone,
        # "custom_transformer": CustomTransformerBackbone,  # Add future backbones here
    }

    @staticmethod
    def create(
        model_name: str = "distilbert-base-uncased",
        backbone_type: Literal["bert"] = "bert",
        threshold: float = 0.5,
        path: Optional[str] = None,
        project_root: Optional[str] = None,
    ) -> ToxicityClassifier:
        """
        Create a toxicity classifier model for binary classification.

        Args:
            model_name: Model identifier (e.g., "distilbert-base-uncased", "bert-base-uncased").
            backbone_type: Type of backbone to use (default: "bert").
                          Supported: "bert", "custom_transformer" (future).
            threshold: Probability threshold for binary classification (default: 0.5).
            path: Optional path to local trained model directory (relative to project_root).
                  If provided, loads from this path instead of model_name.
            project_root: Project root directory for resolving relative paths.

        Returns:
            ToxicityClassifier instance with the specified backbone.

        Raises:
            ValueError: If backbone_type is not supported.
            FileNotFoundError: If path is provided but does not exist.

        Examples:
            # Load from HuggingFace (pretrained)
            model = ModelFactory.create(model_name="distilbert-base-uncased")

            # Load from local trained model
            model = ModelFactory.create(
                model_name="distilbert-base-uncased",
                path="outputs/training/2026-02-14/13-34-03/final_model",
                project_root="/u/siddique-d1/adib/jigsaw"
            )
        """
        # Get backbone class from registry
        if backbone_type not in ModelFactory.BACKBONES:
            raise ValueError(
                f"Unsupported backbone_type: {backbone_type}. "
                f"Supported types: {list(ModelFactory.BACKBONES.keys())}"
            )
        
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
