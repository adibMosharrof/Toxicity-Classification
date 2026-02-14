"""
Model factory for creating and configuring toxicity classifiers.

Provides a clean interface for instantiating toxicity classifiers with different backbones.
"""
import logging
from typing import Literal

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
    ) -> ToxicityClassifier:
        """
        Create a toxicity classifier model for binary classification.

        Args:
            model_name: Model identifier (e.g., "distilbert-base-uncased", "bert-base-uncased").
            backbone_type: Type of backbone to use (default: "bert").
                          Supported: "bert", "custom_transformer" (future).
            threshold: Probability threshold for binary classification (default: 0.5).

        Returns:
            ToxicityClassifier instance with the specified backbone.

        Raises:
            ValueError: If backbone_type is not supported.

        Examples:
            # Default with BERT backbone
            model = ModelFactory.create()

            # Pre-trained RoBERTa
            model = ModelFactory.create(
                model_name="s-nlp/roberta_toxicity_classifier",
                threshold=0.5,
            )
        """
        # Get backbone class from registry
        if backbone_type not in ModelFactory.BACKBONES:
            raise ValueError(
                f"Unsupported backbone_type: {backbone_type}. "
                f"Supported types: {list(ModelFactory.BACKBONES.keys())}"
            )
        
        backbone_class = ModelFactory.BACKBONES[backbone_type]
        
        logger.info(f"Creating model with {backbone_type} backbone: {model_name}")
        backbone = backbone_class(model_name)
        model = ToxicityClassifier(backbone, threshold=threshold)
        logger.info(f"Model created successfully with threshold={threshold}")
        return model
