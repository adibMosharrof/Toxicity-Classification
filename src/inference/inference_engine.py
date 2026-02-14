"""Inference engine for model predictions and evaluation."""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hydra
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.datamodule import JigsawDataModule
from src.inference.metrics import MetricsCalculator


class InferenceEngine:
    """
    Engine for running inference and computing metrics.
    
    Decoupled from model implementation - accepts any model instance with a
    standard interface (forward(input_ids, attention_mask) -> preds, probs).
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """
        Initialize InferenceEngine.

        Args:
            batch_size: Batch size for inference (default: 32)
            max_length: Maximum sequence length (default: 512)
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.metrics_calculator = MetricsCalculator()

    def init_datamodule(self, tokenizer) -> JigsawDataModule:
        """
        Initialize datamodule.

        Args:
            tokenizer: Tokenizer for encoding text

        Returns:
            Initialized datamodule
        """
        datamodule = JigsawDataModule(
            tokenizer=tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        return datamodule

    @torch.no_grad()
    def run_inference(
        self, dataloader: DataLoader, model
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run model inference on data.
        
        Model should implement forward(input_ids, attention_mask) -> (preds, probs)

        Args:
            dataloader: DataLoader with test data
            model: Model to run inference with (any nn.Module with standard interface)

        Returns:
            Tuple of (predicted_labels, predicted_probs, true_labels)
        """
        all_preds = []
        all_probs = []
        all_labels = []

        model.eval()

        for batch in tqdm(dataloader, desc="Inference"):
            # Move batch to device using accelerate
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"].cpu().numpy()

            # Forward pass - model returns (preds, probs)
            preds, probs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels)

        return (
            np.array(all_preds),
            np.array(all_probs),
            np.array(all_labels),
        )

    def evaluate(
        self,
        test_data_path: str,
        model,
        datamodule: JigsawDataModule,
    ) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
        """
        Run inference and calculate metrics.

        Args:
            test_data_path: Path to test CSV file
            model: Model instance to run inference with
            datamodule: Datamodule for data loading

        Returns:
            Tuple of (metrics_dict, results_df, original_df)
        """
        logger = logging.getLogger(__name__)
        
        # Prepare dataloader using datamodule
        dataloader, df = datamodule.prepare_test_dataloader(test_data_path)

        logger.info("Running inference")
        predictions, probabilities, true_labels = self.run_inference(dataloader, model)

        logger.info("Calculating metrics")
        metrics = self.metrics_calculator.calculate(
            true_labels, predictions, probabilities
        )

        # Create results dataframe with per-row outputs
        results_df = pd.DataFrame({
            "id": df["id"].values,
            "comment_text": df["comment_text"].values,
            "ground_truth_probability": df["target"].values,
            "true_label": true_labels,
            "predicted_label": predictions,
            "predicted_probability": probabilities,
        })

        logger.info(self.metrics_calculator.format_report(metrics))

        return metrics, results_df, df

    def run(
        self,
        test_data_path: str,
        model_name: str,
        backbone_type: str = "bert",
        threshold: float = 0.5,
        output_dir: Path = None,
    ) -> None:
        """
        Execute the full inference pipeline.

        Args:
            test_data_path: Path to test CSV file
            model_name: Model identifier (e.g., "distilbert-base-uncased")
            backbone_type: Type of backbone to use (default: "bert")
            threshold: Binary classification threshold (default: 0.5)
            output_dir: Output directory to save results
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting inference pipeline")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create model using factory
        from src.models import ModelFactory
        model = ModelFactory.create(
            model_name=model_name,
            backbone_type=backbone_type,
            threshold=threshold,
        )
        
        # Initialize datamodule with tokenizer
        datamodule = self.init_datamodule(tokenizer)
        
        # Prepare model with accelerator
        model = self.accelerator.prepare(model)
        model.eval()
        
        # Run evaluation
        metrics, results_df, original_df = self.evaluate(test_data_path, model, datamodule)
        
        # Save results
        self.save_results(output_dir, metrics, results_df)
        
        logger.info("Inference pipeline completed successfully!")

    def save_results(
        self,
        output_dir: Path,
        metrics: Dict[str, float],
        results_df: pd.DataFrame,
    ) -> None:
        """
        Save results to output directory as CSV files.

        Args:
            output_dir: Output directory path
            metrics: Dictionary of metrics
            results_df: DataFrame with per-row predictions and probabilities
        """
        logger = logging.getLogger(__name__)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save per-row results
        results_file = output_dir / "detailed_results.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Saved detailed results to {results_file}")

        # Save summary metrics
        summary_df = pd.DataFrame([metrics])
        summary_file = output_dir / "summary_metrics.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary metrics to {summary_file}")


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@hydra.main(version_base=None, config_path="../../config", config_name="inference")
def main(cfg: DictConfig) -> None:
    """
    Main function to execute inference using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting inference pipeline")
    logger.info(f"Configuration:\n{cfg}")

    # Set random seed
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # Initialize inference engine
    engine = InferenceEngine(
        batch_size=cfg.inference.batch_size,
        max_length=cfg.inference.max_length,
    )

    # Get the Hydra output directory
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Construct full test data path
    test_data_path = str(Path(cfg.project_root) / cfg.data.directory / cfg.data.test_file)
    logger.info(f"Test data path: {test_data_path}")

    # Run inference pipeline - engine handles model and tokenizer loading
    engine.run(
        test_data_path=test_data_path,
        model_name=cfg.model.name,
        backbone_type=cfg.model.backbone_type,
        threshold=cfg.model.threshold,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
