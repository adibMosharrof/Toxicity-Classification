"""Training engine for training toxicity classification models."""

import logging
import sys
from pathlib import Path
from typing import Dict

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.datamodule import JigsawDataModule
from src.datamodule.collators import JigsawCollator
from src.models import ModelFactory
from src.training.hf_callbacks import setup_callbacks

class TrainingEngine:
    """
    Engine for training toxicity classification models.
    
    Orchestrates data loading, model creation, trainer setup, and training.
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_length: int = 512,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 3,
        eval_steps: int = 100,
        save_steps: int = 100,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize TrainingEngine.

        Args:
            batch_size: Training batch size (default: 32)
            max_length: Maximum sequence length (default: 512)
            num_epochs: Number of training epochs (default: 3)
            learning_rate: Learning rate (default: 2e-5)
            warmup_steps: Number of warmup steps (default: 500)
            weight_decay: Weight decay for optimizer (default: 0.01)
            early_stopping_patience: Patience for early stopping (default: 3)
            eval_steps: Evaluation frequency in steps (default: 100)
            save_steps: Save frequency in steps (default: 100)
            gradient_accumulation_steps: Gradient accumulation steps (default: 1)
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.logger = logging.getLogger(__name__)

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

    def create_trainer(
        self,
        model,
        train_dataset,
        eval_dataset,
        tokenizer,
        output_dir: Path,
        random_seed: int = 42,
    ) -> Trainer:
        """
        Create HuggingFace Trainer with training arguments.

        Args:
            model: Model to train
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer for data collation
            output_dir: Output directory for checkpoints
            random_seed: Random seed for reproducibility

        Returns:
            Configured Trainer instance
        """
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            eval_strategy="steps",  
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=5,  
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_steps=10,
            seed=random_seed,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="wandb",
            push_to_hub=False,
            fp16=True,  
        )

        callbacks = setup_callbacks(early_stopping_patience=self.early_stopping_patience)
        data_collator = JigsawCollator(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        return trainer

    def train(
        self,
        train_data_path: str,
        val_data_path: str,
        model_name: str,
        backbone_type: str = "bert",
        threshold: float = 0.5,
        output_dir: Path = None,
        random_seed: int = 42,
    ) -> None:
        """
        Execute the full training pipeline.

        Args:
            train_data_path: Path to training CSV file
            val_data_path: Path to validation CSV file
            model_name: Model identifier (e.g., "distilbert-base-uncased")
            backbone_type: Type of backbone to use (default: "bert")
            threshold: Binary classification threshold (default: 0.5)
            output_dir: Output directory to save results
            random_seed: Random seed for reproducibility
        """
        self.logger.info("Starting training pipeline")

        # Load tokenizer
        self.logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create model using factory
        model = ModelFactory.create(
            model_name=model_name,
            backbone_type=backbone_type,
            threshold=threshold,
        )

        # Initialize datamodule with tokenizer
        datamodule = self.init_datamodule(tokenizer)

        # Prepare train/val datasets from datamodule
        self.logger.info("Preparing training and validation datasets")
        train_dataset, val_dataset = datamodule.prepare_train_val_datasets(
            train_data_path, val_data_path
        )

        # Note: Do NOT use accelerator.prepare() with Trainer - Trainer handles device placement
        self.logger.info("Model ready for training")

        # Create trainer (Trainer handles device placement)
        self.logger.info("Creating trainer")
        trainer = self.create_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            output_dir=output_dir,
            random_seed=random_seed,
        )

        # Train
        self.logger.info("Starting training")
        try:
            trainer.train()
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}", exc_info=True)
            raise

        # Save final model
        self.logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(str(output_dir / "final_model"))

        self.logger.info("Training completed successfully!")


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@hydra.main(version_base=None, config_path="../../config", config_name="training")
def main(cfg: DictConfig) -> None:
    """
    Main function to execute training using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting training pipeline")
    logger.info(f"Configuration:\n{cfg}")

    # Initialize wandb if enabled
    if cfg.wandb.enabled:
        wandb_config = {
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "num_epochs": cfg.training.num_epochs,
            "warmup_steps": cfg.training.warmup_steps,
            "weight_decay": cfg.training.weight_decay,
            "early_stopping_patience": cfg.training.early_stopping_patience,
            "model": cfg.model.name,
            "backbone_type": cfg.model.backbone_type,
        }
        
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            notes=cfg.wandb.notes,
            tags=cfg.wandb.tags,
            config=wandb_config,
        )
        logger.info(f"Wandb initialized with project: {cfg.wandb.project}")

    # Set random seed
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # Initialize training engine
    engine = TrainingEngine(
        batch_size=cfg.training.batch_size,
        max_length=cfg.training.max_length,
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        early_stopping_patience=cfg.training.early_stopping_patience,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
    )

    # Get the Hydra output directory
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Construct full data paths
    train_data_path = str(Path(cfg.project_root) / cfg.data.directory / cfg.data.train_file)
    val_data_path = str(Path(cfg.project_root) / cfg.data.directory / cfg.data.val_file)
    logger.info(f"Train data path: {train_data_path}")
    logger.info(f"Val data path: {val_data_path}")

    # Run training pipeline
    engine.train(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        model_name=cfg.model.name,
        backbone_type=cfg.model.backbone_type,
        threshold=cfg.model.threshold,
        output_dir=output_dir,
        random_seed=cfg.random_seed,
    )

    # Finish wandb run if enabled
    if cfg.wandb.enabled:
        wandb.finish()
        logger.info("Wandb run finished")


if __name__ == "__main__":
    main()
