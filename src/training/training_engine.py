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
from src.inference.inference_engine import InferenceEngine
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
        eval_batch_size: int = 64,
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
            eval_batch_size: Evaluation batch size (default: 64)
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
        self.eval_batch_size = eval_batch_size
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
            per_device_eval_batch_size=self.eval_batch_size,
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
        architecture: dict = None,
        tokenizer_name: str = None,
        loss_config: dict = None,
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
            architecture: Architecture config dict for custom backbones (optional)
        """
        self.logger.info("Starting training pipeline")

        # Load tokenizer - use tokenizer_name if available (for custom models), otherwise use model_name
        actual_tokenizer_name = tokenizer_name if tokenizer_name else model_name
        self.logger.info(f"Loading tokenizer for {actual_tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(actual_tokenizer_name)

        # Create model using factory
        model = ModelFactory.create(
            model_name=model_name,
            backbone_type=backbone_type,
            threshold=threshold,
            architecture=architecture,
            loss_config=loss_config,
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
        final_model_dir = output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model - handle both HuggingFace models and custom models
        if hasattr(model.backbone.model, 'save_pretrained'):
            # HuggingFace model - use save_pretrained
            model.backbone.model.save_pretrained(str(final_model_dir))
        else:
            # Custom model - save state_dict
            import torch
            torch.save(model.backbone.model.state_dict(), str(final_model_dir / "pytorch_model.bin"))
            self.logger.info(f"Saved custom model state_dict to {final_model_dir / 'pytorch_model.bin'}")
        
        tokenizer.save_pretrained(str(final_model_dir))
        self.logger.info(f"Model and tokenizer saved to {final_model_dir}")

        self.logger.info("Training completed successfully!")

    def run_inference(
        self,
        model_checkpoint_path: Path,
        test_data_path: str,
        project_root: str,
        model_name: str,
        backbone_type: str,
        batch_size: int = 32,
        max_length: int = 512,
        threshold: float = 0.5,
        architecture: dict = None,
        tokenizer_name: str = None,
        loss_config: dict = None,
    ) -> None:
        """
        Run inference on test set using trained model checkpoint.

        Args:
            model_checkpoint_path: Path to saved model checkpoint
            test_data_path: Path to test CSV file
            project_root: Project root directory for path resolution
            model_name: Model identifier (e.g., "distilbert-base-uncased")
            backbone_type: Type of backbone to use (e.g., "bert")
            batch_size: Batch size for inference (default: 32)
            max_length: Max sequence length (default: 512)
            threshold: Binary classification threshold (default: 0.5)
        """
        self.logger.info("Running post-training inference")
        
        # Initialize inference engine
        inference_engine = InferenceEngine(
            batch_size=batch_size,
            max_length=max_length,
        )

        # Get output directory for inference results
        inference_output_dir = model_checkpoint_path.parent / "inference_results"
        inference_output_dir.mkdir(parents=True, exist_ok=True)

        # Build relative path from model checkpoint for model loading
        # model_checkpoint_path is like: outputs/training/2026-02-14/14-52-45/final_model
        # We want to pass "outputs/training/2026-02-14/14-52-45/final_model" as the path
        relative_model_path = str(model_checkpoint_path.relative_to(project_root))

        # Run inference
        inference_engine.run(
            test_data_path=test_data_path,
            model_name=model_name,
            backbone_type=backbone_type,
            threshold=threshold,
            output_dir=inference_output_dir,
            path=relative_model_path,
            project_root=project_root,
            architecture=architecture,
            tokenizer_name=tokenizer_name,
            loss_config=loss_config,
        )

        self.logger.info("Post-training inference completed successfully!")


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
            "batch_size": cfg.model.train.batch_size,
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
        batch_size=cfg.model.train.batch_size,
        max_length=cfg.model.max_length,
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        early_stopping_patience=cfg.training.early_stopping_patience,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        gradient_accumulation_steps=cfg.model.train.gradient_accumulation_steps,
        eval_batch_size=cfg.model.train.eval_batch_size,
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
        architecture=dict(cfg.model.architecture) if "architecture" in cfg.model else None,
        tokenizer_name=getattr(cfg.model, 'tokenizer_name', None),
        loss_config=dict(cfg.loss) if "loss" in cfg else {'type': 'bce', 'params': {}},
    )

    # Run post-training inference if enabled
    if cfg.should_run_inference:
        test_data_path = str(Path(cfg.project_root) / cfg.data.directory / "test_split.csv")
        model_checkpoint_path = output_dir / "final_model"
        
        engine.run_inference(
            model_checkpoint_path=model_checkpoint_path,
            test_data_path=test_data_path,
            project_root=cfg.project_root,
            model_name=cfg.model.name,
            backbone_type=cfg.model.backbone_type,
            batch_size=cfg.model.inference.batch_size,
            max_length=cfg.model.max_length,
            threshold=cfg.inference.threshold,
            architecture=dict(cfg.model.architecture) if "architecture" in cfg.model else None,
            tokenizer_name=getattr(cfg.model, 'tokenizer_name', None),
            loss_config=dict(cfg.loss) if "loss" in cfg else {'type': 'bce', 'params': {}},
        )

    # Finish wandb run if enabled
    if cfg.wandb.enabled:
        wandb.finish()
        logger.info("Wandb run finished")


if __name__ == "__main__":
    main()
