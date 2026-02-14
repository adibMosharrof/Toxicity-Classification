"""Metrics calculation utilities for model evaluation."""

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsCalculator:
    """Calculate evaluation metrics."""

    def __init__(self):
        """Initialize MetricsCalculator."""
        self.logger = logging.getLogger(__name__)

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            y_true: True labels (binary: 0 or 1)
            y_pred: Predicted labels (binary: 0 or 1)
            y_pred_proba: Predicted probabilities for positive class (optional, for AUROC)

        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}

        # Binary classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

        # AUROC (requires probabilities)
        if y_pred_proba is not None:
            metrics["auroc"] = roc_auc_score(y_true, y_pred_proba)
        else:
            self.logger.warning("y_pred_proba not provided, AUROC will be skipped")
            metrics["auroc"] = None

        return metrics

    def format_report(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics as a readable report.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Formatted report string
        """
        report = "\n" + "=" * 50 + "\n"
        report += "EVALUATION METRICS\n"
        report += "=" * 50 + "\n"

        for metric_name, value in metrics.items():
            if value is not None:
                report += f"{metric_name.upper():20s}: {value:.4f}\n"
            else:
                report += f"{metric_name.upper():20s}: N/A\n"

        report += "=" * 50 + "\n"

        return report
