#!/usr/bin/env python3
"""Evaluate and compare all trained models.

Usage:
    python scripts/07_evaluate.py
"""

import logging
import sys

from src.models.evaluate import evaluate_all_models, save_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Model Evaluation and Comparison")
    logger.info("=" * 60)

    metrics_df = evaluate_all_models()

    if metrics_df.empty:
        logger.error("No model predictions found. Train models first.")
        sys.exit(1)

    save_evaluation(metrics_df)

    # Print summary
    logger.info("\n--- Summary ---")
    for target in metrics_df["target"].unique():
        logger.info("\nTarget: %s", target)
        subset = metrics_df[metrics_df["target"] == target]
        for _, row in subset.iterrows():
            logger.info(
                "  %s | %s: RMSE=%.1f  MAE=%.1f  MAPE=%.1f%%  DirAcc=%.1f%%",
                row["model"], row["commodity"],
                row["rmse"], row["mae"], row["mape"], row["direction_accuracy"],
            )

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    sys.exit(main() or 0)
