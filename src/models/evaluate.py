"""Model evaluation and comparison framework.

Computes metrics, generates comparison tables, and produces
evaluation outputs for visualization.
"""

import json
import logging

import numpy as np
import pandas as pd

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute forecast evaluation metrics.

    Args:
        actual: Ground truth values.
        predicted: Model predictions.

    Returns:
        Dict with RMSE, MAE, MAPE, direction accuracy, and bias.
    """
    errors = actual - predicted
    abs_errors = np.abs(errors)
    n = len(actual)

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(abs_errors)

    # MAPE (avoid division by zero)
    nonzero_mask = actual != 0
    if nonzero_mask.any():
        mape = np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100
    else:
        mape = np.nan

    # Direction accuracy (did we predict the right direction of change?)
    if n > 1:
        actual_dir = np.diff(actual) > 0
        pred_dir = np.diff(predicted) > 0
        direction_acc = np.mean(actual_dir == pred_dir) * 100
    else:
        direction_acc = np.nan

    # Bias (positive = model overestimates)
    bias = np.mean(errors)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "direction_accuracy": float(direction_acc),
        "bias": float(bias),
        "n_predictions": int(n),
    }


def naive_baseline(actual: np.ndarray) -> np.ndarray:
    """Naive forecast: predict last year's value.

    Returns predictions aligned with actual[1:].
    """
    return actual[:-1]


def evaluate_all_models(
    sarima_path: str | None = None,
    xgboost_path: str | None = None,
) -> pd.DataFrame:
    """Load predictions from all models and compute comparison metrics.

    Returns:
        DataFrame with metrics for each model/commodity/target combination.
    """
    sarima_path = sarima_path or MODELS_DIR / "sarima_predictions.csv"
    xgboost_path = xgboost_path or MODELS_DIR / "xgboost_predictions.csv"

    all_metrics = []

    # Load model predictions
    model_dfs = {}
    for name, path in [("SARIMA", sarima_path), ("XGBoost", xgboost_path)]:
        try:
            model_dfs[name] = pd.read_csv(path)
            logger.info("Loaded %s predictions from %s", name, path)
        except FileNotFoundError:
            logger.warning("  %s predictions not found at %s", name, path)

    if not model_dfs:
        logger.error("No model predictions found")
        return pd.DataFrame()

    # Evaluate each model
    for model_name, pred_df in model_dfs.items():
        for (commodity, target), grp in pred_df.groupby(["commodity", "target"]):
            grp = grp.sort_values("Market_Year")
            valid = grp.dropna(subset=["actual", "predicted"])

            if len(valid) < 3:
                continue

            actual = valid["actual"].values
            predicted = valid["predicted"].values

            metrics = compute_metrics(actual, predicted)
            metrics["model"] = model_name
            metrics["commodity"] = commodity
            metrics["target"] = target
            all_metrics.append(metrics)

    # Add naive baseline
    for model_name, pred_df in model_dfs.items():
        for (commodity, target), grp in pred_df.groupby(["commodity", "target"]):
            grp = grp.sort_values("Market_Year")
            valid = grp.dropna(subset=["actual"])
            if len(valid) < 4:
                continue

            actual = valid["actual"].values
            naive_preds = naive_baseline(actual)

            metrics = compute_metrics(actual[1:], naive_preds)
            metrics["model"] = "Naive"
            metrics["commodity"] = commodity
            metrics["target"] = target

            # Only add once per commodity/target
            key = (commodity, target, "Naive")
            if not any(
                (m["commodity"], m["target"], m["model"]) == key
                for m in all_metrics
            ):
                all_metrics.append(metrics)
            break  # Only need one naive per commodity/target

    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df


def save_evaluation(metrics_df: pd.DataFrame) -> None:
    """Save evaluation metrics to files."""
    # CSV for easy viewing
    metrics_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

    # JSON for programmatic access
    metrics_dict = {}
    for _, row in metrics_df.iterrows():
        key = f"{row['model']}_{row['commodity']}_{row['target']}"
        metrics_dict[key] = {
            "model": row["model"],
            "commodity": row["commodity"],
            "target": row["target"],
            "rmse": row["rmse"],
            "mae": row["mae"],
            "mape": row["mape"],
            "direction_accuracy": row["direction_accuracy"],
            "bias": row["bias"],
            "n_predictions": int(row["n_predictions"]),
        }

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info("Saved evaluation results to %s", MODELS_DIR)

    # Print summary table
    pivot = metrics_df.pivot_table(
        index=["commodity", "target"],
        columns="model",
        values=["rmse", "mae", "mape", "direction_accuracy"],
    )
    logger.info("\n%s", pivot.to_string())
