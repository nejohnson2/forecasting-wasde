#!/usr/bin/env python3
"""Train XGBoost models for grain supply forecasting.

Usage:
    python scripts/06_train_xgboost.py --all
    python scripts/06_train_xgboost.py --commodities wheat
"""

import argparse
import logging
import sys

import pandas as pd

from src.config import FEATURES_DIR
from src.models.xgboost_model import run_xgboost_forecasts, save_xgboost_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="All commodities")
    group.add_argument("--commodities", nargs="+", help="Specific commodities")
    parser.add_argument(
        "--min-train", type=int, default=20,
        help="Minimum training window (years, default 20)",
    )
    parser.add_argument(
        "--targets", nargs="+", default=["ending_stocks", "total_supply"],
        help="Target variables",
    )
    args = parser.parse_args()

    # Load feature matrix
    feature_path = FEATURES_DIR / "feature_matrix.parquet"
    if not feature_path.exists():
        logger.error("Feature matrix not found. Run 04_engineer_features.py first.")
        sys.exit(1)

    df = pd.read_parquet(feature_path)

    if not args.all and args.commodities:
        name_map = {"wheat": "Wheat", "corn": "Corn", "rice": "Rice, Milled"}
        commodities = [name_map.get(c, c) for c in args.commodities]
        df = df[df["Commodity_Description"].isin(commodities)]

    logger.info("Training XGBoost on %d rows, %d columns", len(df), len(df.columns))

    results = run_xgboost_forecasts(df, targets=args.targets, min_train=args.min_train)
    save_xgboost_results(results)

    logger.info("XGBoost training complete: %d model(s)", len(results))


if __name__ == "__main__":
    sys.exit(main() or 0)
