#!/usr/bin/env python3
"""Train SARIMA models for grain supply forecasting.

Usage:
    python scripts/05_train_sarima.py --all
    python scripts/05_train_sarima.py --commodities wheat
    python scripts/05_train_sarima.py --all --min-train 25
"""

import argparse
import logging
import sys

import pandas as pd

from src.config import PROCESSED_DIR
from src.models.sarima import run_sarima_forecasts, save_sarima_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train SARIMA models")
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

    # Load data
    df = pd.read_parquet(PROCESSED_DIR / "psd_world_balance_sheets.parquet")

    if not args.all and args.commodities:
        name_map = {"wheat": "Wheat", "corn": "Corn", "rice": "Rice, Milled"}
        commodities = [name_map.get(c, c) for c in args.commodities]
        df = df[df["Commodity_Description"].isin(commodities)]

    logger.info("Training SARIMA on %d rows", len(df))

    results = run_sarima_forecasts(df, targets=args.targets, min_train=args.min_train)
    save_sarima_results(results)

    logger.info("SARIMA training complete: %d model(s)", len(results))


if __name__ == "__main__":
    sys.exit(main() or 0)
