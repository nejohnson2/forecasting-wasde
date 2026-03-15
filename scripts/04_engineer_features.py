#!/usr/bin/env python3
"""Build feature matrices for grain supply forecasting models.

Usage:
    python scripts/04_engineer_features.py --all
    python scripts/04_engineer_features.py --commodities wheat
    python scripts/04_engineer_features.py --all --start-year 1980
"""

import argparse
import logging
import sys

from src.features.engineer import build_feature_matrix, save_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build feature matrices")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="All commodities")
    group.add_argument("--commodities", nargs="+", help="Specific commodities")
    parser.add_argument(
        "--start-year", type=int, default=None, help="Earliest year in output"
    )
    args = parser.parse_args()

    commodities = None if args.all else args.commodities
    # Map short names to PSD descriptions if needed
    if commodities:
        name_map = {"wheat": "Wheat", "corn": "Corn", "rice": "Rice, Milled"}
        commodities = [name_map.get(c, c) for c in commodities]

    df = build_feature_matrix(commodities=commodities, start_year=args.start_year)

    path = save_feature_matrix(df)
    logger.info("Feature matrix saved: %d rows x %d cols", len(df), len(df.columns))
    logger.info("Columns: %s", list(df.columns))

    # Report NaN counts for feature columns
    feature_cols = [c for c in df.columns if "lag" in c or "roll" in c or "cross" in c or "revision" in c]
    nan_counts = df[feature_cols].isna().sum()
    non_zero_nans = nan_counts[nan_counts > 0]
    if len(non_zero_nans) > 0:
        logger.info("Features with NaNs (expected for early years):")
        for col, count in non_zero_nans.items():
            logger.info("  %s: %d NaN", col, count)


if __name__ == "__main__":
    sys.exit(main() or 0)
