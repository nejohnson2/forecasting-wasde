#!/usr/bin/env python3
"""Process raw PSD and WASDE data into analysis-ready datasets.

Usage:
    python scripts/02_process_data.py --all
    python scripts/02_process_data.py --commodities wheat
    python scripts/02_process_data.py --commodities wheat --start-year 2005
    python scripts/02_process_data.py --all --skip-wasde
"""

import argparse
import logging
import sys

from src.data.psd_processor import process_psd
from src.data.wasde_processor import process_wasde_xmls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process PSD data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Process all commodities")
    group.add_argument(
        "--commodities", nargs="+", help="Specific commodities to process"
    )
    parser.add_argument(
        "--start-year", type=int, default=None, help="Earliest marketing year"
    )
    parser.add_argument(
        "--skip-wasde", action="store_true", help="Skip WASDE processing"
    )
    args = parser.parse_args()

    # Process PSD
    commodities = None if args.all else args.commodities
    df = process_psd(commodities=commodities, start_year=args.start_year)
    logger.info("Processed %d PSD balance sheet rows", len(df))
    logger.info("Columns: %s", list(df.columns))
    logger.info("Year range: %d - %d", df["Market_Year"].min(), df["Market_Year"].max())
    for desc, grp in df.groupby("Commodity_Description"):
        logger.info(
            "  %s: %d years (%d - %d)",
            desc, len(grp), grp["Market_Year"].min(), grp["Market_Year"].max(),
        )

    # Process WASDE
    if not args.skip_wasde:
        logger.info("\n--- Processing WASDE revision history ---")
        wdf = process_wasde_xmls()
        logger.info("WASDE revision history: %d records", len(wdf))
        for commodity, grp in wdf.groupby("commodity"):
            n_my = grp["marketing_year"].nunique()
            logger.info("  %s: %d marketing years, %d monthly observations",
                        commodity, n_my, len(grp))
    else:
        logger.info("Skipping WASDE processing")


if __name__ == "__main__":
    sys.exit(main() or 0)
