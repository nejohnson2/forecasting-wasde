#!/usr/bin/env python3
"""Download raw data from USDA PSD and WASDE.

Usage:
    python scripts/01_fetch_data.py --commodities wheat corn rice
    python scripts/01_fetch_data.py --commodities wheat --force
    python scripts/01_fetch_data.py --skip-wasde
"""

import argparse
import logging
import sys

from src.data.psd_fetcher import fetch_psd_data
from src.data.wasde_fetcher import fetch_wasde_xmls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch USDA PSD and WASDE data")
    parser.add_argument(
        "--commodities",
        nargs="+",
        default=["wheat", "corn", "rice"],
        help="Commodities to fetch (used for logging only; PSD bulk includes all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cache is valid",
    )
    parser.add_argument(
        "--skip-wasde",
        action="store_true",
        help="Skip WASDE XML download",
    )
    parser.add_argument(
        "--wasde-start-year",
        type=int,
        default=2012,
        help="Earliest year for WASDE downloads (default: 2012)",
    )
    args = parser.parse_args()

    # PSD data
    logger.info("Fetching PSD data for: %s", args.commodities)
    path = fetch_psd_data(force=args.force)
    logger.info("PSD data ready at: %s", path)

    # WASDE data
    if not args.skip_wasde:
        logger.info("Fetching WASDE XML files (from %d)", args.wasde_start_year)
        paths = fetch_wasde_xmls(
            start_year=args.wasde_start_year, force=args.force
        )
        logger.info("WASDE: %d XML files ready", len(paths))
    else:
        logger.info("Skipping WASDE download")


if __name__ == "__main__":
    sys.exit(main() or 0)
