#!/usr/bin/env python3
"""Exploratory data analysis on PSD world balance sheets.

Saves all computed statistics and analysis results to results/eda/.
Visualization is handled separately by 08_visualize.py.

Usage:
    python scripts/03_eda_analysis.py
"""

import json
import logging
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src.config import EDA_DIR, PROCESSED_DIR, SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

np.random.seed(SEED)


def load_balance_sheets() -> pd.DataFrame:
    """Load processed world balance sheet data."""
    path = PROCESSED_DIR / "psd_world_balance_sheets.parquet"
    df = pd.read_parquet(path)
    logger.info("Loaded %d balance sheet rows", len(df))
    return df


def summary_statistics(df: pd.DataFrame) -> None:
    """Compute and save per-commodity summary statistics."""
    balance_cols = [
        "beginning_stocks", "production", "imports",
        "domestic_consumption", "exports", "ending_stocks",
        "total_supply", "total_distribution",
    ]
    available_cols = [c for c in balance_cols if c in df.columns]

    all_stats = {}
    for commodity, grp in df.groupby("Commodity_Description"):
        commodity_stats = {}
        for col in available_cols:
            series = grp[col].dropna()
            commodity_stats[col] = {
                "count": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "skew": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
            }
        # Year range
        commodity_stats["_meta"] = {
            "year_min": int(grp["Market_Year"].min()),
            "year_max": int(grp["Market_Year"].max()),
            "n_years": int(len(grp)),
        }
        all_stats[commodity] = commodity_stats
        logger.info("  %s: %d years, ending_stocks mean=%.0f",
                     commodity, len(grp), grp["ending_stocks"].mean())

    out = EDA_DIR / "summary_statistics.json"
    with open(out, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info("Saved summary statistics to %s", out)


def compute_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add stocks-to-use ratio and other derived metrics."""
    df = df.copy()
    df["stocks_to_use"] = df["ending_stocks"] / df["domestic_consumption"]
    df["production_consumption_ratio"] = df["production"] / df["domestic_consumption"]
    df["import_share"] = df["imports"] / df["total_supply"]
    df["export_share"] = df["exports"] / df["total_distribution"]

    # Year-over-year changes
    for commodity, grp in df.groupby("Commodity_Code"):
        idx = grp.sort_values("Market_Year").index
        for col in ["ending_stocks", "total_supply", "production"]:
            df.loc[idx, f"{col}_yoy_change"] = grp.loc[idx, col].diff()
            df.loc[idx, f"{col}_yoy_pct"] = grp.loc[idx, col].pct_change() * 100

    return df


def balance_sheet_time_series(df: pd.DataFrame) -> None:
    """Save time series of balance sheet components per commodity for plotting."""
    df_enriched = compute_derived_ratios(df)

    for commodity, grp in df_enriched.groupby("Commodity_Description"):
        grp_sorted = grp.sort_values("Market_Year")
        safe_name = commodity.lower().replace(",", "").replace(" ", "_")
        out = EDA_DIR / f"balance_sheet_ts_{safe_name}.csv"
        grp_sorted.to_csv(out, index=False)
        logger.info("Saved %s time series to %s", commodity, out)

    # Also save the enriched full dataset
    df_enriched.to_csv(EDA_DIR / "balance_sheet_all_enriched.csv", index=False)
    logger.info("Saved enriched balance sheet data")


def stationarity_tests(df: pd.DataFrame) -> None:
    """Run ADF and KPSS tests on target variables."""
    targets = ["ending_stocks", "total_supply"]
    results = {}

    for commodity, grp in df.groupby("Commodity_Description"):
        grp_sorted = grp.sort_values("Market_Year")
        results[commodity] = {}

        for target in targets:
            series = grp_sorted[target].dropna().values
            if len(series) < 20:
                logger.warning("  %s %s: too few observations (%d)",
                               commodity, target, len(series))
                continue

            target_results = {}

            # ADF test on levels
            adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(
                series, autolag="AIC"
            )
            target_results["adf_levels"] = {
                "statistic": float(adf_stat),
                "p_value": float(adf_p),
                "lags": int(adf_lags),
                "nobs": int(adf_nobs),
                "critical_values": {k: float(v) for k, v in adf_crit.items()},
                "stationary": bool(adf_p < 0.05),
            }

            # ADF test on first differences
            diff_series = np.diff(series)
            adf_d_stat, adf_d_p, *_ = adfuller(diff_series, autolag="AIC")
            target_results["adf_first_diff"] = {
                "statistic": float(adf_d_stat),
                "p_value": float(adf_d_p),
                "stationary": bool(adf_d_p < 0.05),
            }

            # KPSS test on levels
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
                series, regression="ct", nlags="auto"
            )
            target_results["kpss_levels"] = {
                "statistic": float(kpss_stat),
                "p_value": float(kpss_p),
                "lags": int(kpss_lags),
                "critical_values": {k: float(v) for k, v in kpss_crit.items()},
                "stationary": bool(kpss_p > 0.05),
            }

            # ACF and PACF (save values for plotting)
            n_lags = min(20, len(series) // 3)
            acf_vals = acf(series, nlags=n_lags, fft=True).tolist()
            pacf_vals = pacf(series, nlags=n_lags).tolist()
            target_results["acf"] = acf_vals
            target_results["pacf"] = pacf_vals
            target_results["n_lags"] = n_lags

            results[commodity][target] = target_results
            logger.info(
                "  %s %s: ADF p=%.4f (%s), KPSS p=%.4f (%s), diff ADF p=%.4f",
                commodity, target,
                adf_p, "stationary" if adf_p < 0.05 else "non-stationary",
                kpss_p, "stationary" if kpss_p > 0.05 else "non-stationary",
                adf_d_p,
            )

    out = EDA_DIR / "stationarity_tests.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved stationarity test results to %s", out)


def cross_commodity_analysis(df: pd.DataFrame) -> None:
    """Compute cross-commodity correlations and save."""
    # Pivot to wide format: one column per commodity's ending stocks
    pivot = df.pivot_table(
        index="Market_Year",
        columns="Commodity_Description",
        values=["ending_stocks", "total_supply", "production"],
    )

    # Correlation matrix for ending stocks
    es_corr = pivot["ending_stocks"].corr()
    ts_corr = pivot["total_supply"].corr()

    results = {
        "ending_stocks_correlation": es_corr.to_dict(),
        "total_supply_correlation": ts_corr.to_dict(),
    }

    # Year-over-year change correlations
    es_changes = pivot["ending_stocks"].diff().dropna()
    ts_changes = pivot["total_supply"].diff().dropna()
    results["ending_stocks_change_correlation"] = es_changes.corr().to_dict()
    results["total_supply_change_correlation"] = ts_changes.corr().to_dict()

    out = EDA_DIR / "cross_commodity_correlations.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved cross-commodity correlations to %s", out)

    # Save the pivoted data for plotting
    pivot.to_csv(EDA_DIR / "cross_commodity_pivot.csv")


def structural_breaks_analysis(df: pd.DataFrame) -> None:
    """Identify notable structural breaks and outliers."""
    results = {}

    for commodity, grp in df.groupby("Commodity_Description"):
        grp_sorted = grp.sort_values("Market_Year")
        commodity_breaks = []

        for col in ["ending_stocks", "total_supply", "production"]:
            series = grp_sorted[col].values
            years = grp_sorted["Market_Year"].values

            # Year-over-year percentage changes
            pct_changes = np.diff(series) / series[:-1] * 100
            change_years = years[1:]

            # Flag years with >15% change as notable
            threshold = 15.0
            notable = np.abs(pct_changes) > threshold
            for i, is_notable in enumerate(notable):
                if is_notable:
                    commodity_breaks.append({
                        "variable": col,
                        "year": int(change_years[i]),
                        "pct_change": float(pct_changes[i]),
                        "value_before": float(series[i]),
                        "value_after": float(series[i + 1]),
                    })

        results[commodity] = sorted(commodity_breaks, key=lambda x: x["year"])
        logger.info("  %s: %d notable breaks (>15%% change)",
                     commodity, len(commodity_breaks))

    out = EDA_DIR / "structural_breaks.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved structural breaks analysis to %s", out)


def main():
    logger.info("=" * 60)
    logger.info("EDA Analysis: PSD World Balance Sheets")
    logger.info("=" * 60)

    df = load_balance_sheets()

    logger.info("\n--- Summary Statistics ---")
    summary_statistics(df)

    logger.info("\n--- Balance Sheet Time Series ---")
    balance_sheet_time_series(df)

    logger.info("\n--- Stationarity Tests ---")
    stationarity_tests(df)

    logger.info("\n--- Cross-Commodity Analysis ---")
    cross_commodity_analysis(df)

    logger.info("\n--- Structural Breaks ---")
    structural_breaks_analysis(df)

    logger.info("\nEDA analysis complete. Results saved to %s", EDA_DIR)


if __name__ == "__main__":
    sys.exit(main() or 0)
