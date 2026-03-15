"""Feature engineering for grain supply forecasting.

Builds feature matrices from PSD balance sheets and WASDE revision history
for use with XGBoost and other ML models.

Features include:
- Lagged balance sheet values (t-1, t-2, t-3)
- Derived ratios (stocks-to-use, production/consumption)
- Rolling statistics (3yr, 5yr means and trends)
- Cross-commodity features (other grains' ending stocks)
- WASDE revision summary features (historical revision patterns)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURES_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

# Balance sheet columns available for feature construction
BALANCE_COLS = [
    "beginning_stocks", "production", "imports",
    "domestic_consumption", "exports", "ending_stocks",
    "total_supply", "total_distribution",
]

# Target variables we forecast
TARGETS = ["ending_stocks", "total_supply"]


def _add_lag_features(df: pd.DataFrame, commodity_col: str = "Commodity_Description") -> pd.DataFrame:
    """Add lagged values of key balance sheet components."""
    df = df.copy()
    lag_cols = ["ending_stocks", "production", "domestic_consumption",
                "imports", "exports", "total_supply", "beginning_stocks"]
    available = [c for c in lag_cols if c in df.columns]

    for col in available:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df.groupby(commodity_col)[col].shift(lag)

    return df


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived ratio features."""
    df = df.copy()

    # Stocks-to-use ratio
    if "ending_stocks" in df.columns and "domestic_consumption" in df.columns:
        df["stocks_to_use"] = df["ending_stocks"] / df["domestic_consumption"]
        # Lagged ratios
        df["stocks_to_use_lag1"] = df.groupby("Commodity_Description")["stocks_to_use"].shift(1)

    # Production/consumption ratio
    if "production" in df.columns and "domestic_consumption" in df.columns:
        df["prod_cons_ratio"] = df["production"] / df["domestic_consumption"]
        df["prod_cons_ratio_lag1"] = df.groupby("Commodity_Description")["prod_cons_ratio"].shift(1)

    # Import dependency
    if "imports" in df.columns and "total_supply" in df.columns:
        df["import_share"] = df["imports"] / df["total_supply"]

    # Export share
    if "exports" in df.columns and "total_distribution" in df.columns:
        df["export_share"] = df["exports"] / df["total_distribution"]

    return df


def _add_yoy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year-over-year change features."""
    df = df.copy()
    change_cols = ["ending_stocks", "production", "domestic_consumption", "total_supply"]
    available = [c for c in change_cols if c in df.columns]

    for col in available:
        df[f"{col}_yoy_change"] = df.groupby("Commodity_Description")[col].diff()
        df[f"{col}_yoy_pct"] = df.groupby("Commodity_Description")[col].pct_change() * 100

    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and trend features."""
    df = df.copy()
    roll_cols = ["ending_stocks", "production", "total_supply"]
    available = [c for c in roll_cols if c in df.columns]

    for col in available:
        for window in [3, 5]:
            # Rolling mean (shift by 1 to avoid leakage — use only past data)
            df[f"{col}_roll{window}_mean"] = (
                df.groupby("Commodity_Description")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=window).mean())
            )

        # Linear trend over last 5 years (slope)
        df[f"{col}_trend5"] = (
            df.groupby("Commodity_Description")[col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=5).apply(
                lambda w: np.polyfit(range(len(w)), w, 1)[0] if len(w) == 5 else np.nan,
                raw=True
            ))
        )

    return df


def _add_cross_commodity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add other commodities' lagged ending stocks as features."""
    df = df.copy()

    # Pivot to get each commodity's ending stocks by year
    pivot = df.pivot_table(
        index="Market_Year",
        columns="Commodity_Description",
        values="ending_stocks",
    )

    commodities = pivot.columns.tolist()

    for commodity in commodities:
        safe_name = commodity.lower().replace(",", "").replace(" ", "_")
        for other in commodities:
            if other == commodity:
                continue
            other_safe = other.lower().replace(",", "").replace(" ", "_")
            col_name = f"cross_{other_safe}_es_lag1"
            # Lag by 1 year to avoid leakage
            pivot[col_name + f"_for_{safe_name}"] = pivot[other].shift(1)

    # Melt back and merge
    cross_features = {}
    for commodity in commodities:
        safe_name = commodity.lower().replace(",", "").replace(" ", "_")
        cols = [c for c in pivot.columns if c.endswith(f"_for_{safe_name}")]
        if cols:
            # Rename to remove the _for_ suffix
            renamed = {c: c.replace(f"_for_{safe_name}", "") for c in cols}
            cross_features[commodity] = pivot[cols].rename(columns=renamed)

    for commodity, cross_df in cross_features.items():
        cross_df = cross_df.reset_index()
        mask = df["Commodity_Description"] == commodity
        df = df.merge(
            cross_df,
            on="Market_Year",
            how="left",
            suffixes=("", "_dup"),
        )
        # Remove any duplicates from non-matching commodities
        dup_cols = [c for c in df.columns if c.endswith("_dup")]
        if dup_cols:
            for dc in dup_cols:
                orig = dc.replace("_dup", "")
                # Keep original where commodity matches, use dup otherwise
                df[orig] = df[orig].fillna(df[dc])
            df.drop(columns=dup_cols, inplace=True)

    return df


def _add_revision_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add WASDE revision summary features from historical patterns.

    For each commodity, computes average revision behavior from completed
    marketing years and attaches as features.
    """
    df = df.copy()

    wasde_path = PROCESSED_DIR / "wasde_revision_history.parquet"
    if not wasde_path.exists():
        logger.warning("WASDE revision history not found, skipping revision features")
        return df

    wdf = pd.read_parquet(wasde_path)

    # Map WASDE commodity names to PSD names
    commodity_map = {
        "wheat": "Wheat",
        "corn": "Corn",
        "rice": "Rice, Milled",
    }

    # Extract start year from marketing year (e.g., "2024/25" -> 2024)
    wdf["my_start_year"] = wdf["marketing_year"].str[:4].astype(int)

    for wasde_commodity, psd_commodity in commodity_map.items():
        wc = wdf[wdf["commodity"] == wasde_commodity].copy()
        if wc.empty:
            continue

        # Per marketing year: final revision magnitude, n_revisions, revision std
        my_stats = []
        for my, grp in wc.groupby("marketing_year"):
            start_year = int(my[:4])
            es = grp["ending_stocks"].dropna()
            if len(es) < 2:
                continue

            first_est = es.iloc[0]
            final_est = es.iloc[-1]
            total_revision = final_est - first_est
            revision_pct = (total_revision / first_est * 100) if first_est != 0 else 0

            my_stats.append({
                "my_start_year": start_year,
                "es_total_revision": total_revision,
                "es_revision_pct": revision_pct,
                "es_revision_std": grp["ending_stocks_mom_change"].std(),
                "n_monthly_reports": len(grp),
            })

        if not my_stats:
            continue

        stats_df = pd.DataFrame(my_stats).sort_values("my_start_year")

        # Compute expanding mean of revision stats (lagged to avoid leakage)
        stats_df["avg_revision_pct"] = stats_df["es_revision_pct"].expanding().mean().shift(1)
        stats_df["avg_revision_std"] = stats_df["es_revision_std"].expanding().mean().shift(1)
        stats_df["revision_bias"] = stats_df["es_total_revision"].expanding().mean().shift(1)

        # Merge into PSD data
        merge_cols = ["my_start_year", "avg_revision_pct", "avg_revision_std", "revision_bias"]
        stats_merge = stats_df[merge_cols].rename(columns={"my_start_year": "Market_Year"})

        mask = df["Commodity_Description"] == psd_commodity
        df = df.merge(stats_merge, on="Market_Year", how="left", suffixes=("", f"_{wasde_commodity}"))

    return df


def build_feature_matrix(
    commodities: list[str] | None = None,
    start_year: int | None = None,
) -> pd.DataFrame:
    """Build the full feature matrix for modeling.

    Args:
        commodities: List of commodity descriptions to include.
                     None means all available.
        start_year: Earliest year to include in output (features
                    still computed from full history for lags).

    Returns:
        DataFrame with features and target columns, ready for modeling.
    """
    # Load PSD balance sheets
    psd_path = PROCESSED_DIR / "psd_world_balance_sheets.parquet"
    df = pd.read_parquet(psd_path)
    logger.info("Loaded %d PSD balance sheet rows", len(df))

    if commodities:
        df = df[df["Commodity_Description"].isin(commodities)]
        logger.info("Filtered to %d rows for %s", len(df), commodities)

    df = df.sort_values(["Commodity_Description", "Market_Year"]).reset_index(drop=True)

    # Build features (order matters — some depend on previous steps)
    logger.info("Adding lag features...")
    df = _add_lag_features(df)

    logger.info("Adding ratio features...")
    df = _add_ratio_features(df)

    logger.info("Adding year-over-year features...")
    df = _add_yoy_features(df)

    logger.info("Adding rolling features...")
    df = _add_rolling_features(df)

    logger.info("Adding cross-commodity features...")
    df = _add_cross_commodity_features(df)

    logger.info("Adding revision features...")
    df = _add_revision_features(df)

    # Apply start_year filter after feature computation
    if start_year:
        df = df[df["Market_Year"] >= start_year].reset_index(drop=True)
        logger.info("Filtered to years >= %d: %d rows", start_year, len(df))

    # Identify feature columns (everything that's not metadata or raw targets)
    meta_cols = ["Commodity_Code", "Commodity_Description", "Market_Year"]
    raw_balance = BALANCE_COLS + ["total_distribution"]
    feature_cols = [c for c in df.columns if c not in meta_cols + raw_balance]

    logger.info("Feature matrix: %d rows, %d features", len(df), len(feature_cols))

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excluding metadata and targets)."""
    meta_cols = {"Commodity_Code", "Commodity_Description", "Market_Year"}
    raw_balance = set(BALANCE_COLS) | {"total_distribution"}
    return [c for c in df.columns if c not in meta_cols and c not in raw_balance]


def save_feature_matrix(df: pd.DataFrame, filename: str = "feature_matrix.parquet") -> Path:
    """Save feature matrix to parquet."""
    out = FEATURES_DIR / filename
    df.to_parquet(out, index=False)
    logger.info("Saved feature matrix to %s", out)
    return out
