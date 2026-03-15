#!/usr/bin/env python3
"""Generate all visualizations from saved analysis results.

Reads from results/eda/ and results/models/, generates plots to results/figures/.
Completely separated from analysis logic per project standards.

Usage:
    python scripts/08_visualize.py
    python scripts/08_visualize.py --section eda
"""

import argparse
import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import EDA_DIR, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")
DPI = 300
COMMODITY_COLORS = {
    "Wheat": "#E69F00",
    "Corn": "#009E73",
    "Rice, Milled": "#0072B2",
}
COMMODITY_ORDER = ["Wheat", "Corn", "Rice, Milled"]


def _save_fig(fig, name: str) -> None:
    """Save figure and close."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def _load_commodity_ts(commodity_name: str) -> pd.DataFrame:
    """Load a commodity's enriched time series CSV."""
    safe = commodity_name.lower().replace(",", "").replace(" ", "_")
    path = EDA_DIR / f"balance_sheet_ts_{safe}.csv"
    return pd.read_csv(path)


# ============================================================
# EDA Plots
# ============================================================

def plot_balance_sheet_overview() -> None:
    """Stacked area charts of supply and demand components per commodity."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=False)
    fig.suptitle("Global Grain Balance Sheets (1960–2025)", fontsize=16, y=1.02)

    for i, commodity in enumerate(COMMODITY_ORDER):
        df = _load_commodity_ts(commodity)
        df = df.sort_values("Market_Year")
        years = df["Market_Year"]

        color = COMMODITY_COLORS[commodity]

        # Supply side (left column)
        ax_s = axes[i, 0]
        supply_cols = ["beginning_stocks", "production", "imports"]
        available = [c for c in supply_cols if c in df.columns]
        ax_s.stackplot(
            years,
            *[df[c].values for c in available],
            labels=[c.replace("_", " ").title() for c in available],
            alpha=0.7,
        )
        ax_s.set_title(f"{commodity} — Supply", fontsize=12)
        ax_s.set_ylabel("1000 MT")
        ax_s.legend(loc="upper left", fontsize=8)

        # Demand side (right column)
        ax_d = axes[i, 1]
        demand_cols = ["domestic_consumption", "exports", "ending_stocks"]
        available = [c for c in demand_cols if c in df.columns]
        ax_d.stackplot(
            years,
            *[df[c].values for c in available],
            labels=[c.replace("_", " ").title() for c in available],
            alpha=0.7,
        )
        ax_d.set_title(f"{commodity} — Distribution", fontsize=12)
        ax_d.set_ylabel("1000 MT")
        ax_d.legend(loc="upper left", fontsize=8)

    axes[-1, 0].set_xlabel("Marketing Year")
    axes[-1, 1].set_xlabel("Marketing Year")
    fig.tight_layout()
    _save_fig(fig, "balance_sheet_overview")


def plot_ending_stocks_comparison() -> None:
    """Line chart comparing ending stocks across all three grains."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for commodity in COMMODITY_ORDER:
        df = _load_commodity_ts(commodity)
        df = df.sort_values("Market_Year")
        ax.plot(
            df["Market_Year"], df["ending_stocks"],
            label=commodity,
            color=COMMODITY_COLORS[commodity],
            linewidth=2,
        )

    ax.set_title("Global Ending Stocks by Commodity (1960–2025)", fontsize=14)
    ax.set_xlabel("Marketing Year")
    ax.set_ylabel("Ending Stocks (1000 MT)")
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "ending_stocks_comparison")


def plot_total_supply_comparison() -> None:
    """Line chart comparing total supply across all three grains."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for commodity in COMMODITY_ORDER:
        df = _load_commodity_ts(commodity)
        df = df.sort_values("Market_Year")
        ax.plot(
            df["Market_Year"], df["total_supply"],
            label=commodity,
            color=COMMODITY_COLORS[commodity],
            linewidth=2,
        )

    ax.set_title("Global Total Supply by Commodity (1960–2025)", fontsize=14)
    ax.set_xlabel("Marketing Year")
    ax.set_ylabel("Total Supply (1000 MT)")
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "total_supply_comparison")


def plot_stocks_to_use() -> None:
    """Stocks-to-use ratio over time per commodity."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for commodity in COMMODITY_ORDER:
        df = _load_commodity_ts(commodity)
        df = df.sort_values("Market_Year")
        ax.plot(
            df["Market_Year"], df["stocks_to_use"] * 100,
            label=commodity,
            color=COMMODITY_COLORS[commodity],
            linewidth=2,
        )

    ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="20% threshold")
    ax.set_title("Global Stocks-to-Use Ratio (1960–2025)", fontsize=14)
    ax.set_xlabel("Marketing Year")
    ax.set_ylabel("Stocks-to-Use (%)")
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "stocks_to_use_ratio")


def plot_yoy_changes() -> None:
    """Year-over-year percentage changes in ending stocks and total supply."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    targets = [
        ("ending_stocks_yoy_pct", "Ending Stocks"),
        ("total_supply_yoy_pct", "Total Supply"),
    ]

    for ax, (col, label) in zip(axes, targets):
        for commodity in COMMODITY_ORDER:
            df = _load_commodity_ts(commodity)
            df = df.sort_values("Market_Year")
            if col in df.columns:
                ax.bar(
                    df["Market_Year"] + COMMODITY_ORDER.index(commodity) * 0.25 - 0.25,
                    df[col],
                    width=0.25,
                    label=commodity,
                    color=COMMODITY_COLORS[commodity],
                    alpha=0.8,
                )
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(f"Year-over-Year Change in {label} (%)", fontsize=12)
        ax.set_ylabel("Change (%)")
        ax.legend(fontsize=10)

    axes[-1].set_xlabel("Marketing Year")
    fig.tight_layout()
    _save_fig(fig, "yoy_changes")


def plot_supply_demand_balance() -> None:
    """Production vs consumption with ending stocks as bar underneath."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, commodity in zip(axes, COMMODITY_ORDER):
        df = _load_commodity_ts(commodity)
        df = df.sort_values("Market_Year")
        years = df["Market_Year"]

        ax.plot(years, df["production"], label="Production",
                color="#2196F3", linewidth=2)
        ax.plot(years, df["domestic_consumption"], label="Consumption",
                color="#F44336", linewidth=2)
        ax.fill_between(
            years, df["production"], df["domestic_consumption"],
            where=df["production"] >= df["domestic_consumption"],
            alpha=0.15, color="green", label="Surplus",
        )
        ax.fill_between(
            years, df["production"], df["domestic_consumption"],
            where=df["production"] < df["domestic_consumption"],
            alpha=0.15, color="red", label="Deficit",
        )
        ax.set_title(f"{commodity}", fontsize=13)
        ax.set_xlabel("Marketing Year")
        ax.set_ylabel("1000 MT")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Production vs. Consumption (Surplus/Deficit)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "production_vs_consumption")


def plot_acf_pacf() -> None:
    """ACF and PACF diagnostic plots for target variables."""
    stat_path = EDA_DIR / "stationarity_tests.json"
    if not stat_path.exists():
        logger.warning("Stationarity tests not found, skipping ACF/PACF plots")
        return

    with open(stat_path) as f:
        stat_results = json.load(f)

    targets = ["ending_stocks", "total_supply"]
    commodities = [c for c in COMMODITY_ORDER if c in stat_results]
    n_commodities = len(commodities)

    fig, axes = plt.subplots(n_commodities, 4, figsize=(20, 4 * n_commodities))
    if n_commodities == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("ACF & PACF Diagnostics", fontsize=14, y=1.02)

    for i, commodity in enumerate(commodities):
        for j, target in enumerate(targets):
            if target not in stat_results[commodity]:
                continue
            data = stat_results[commodity][target]
            acf_vals = data["acf"]
            pacf_vals = data["pacf"]
            n_lags = data["n_lags"]
            lags = list(range(n_lags + 1))

            # ACF
            ax_acf = axes[i, j * 2]
            ax_acf.bar(lags, acf_vals, color=COMMODITY_COLORS.get(commodity, "gray"),
                       alpha=0.7, width=0.6)
            ci = 1.96 / np.sqrt(data.get("n_lags", 20) * 3)
            ax_acf.axhline(y=ci, color="red", linestyle="--", alpha=0.5)
            ax_acf.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
            ax_acf.axhline(y=0, color="black", linewidth=0.5)
            ax_acf.set_title(f"{commodity}\n{target.replace('_', ' ').title()} ACF",
                             fontsize=10)
            ax_acf.set_xlabel("Lag")

            # PACF
            ax_pacf = axes[i, j * 2 + 1]
            ax_pacf.bar(lags, pacf_vals, color=COMMODITY_COLORS.get(commodity, "gray"),
                        alpha=0.7, width=0.6)
            ax_pacf.axhline(y=ci, color="red", linestyle="--", alpha=0.5)
            ax_pacf.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
            ax_pacf.axhline(y=0, color="black", linewidth=0.5)
            ax_pacf.set_title(f"{commodity}\n{target.replace('_', ' ').title()} PACF",
                              fontsize=10)
            ax_pacf.set_xlabel("Lag")

    fig.tight_layout()
    _save_fig(fig, "acf_pacf_diagnostics")


def plot_cross_commodity_correlations() -> None:
    """Heatmaps of cross-commodity correlations."""
    corr_path = EDA_DIR / "cross_commodity_correlations.json"
    if not corr_path.exists():
        logger.warning("Correlation data not found, skipping")
        return

    with open(corr_path) as f:
        corr_data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    titles = [
        ("ending_stocks_correlation", "Ending Stocks (Levels)"),
        ("total_supply_correlation", "Total Supply (Levels)"),
        ("ending_stocks_change_correlation", "Ending Stocks (YoY Changes)"),
        ("total_supply_change_correlation", "Total Supply (YoY Changes)"),
    ]

    for ax, (key, title) in zip(axes.flat, titles):
        if key not in corr_data:
            continue
        corr_df = pd.DataFrame(corr_data[key])
        sns.heatmap(
            corr_df, annot=True, fmt=".2f", cmap="RdYlBu_r",
            vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5,
        )
        ax.set_title(title, fontsize=11)

    fig.suptitle("Cross-Commodity Correlations", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "cross_commodity_correlations")


def plot_structural_breaks() -> None:
    """Timeline of structural breaks (>15% YoY changes)."""
    breaks_path = EDA_DIR / "structural_breaks.json"
    if not breaks_path.exists():
        logger.warning("Structural breaks data not found, skipping")
        return

    with open(breaks_path) as f:
        breaks_data = json.load(f)

    fig, ax = plt.subplots(figsize=(16, 6))

    y_positions = {"Wheat": 2, "Corn": 1, "Rice, Milled": 0}
    markers = {
        "ending_stocks": "o",
        "total_supply": "s",
        "production": "^",
    }

    for commodity, events in breaks_data.items():
        y = y_positions.get(commodity, 0)
        color = COMMODITY_COLORS.get(commodity, "gray")

        for event in events:
            marker = markers.get(event["variable"], "o")
            size = min(abs(event["pct_change"]) * 5, 200)
            face_color = "green" if event["pct_change"] > 0 else "red"
            ax.scatter(
                event["year"], y, s=size, marker=marker,
                facecolors=face_color, edgecolors=color,
                alpha=0.6, linewidths=1.5,
            )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel("Marketing Year")
    ax.set_title("Structural Breaks (>15% YoY Change)\nSize = magnitude, Green = increase, Red = decrease",
                 fontsize=13)

    # Legend for variable markers
    for var, marker in markers.items():
        ax.scatter([], [], marker=marker, c="gray", s=60,
                   label=var.replace("_", " ").title())
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "structural_breaks_timeline")


def plot_component_shares() -> None:
    """How supply and demand component shares have evolved over time."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, commodity in zip(axes, COMMODITY_ORDER):
        df = _load_commodity_ts(commodity)
        df = df.sort_values("Market_Year")

        # Demand-side shares
        total = df["total_distribution"]
        shares = pd.DataFrame({
            "Consumption": df["domestic_consumption"] / total * 100,
            "Exports": df["exports"] / total * 100,
            "Ending Stocks": df["ending_stocks"] / total * 100,
        })

        ax.stackplot(
            df["Market_Year"],
            shares["Consumption"], shares["Exports"], shares["Ending Stocks"],
            labels=shares.columns,
            alpha=0.8,
        )
        ax.set_title(f"{commodity}", fontsize=12)
        ax.set_xlabel("Marketing Year")
        ax.set_ylabel("Share of Distribution (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Distribution Component Shares Over Time", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "component_shares")


def run_eda_plots() -> None:
    """Generate all EDA-related plots."""
    logger.info("--- Generating EDA plots ---")
    plot_balance_sheet_overview()
    plot_ending_stocks_comparison()
    plot_total_supply_comparison()
    plot_stocks_to_use()
    plot_yoy_changes()
    plot_supply_demand_balance()
    plot_acf_pacf()
    plot_cross_commodity_correlations()
    plot_structural_breaks()
    plot_component_shares()


# ============================================================
# WASDE Revision Plots
# ============================================================

def _load_revision_history() -> pd.DataFrame | None:
    """Load WASDE revision history parquet."""
    path = PROCESSED_DIR / "wasde_revision_history.parquet"
    if not path.exists():
        logger.warning("Revision history not found at %s", path)
        return None
    return pd.read_parquet(path)


def plot_revision_funnels() -> None:
    """Show how ending stocks estimates evolve over time for each marketing year.

    Each line represents one marketing year, showing how the estimate
    changes from first forecast to final value.
    """
    df = _load_revision_history()
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    commodity_names = {"wheat": "Wheat", "corn": "Corn", "rice": "Rice, Milled"}

    for ax, (commodity_key, commodity_label) in zip(axes, commodity_names.items()):
        cdf = df[df["commodity"] == commodity_key].copy()
        if cdf.empty:
            continue

        marketing_years = sorted(cdf["marketing_year"].unique())
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(marketing_years)))

        for i, my in enumerate(marketing_years):
            my_data = cdf[cdf["marketing_year"] == my].sort_values("report_date")
            # Create a sequence number (months since first estimate)
            my_data = my_data.reset_index(drop=True)
            ax.plot(
                range(len(my_data)), my_data["ending_stocks"],
                color=cmap[i], alpha=0.7, linewidth=1.5,
                label=my if i % 3 == 0 else None,  # Label every 3rd
            )

        ax.set_title(f"{commodity_label} — Ending Stocks Revisions", fontsize=12)
        ax.set_xlabel("Months Since First Estimate")
        ax.set_ylabel("Ending Stocks (MMT)")
        ax.legend(fontsize=7, ncol=2, title="MY", title_fontsize=8)

    fig.suptitle("How USDA Ending Stocks Estimates Evolve Over Time", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "revision_funnels_ending_stocks")


def plot_revision_magnitude() -> None:
    """Distribution of month-over-month revision sizes."""
    df = _load_revision_history()
    if df is None:
        return

    if "ending_stocks_mom_change" not in df.columns:
        logger.warning("No revision metrics found, skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    commodity_names = {"wheat": "Wheat", "corn": "Corn", "rice": "Rice, Milled"}

    for ax, (key, label) in zip(axes, commodity_names.items()):
        cdf = df[df["commodity"] == key]
        changes = cdf["ending_stocks_mom_change"].dropna()

        if changes.empty:
            continue

        ax.hist(changes, bins=30, color=COMMODITY_COLORS[label],
                alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="black", linewidth=1, linestyle="--")
        ax.axvline(x=changes.mean(), color="red", linewidth=1.5,
                   linestyle="-", label=f"Mean: {changes.mean():.2f}")
        ax.set_title(f"{label}", fontsize=12)
        ax.set_xlabel("Month-over-Month Change (MMT)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    fig.suptitle("Distribution of Monthly Ending Stocks Revisions", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "revision_magnitude_distribution")


def plot_revision_by_forecast_month() -> None:
    """Average absolute revision by position in the forecast cycle.

    Shows whether early or late estimates tend to have larger revisions.
    """
    df = _load_revision_history()
    if df is None:
        return

    if "ending_stocks_mom_change" not in df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    commodity_names = {"wheat": "Wheat", "corn": "Corn", "rice": "Rice, Milled"}

    for ax, (key, label) in zip(axes, commodity_names.items()):
        cdf = df[df["commodity"] == key].copy()
        if cdf.empty:
            continue

        # Compute forecast month number within each marketing year
        cdf["forecast_month"] = cdf.groupby("marketing_year").cumcount()

        # Average absolute revision by forecast month
        avg_rev = (
            cdf.groupby("forecast_month")["ending_stocks_mom_change"]
            .agg(["mean", "std", lambda x: x.abs().mean()])
        )
        avg_rev.columns = ["mean_revision", "std_revision", "mean_abs_revision"]

        ax.bar(avg_rev.index, avg_rev["mean_abs_revision"],
               color=COMMODITY_COLORS[label], alpha=0.7)
        ax.errorbar(avg_rev.index, avg_rev["mean_revision"],
                    yerr=avg_rev["std_revision"], fmt="o-",
                    color="black", markersize=4, capsize=3, label="Mean +/- SD")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.set_title(f"{label}", fontsize=12)
        ax.set_xlabel("Forecast Month (0 = First Estimate)")
        ax.set_ylabel("Revision (MMT)")
        ax.legend(fontsize=8)

    fig.suptitle("Average Revision Magnitude by Forecast Position",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "revision_by_forecast_month")


def plot_cumulative_revision_paths() -> None:
    """Cumulative revision from first estimate, showing convergence patterns."""
    df = _load_revision_history()
    if df is None:
        return

    if "ending_stocks_cum_revision" not in df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    commodity_names = {"wheat": "Wheat", "corn": "Corn", "rice": "Rice, Milled"}

    for ax, (key, label) in zip(axes, commodity_names.items()):
        cdf = df[df["commodity"] == key].copy()
        if cdf.empty:
            continue

        cdf["forecast_month"] = cdf.groupby("marketing_year").cumcount()

        marketing_years = sorted(cdf["marketing_year"].unique())
        cmap = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(marketing_years)))

        for i, my in enumerate(marketing_years):
            my_data = cdf[cdf["marketing_year"] == my]
            ax.plot(
                my_data["forecast_month"],
                my_data["ending_stocks_cum_revision"],
                color=cmap[i], alpha=0.6, linewidth=1.5,
                label=my if i % 3 == 0 else None,
            )

        ax.axhline(y=0, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{label}", fontsize=12)
        ax.set_xlabel("Forecast Month (0 = First Estimate)")
        ax.set_ylabel("Cumulative Revision from First Estimate (MMT)")
        ax.legend(fontsize=7, ncol=2, title="MY", title_fontsize=8)

    fig.suptitle("Cumulative Revision Paths — Ending Stocks",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "cumulative_revision_paths")


def plot_revision_bias() -> None:
    """Check for systematic bias: does USDA tend to over/under-estimate?"""
    df = _load_revision_history()
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    commodity_names = {"wheat": "Wheat", "corn": "Corn", "rice": "Rice, Milled"}

    bar_width = 0.25
    x_offset = {"wheat": -bar_width, "corn": 0, "rice": bar_width}

    for key, label in commodity_names.items():
        cdf = df[df["commodity"] == key].copy()
        if cdf.empty:
            continue

        # For each marketing year, compute total revision (last - first)
        total_rev = cdf.groupby("marketing_year").agg(
            first_est=("ending_stocks", "first"),
            last_est=("ending_stocks", "last"),
        )
        total_rev["revision"] = total_rev["last_est"] - total_rev["first_est"]

        x = np.arange(len(total_rev))
        colors = [
            "green" if r > 0 else "red" for r in total_rev["revision"]
        ]
        ax.bar(
            x + x_offset[key], total_rev["revision"],
            width=bar_width, label=label,
            color=COMMODITY_COLORS[label], alpha=0.8, edgecolor="white",
        )

    ax.axhline(y=0, color="black", linewidth=1)
    marketing_years = sorted(df["marketing_year"].unique())
    ax.set_xticks(np.arange(len(marketing_years)))
    ax.set_xticklabels(marketing_years, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Marketing Year")
    ax.set_ylabel("Total Revision: Final - First Estimate (MMT)")
    ax.set_title("USDA Forecast Bias: Total Revision from First to Final Estimate",
                 fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save_fig(fig, "revision_bias")


def run_revision_plots() -> None:
    """Generate all WASDE revision-related plots."""
    logger.info("--- Generating revision plots ---")
    plot_revision_funnels()
    plot_revision_magnitude()
    plot_revision_by_forecast_month()
    plot_cumulative_revision_paths()
    plot_revision_bias()


# ============================================================
# Model Comparison Plots
# ============================================================

def _load_model_predictions(model: str) -> pd.DataFrame | None:
    """Load predictions CSV for a model."""
    path = MODELS_DIR / f"{model.lower()}_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_forecast_overlay() -> None:
    """Overlay SARIMA and XGBoost forecasts against actual values."""
    sarima_df = _load_model_predictions("sarima")
    xgb_df = _load_model_predictions("xgboost")

    if sarima_df is None and xgb_df is None:
        logger.warning("No model predictions found, skipping forecast overlay")
        return

    targets = ["ending_stocks", "total_supply"]
    commodities = COMMODITY_ORDER

    fig, axes = plt.subplots(
        len(commodities), len(targets),
        figsize=(16, 5 * len(commodities)),
        sharex=False,
    )
    if len(commodities) == 1:
        axes = axes.reshape(1, -1)

    for i, commodity in enumerate(commodities):
        for j, target in enumerate(targets):
            ax = axes[i, j]

            # Plot actual
            actual_plotted = False
            for label, pred_df, color, ls in [
                ("SARIMA", sarima_df, "#E74C3C", "--"),
                ("XGBoost", xgb_df, "#3498DB", "-."),
            ]:
                if pred_df is None:
                    continue
                subset = pred_df[
                    (pred_df["commodity"] == commodity) &
                    (pred_df["target"] == target)
                ].sort_values("Market_Year")

                if subset.empty:
                    continue

                if not actual_plotted:
                    ax.plot(
                        subset["Market_Year"], subset["actual"],
                        "k-", linewidth=2, label="Actual", zorder=3,
                    )
                    actual_plotted = True

                ax.plot(
                    subset["Market_Year"], subset["predicted"],
                    color=color, linestyle=ls, linewidth=1.5,
                    label=label, alpha=0.8,
                )

                # Confidence interval for SARIMA
                if label == "SARIMA" and "lower_90" in subset.columns:
                    ax.fill_between(
                        subset["Market_Year"],
                        subset["lower_90"],
                        subset["upper_90"],
                        alpha=0.1, color=color,
                    )

            ax.set_title(
                f"{commodity} — {target.replace('_', ' ').title()}",
                fontsize=11,
            )
            ax.set_ylabel("1000 MT")
            ax.legend(fontsize=9)

    axes[-1, 0].set_xlabel("Marketing Year")
    axes[-1, 1].set_xlabel("Marketing Year")
    fig.suptitle("Model Forecasts vs Actual", fontsize=14, y=1.01)
    fig.tight_layout()
    _save_fig(fig, "forecast_overlay")


def plot_residual_diagnostics() -> None:
    """Residual analysis for each model."""
    models = []
    for name in ["sarima", "xgboost"]:
        df = _load_model_predictions(name)
        if df is not None:
            models.append((name.upper() if name == "sarima" else "XGBoost", df))

    if not models:
        return

    fig, axes = plt.subplots(
        len(models), 3, figsize=(18, 5 * len(models)),
    )
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    for i, (model_name, pred_df) in enumerate(models):
        # Use ending_stocks residuals (combine all commodities)
        subset = pred_df[pred_df["target"] == "ending_stocks"].dropna(
            subset=["actual", "predicted"]
        )
        if subset.empty:
            continue

        residuals = subset["actual"] - subset["predicted"]

        # Histogram
        axes[i, 0].hist(residuals, bins=25, color="#7f8c8d", edgecolor="white", alpha=0.8)
        axes[i, 0].axvline(x=0, color="red", linestyle="--")
        axes[i, 0].set_title(f"{model_name} — Residual Distribution")
        axes[i, 0].set_xlabel("Residual (Actual - Predicted)")

        # Predicted vs Actual scatter
        for commodity in COMMODITY_ORDER:
            c_subset = subset[subset["commodity"] == commodity]
            if c_subset.empty:
                continue
            axes[i, 1].scatter(
                c_subset["predicted"], c_subset["actual"],
                color=COMMODITY_COLORS[commodity], label=commodity,
                alpha=0.7, s=40,
            )

        # Perfect prediction line
        all_vals = pd.concat([subset["actual"], subset["predicted"]])
        lims = [all_vals.min() * 0.9, all_vals.max() * 1.1]
        axes[i, 1].plot(lims, lims, "k--", alpha=0.5, label="Perfect")
        axes[i, 1].set_title(f"{model_name} — Predicted vs Actual")
        axes[i, 1].set_xlabel("Predicted")
        axes[i, 1].set_ylabel("Actual")
        axes[i, 1].legend(fontsize=8)

        # Residuals over time
        for commodity in COMMODITY_ORDER:
            c_subset = subset[subset["commodity"] == commodity]
            if c_subset.empty:
                continue
            c_resid = c_subset["actual"] - c_subset["predicted"]
            axes[i, 2].plot(
                c_subset["Market_Year"], c_resid,
                color=COMMODITY_COLORS[commodity], label=commodity,
                marker="o", markersize=3,
            )
        axes[i, 2].axhline(y=0, color="black", linewidth=0.5)
        axes[i, 2].set_title(f"{model_name} — Residuals Over Time")
        axes[i, 2].set_xlabel("Marketing Year")
        axes[i, 2].set_ylabel("Residual")
        axes[i, 2].legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, "residual_diagnostics")


def plot_model_comparison_bars() -> None:
    """Bar chart comparing model metrics side-by-side."""
    metrics_path = MODELS_DIR / "model_comparison.csv"
    if not metrics_path.exists():
        logger.warning("No model comparison file found, skipping")
        return

    metrics_df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metric_names = [
        ("rmse", "RMSE (1000 MT)"),
        ("mae", "MAE (1000 MT)"),
        ("mape", "MAPE (%)"),
        ("direction_accuracy", "Direction Accuracy (%)"),
    ]

    for ax, (metric, label) in zip(axes.flat, metric_names):
        # Focus on ending_stocks
        subset = metrics_df[metrics_df["target"] == "ending_stocks"].copy()
        if subset.empty:
            continue

        models = subset["model"].unique()
        commodities = [c for c in COMMODITY_ORDER if c in subset["commodity"].values]
        x = np.arange(len(commodities))
        width = 0.8 / len(models)

        for k, model in enumerate(models):
            model_data = subset[subset["model"] == model]
            vals = [
                model_data[model_data["commodity"] == c][metric].values[0]
                if c in model_data["commodity"].values else 0
                for c in commodities
            ]
            ax.bar(x + k * width - width, vals, width, label=model, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(commodities, fontsize=10)
        ax.set_ylabel(label)
        ax.set_title(f"{label} — Ending Stocks", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Model Performance Comparison", fontsize=14, y=1.01)
    fig.tight_layout()
    _save_fig(fig, "model_comparison_bars")


def plot_feature_importance() -> None:
    """Top feature importance from XGBoost."""
    imp_path = MODELS_DIR / "xgboost_feature_importance.csv"
    if not imp_path.exists():
        return

    imp_df = pd.read_csv(imp_path)

    # One subplot per commodity, show ending_stocks target
    commodities = [c for c in COMMODITY_ORDER if c in imp_df["commodity"].values]
    if not commodities:
        return

    fig, axes = plt.subplots(1, len(commodities), figsize=(7 * len(commodities), 8))
    if len(commodities) == 1:
        axes = [axes]

    for ax, commodity in zip(axes, commodities):
        subset = imp_df[
            (imp_df["commodity"] == commodity) &
            (imp_df["target"] == "ending_stocks")
        ].nlargest(15, "importance")

        if subset.empty:
            continue

        ax.barh(
            range(len(subset)),
            subset["importance"].values,
            color=COMMODITY_COLORS.get(commodity, "gray"),
            alpha=0.8,
        )
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(
            [f.replace("_", " ") for f in subset["feature"].values],
            fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_title(f"{commodity}", fontsize=12)
        ax.set_xlabel("Feature Importance (avg gain)")

    fig.suptitle("XGBoost Top 15 Features — Ending Stocks", fontsize=14, y=1.01)
    fig.tight_layout()
    _save_fig(fig, "feature_importance")


def run_model_plots() -> None:
    """Generate all model-related plots."""
    logger.info("--- Generating model plots ---")
    plot_forecast_overlay()
    plot_residual_diagnostics()
    plot_model_comparison_bars()
    plot_feature_importance()


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument(
        "--section",
        choices=["eda", "revisions", "models", "all"],
        default="all",
        help="Which section to visualize",
    )
    args = parser.parse_args()

    if args.section in ("eda", "all"):
        run_eda_plots()

    if args.section in ("revisions", "all"):
        run_revision_plots()

    if args.section in ("models", "all"):
        # Check if any model results exist
        has_models = any(
            (MODELS_DIR / f).exists()
            for f in ["sarima_predictions.csv", "xgboost_predictions.csv"]
        )
        if has_models:
            run_model_plots()
        else:
            logger.info("No model results found yet, skipping model plots")

    logger.info("Visualization complete. Figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    sys.exit(main() or 0)
