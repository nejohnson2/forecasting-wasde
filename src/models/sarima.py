"""SARIMA baseline model for grain supply forecasting.

Uses pmdarima's auto_arima for automatic order selection.
Supports expanding-window time-series cross-validation.
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pmdarima as pm
from tqdm import tqdm

from src.config import MODELS_DIR, SEED

logger = logging.getLogger(__name__)

np.random.seed(SEED)


@dataclass
class SARIMAResult:
    """Container for SARIMA forecast results."""
    commodity: str
    target: str
    predictions: pd.DataFrame  # year, actual, predicted, lower, upper
    order: tuple
    seasonal_order: tuple
    aic: float


def train_sarima(
    series: np.ndarray,
    years: np.ndarray,
    commodity: str,
    target: str,
    min_train: int = 20,
    forecast_horizon: int = 1,
) -> SARIMAResult:
    """Train SARIMA with expanding-window CV.

    Args:
        series: Target variable values (chronologically ordered).
        years: Corresponding years.
        commodity: Commodity name for labeling.
        target: Target variable name.
        min_train: Minimum training window size.
        forecast_horizon: Number of steps ahead to forecast.

    Returns:
        SARIMAResult with predictions and model info.
    """
    n = len(series)
    predictions = []

    # Use the first min_train years to find initial order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initial_model = pm.auto_arima(
            series[:min_train],
            seasonal=False,
            d=1,
            max_p=5,
            max_q=5,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            random_state=SEED,
        )
    best_order = initial_model.order
    logger.info("  %s %s: initial ARIMA order %s", commodity, target, best_order)

    # Expanding window forecasts
    for t in tqdm(
        range(min_train, n),
        desc=f"SARIMA {commodity} {target}",
        leave=False,
    ):
        train = series[:t]
        actual = series[t] if t < n else np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Refit with auto_arima every 10 steps, otherwise use fixed order
                if (t - min_train) % 10 == 0:
                    model = pm.auto_arima(
                        train,
                        seasonal=False,
                        d=1,
                        max_p=5,
                        max_q=5,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore",
                        random_state=SEED,
                    )
                    best_order = model.order
                else:
                    model = pm.ARIMA(order=best_order)
                    model.fit(train)

                fc, ci = model.predict(
                    n_periods=forecast_horizon,
                    return_conf_int=True,
                    alpha=0.1,
                )
                pred = fc[-1]  # Take the h-step-ahead forecast
                lower, upper = ci[-1]

            except Exception as e:
                logger.debug("SARIMA failed at t=%d: %s", t, e)
                pred = train[-1]  # Fallback to naive
                lower = upper = pred

        predictions.append({
            "Market_Year": int(years[t]),
            "actual": float(actual),
            "predicted": float(pred),
            "lower_90": float(lower),
            "upper_90": float(upper),
        })

    pred_df = pd.DataFrame(predictions)

    return SARIMAResult(
        commodity=commodity,
        target=target,
        predictions=pred_df,
        order=best_order,
        seasonal_order=(0, 0, 0, 0),
        aic=float(initial_model.aic()),
    )


def run_sarima_forecasts(
    df: pd.DataFrame,
    targets: list[str] | None = None,
    min_train: int = 20,
) -> list[SARIMAResult]:
    """Run SARIMA for all commodities and targets.

    Args:
        df: PSD balance sheet DataFrame.
        targets: Target columns to forecast. Defaults to ending_stocks, total_supply.
        min_train: Minimum training window.

    Returns:
        List of SARIMAResult objects.
    """
    if targets is None:
        targets = ["ending_stocks", "total_supply"]

    results = []

    for commodity, grp in df.groupby("Commodity_Description"):
        grp = grp.sort_values("Market_Year")
        years = grp["Market_Year"].values

        for target in targets:
            if target not in grp.columns:
                logger.warning("Target %s not in data", target)
                continue

            series = grp[target].values
            if len(series) < min_train + 5:
                logger.warning("  %s: too few observations for %s", commodity, target)
                continue

            logger.info("Training SARIMA for %s - %s (%d obs)", commodity, target, len(series))
            result = train_sarima(
                series, years, commodity, target, min_train=min_train
            )
            results.append(result)

            # Log performance summary
            pred_df = result.predictions
            mae = np.mean(np.abs(pred_df["actual"] - pred_df["predicted"]))
            logger.info("  %s %s: MAE=%.1f, order=%s", commodity, target, mae, result.order)

    return results


def save_sarima_results(results: list[SARIMAResult]) -> None:
    """Save SARIMA predictions and metadata."""
    all_preds = []
    metadata = []

    for r in results:
        preds = r.predictions.copy()
        preds["commodity"] = r.commodity
        preds["target"] = r.target
        preds["model"] = "SARIMA"
        all_preds.append(preds)

        metadata.append({
            "commodity": r.commodity,
            "target": r.target,
            "order": str(r.order),
            "seasonal_order": str(r.seasonal_order),
            "aic": r.aic,
        })

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(MODELS_DIR / "sarima_predictions.csv", index=False)

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(MODELS_DIR / "sarima_metadata.csv", index=False)

    logger.info("Saved SARIMA results to %s", MODELS_DIR)
