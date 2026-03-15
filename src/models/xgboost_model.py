"""XGBoost model for grain supply forecasting.

Uses engineered features with expanding-window time-series CV.
Includes SHAP analysis for interpretability.
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from src.config import MODELS_DIR, SEED
from src.features.engineer import get_feature_columns

logger = logging.getLogger(__name__)

np.random.seed(SEED)


@dataclass
class XGBResult:
    """Container for XGBoost forecast results."""
    commodity: str
    target: str
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    shap_values: np.ndarray | None
    shap_feature_names: list[str] | None
    best_params: dict


DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "verbosity": 0,
}


def _tune_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = 3,
) -> dict:
    """Simple hyperparameter tuning via time-series CV.

    Tests a small grid and returns the best params by RMSE.
    """
    if len(X_train) < 15:
        return DEFAULT_PARAMS.copy()

    param_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 150},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200},
        {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100},
        {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 250},
    ]

    best_rmse = np.inf
    best_params = DEFAULT_PARAMS.copy()

    n_splits = min(n_splits, len(X_train) // 5)
    if n_splits < 2:
        return best_params

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for params in param_grid:
        trial_params = DEFAULT_PARAMS.copy()
        trial_params.update(params)

        fold_rmses = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train[val_idx]

            model = xgb.XGBRegressor(**trial_params)
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - preds) ** 2))
            fold_rmses.append(rmse)

        mean_rmse = np.mean(fold_rmses)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = trial_params

    return best_params


def train_xgboost(
    df: pd.DataFrame,
    commodity: str,
    target: str,
    min_train: int = 20,
) -> XGBResult:
    """Train XGBoost with expanding-window CV for a single commodity/target.

    Args:
        df: Feature matrix for a single commodity (sorted by year).
        commodity: Commodity name.
        target: Target column name.
        min_train: Minimum training window.

    Returns:
        XGBResult with predictions, feature importance, and SHAP values.
    """
    feature_cols = get_feature_columns(df)
    years = df["Market_Year"].values
    y = df[target].values
    n = len(df)

    # Drop rows with NaN in target
    valid_mask = ~np.isnan(y)
    if not valid_mask.all():
        logger.info("  Dropping %d rows with NaN target", (~valid_mask).sum())

    predictions = []
    all_importances = []

    for t in tqdm(
        range(min_train, n),
        desc=f"XGB {commodity} {target}",
        leave=False,
    ):
        # Training data: all rows up to t
        train_mask = np.arange(n) < t
        train_mask &= valid_mask

        X_train = df.loc[train_mask, feature_cols].copy()
        y_train = y[train_mask]

        # Fill NaN features with column median from training set
        for col in X_train.columns:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)

        # Test point
        X_test = df.iloc[[t]][feature_cols].copy()
        for col in X_test.columns:
            X_test[col] = X_test[col].fillna(X_train[col].median())

        actual = y[t]

        try:
            # Tune params periodically
            if (t - min_train) % 10 == 0:
                best_params = _tune_xgb(X_train, y_train)

            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train, y_train, verbose=False)

            pred = model.predict(X_test)[0]

            # Feature importance from this fold
            imp = pd.Series(
                model.feature_importances_,
                index=feature_cols,
            )
            all_importances.append(imp)

        except Exception as e:
            logger.debug("XGBoost failed at t=%d: %s", t, e)
            pred = y_train[-1]  # Fallback to naive

        predictions.append({
            "Market_Year": int(years[t]),
            "actual": float(actual),
            "predicted": float(pred),
        })

    pred_df = pd.DataFrame(predictions)

    # Average feature importance across folds
    if all_importances:
        avg_importance = pd.concat(all_importances, axis=1).mean(axis=1)
        importance_df = avg_importance.reset_index()
        importance_df.columns = ["feature", "importance"]
        importance_df = importance_df.sort_values("importance", ascending=False)
    else:
        importance_df = pd.DataFrame(columns=["feature", "importance"])

    # SHAP analysis on the final model (trained on all data except last point)
    shap_values = None
    shap_features = None
    try:
        # Retrain on all available data for SHAP
        X_full = df.loc[valid_mask, feature_cols].copy()
        y_full = y[valid_mask]
        for col in X_full.columns:
            X_full[col] = X_full[col].fillna(X_full[col].median())

        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X_full, y_full, verbose=False)

        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_full)
        shap_features = feature_cols
    except Exception as e:
        logger.warning("SHAP analysis failed: %s", e)

    return XGBResult(
        commodity=commodity,
        target=target,
        predictions=pred_df,
        feature_importance=importance_df,
        shap_values=shap_values,
        shap_feature_names=shap_features,
        best_params=best_params,
    )


def run_xgboost_forecasts(
    df: pd.DataFrame,
    targets: list[str] | None = None,
    min_train: int = 20,
) -> list[XGBResult]:
    """Run XGBoost for all commodities and targets.

    Args:
        df: Feature matrix DataFrame (all commodities).
        targets: Target columns. Defaults to ending_stocks, total_supply.
        min_train: Minimum training window.

    Returns:
        List of XGBResult objects.
    """
    if targets is None:
        targets = ["ending_stocks", "total_supply"]

    results = []

    for commodity, grp in df.groupby("Commodity_Description"):
        grp = grp.sort_values("Market_Year").reset_index(drop=True)

        for target in targets:
            if target not in grp.columns:
                continue

            logger.info("Training XGBoost for %s - %s (%d obs)", commodity, target, len(grp))
            result = train_xgboost(grp, commodity, target, min_train=min_train)
            results.append(result)

            # Log performance
            pred_df = result.predictions
            valid = pred_df.dropna(subset=["actual", "predicted"])
            if len(valid) > 0:
                mae = np.mean(np.abs(valid["actual"] - valid["predicted"]))
                logger.info("  %s %s: MAE=%.1f", commodity, target, mae)

    return results


def save_xgboost_results(results: list[XGBResult]) -> None:
    """Save XGBoost predictions, feature importance, and SHAP values."""
    all_preds = []
    all_importance = []

    for r in results:
        preds = r.predictions.copy()
        preds["commodity"] = r.commodity
        preds["target"] = r.target
        preds["model"] = "XGBoost"
        all_preds.append(preds)

        imp = r.feature_importance.copy()
        imp["commodity"] = r.commodity
        imp["target"] = r.target
        all_importance.append(imp)

        # Save SHAP values per commodity/target
        if r.shap_values is not None:
            safe_commodity = r.commodity.lower().replace(",", "").replace(" ", "_")
            shap_path = MODELS_DIR / f"shap_values_{safe_commodity}_{r.target}.npz"
            np.savez(
                shap_path,
                shap_values=r.shap_values,
                feature_names=r.shap_feature_names,
            )

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(MODELS_DIR / "xgboost_predictions.csv", index=False)

    imp_df = pd.concat(all_importance, ignore_index=True)
    imp_df.to_csv(MODELS_DIR / "xgboost_feature_importance.csv", index=False)

    logger.info("Saved XGBoost results to %s", MODELS_DIR)
