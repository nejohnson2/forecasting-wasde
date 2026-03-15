# Status — Forecasting WASDE

**Last updated:** 2026-03-14

## Completed

### Phase 1: Data Foundation
- PSD bulk CSV fetcher/processor with world-total aggregation
- Balance sheet validation (supply = distribution identity)
- 198 rows: 3 commodities x 66 years (1960-2025)

### Phase 2: WASDE Ingestion
- ESMIS API fetcher downloads 171 XML files (2012-2026)
- XML parser handles both filename formats and nested Cell elements
- 786 revision history records with month-over-month and cumulative metrics

### Phase 3: EDA & Visualization
- Summary statistics, stationarity tests (ADF/KPSS), cross-commodity correlations
- Structural breaks analysis
- 10 EDA plots + 5 revision plots in results/figures/

### Phase 4: Feature Engineering
- 56 engineered features: lags (1-3yr), ratios, YoY changes, rolling stats, cross-commodity, WASDE revision features
- Feature matrix saved to data/features/feature_matrix.parquet

### Phase 5: Modeling & Evaluation
- **SARIMA** (pmdarima auto_arima, expanding-window CV, min 20yr train)
- **XGBoost** (hyperparameter tuning via TSCV, SHAP analysis)
- **Naive baseline** for comparison
- 4 model visualization plots: forecast overlay, residuals, comparison bars, feature importance

## Key Results (Ending Stocks)

| Model   | Wheat MAPE | Corn MAPE | Rice MAPE | Wheat DirAcc | Corn DirAcc | Rice DirAcc |
|---------|------------|-----------|-----------|--------------|-------------|-------------|
| Naive   | —          | 13.7%     | —         | —            | 52.3%       | —           |
| SARIMA  | 8.4%       | 13.8%     | 5.1%      | 53.3%        | 53.3%       | 82.2%       |
| XGBoost | 7.3%       | 11.3%     | 8.0%      | 75.6%        | 73.3%       | 82.2%       |

- XGBoost has better direction accuracy across all commodities
- SARIMA has lower RMSE for rice and total_supply targets (smoother trend = easier for ARIMA)
- XGBoost struggles with total_supply (strong upward trend hard to capture with tree models)

## What's Left
- [ ] README.md with full project documentation
- [ ] Consider TFT or other deep learning approach (if warranted)
- [ ] Write up findings for internal research report

## Known Issues
- 4 early-1980s PSD rows have minor supply/distribution discrepancies (aggregation artifacts)
- WASDE revision features are NaN before 2012 (WASDE data starts there) — filled with median during XGBoost training
- XGBoost overfits on total_supply due to strong trend component; consider differencing target
