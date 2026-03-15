# Forecasting Global Grain Supplies from USDA WASDE/PSD Data

Forecasting global ending stocks and total supply for wheat, corn, and rice using historical USDA data. This project ingests 65 years of Production, Supply & Distribution (PSD) balance sheets and 14 years of monthly WASDE revision history to build SARIMA and XGBoost forecasting models.

## Research Objectives

1. **Understand** the global grain supply/demand balance sheet structure and how it has evolved since 1960
2. **Analyze** USDA forecast revision dynamics — how estimates change from first forecast to final published value
3. **Forecast** ending stocks and total supply at the global level using time-series and ML models
4. **Compare** model performance: naive baseline vs SARIMA vs XGBoost with engineered features

## Data Sources

| Source | Coverage | Granularity | Format |
|--------|----------|-------------|--------|
| [USDA PSD](https://apps.fas.usda.gov/psdonline/) | 1960–2025 | Country-level, annual | Bulk CSV (no API key) |
| [USDA WASDE](https://www.usda.gov/oce/commodity/wasde/) | 2012–2026 | Monthly revisions per marketing year | XML via ESMIS API |

**Commodities:** Wheat (0410000), Corn (0440000), Rice Milled (0422110)

**Target variables:** Ending Stocks, Total Supply (global aggregates, 1000 MT)

## Key Results

### Ending Stocks Forecasting (1-step ahead, expanding window CV)

| Model | Wheat MAPE | Corn MAPE | Rice MAPE | Wheat DirAcc | Corn DirAcc | Rice DirAcc |
|-------|-----------|-----------|-----------|-------------|-------------|-------------|
| Naive | — | 13.7% | — | — | 52.3% | — |
| SARIMA | 8.4% | 13.8% | **5.1%** | 53.3% | 53.3% | 82.2% |
| XGBoost | **7.3%** | **11.3%** | 8.0% | **75.6%** | **73.3%** | **82.2%** |

- XGBoost achieves significantly better **direction accuracy** (73–82%) by leveraging cross-commodity signals and lagged features
- SARIMA excels on rice where the upward trend is smooth and consistent
- Both models beat the naive (last-year) baseline
- USDA tends to under-estimate ending stocks (positive revision bias), with corn showing the most revision volatility

### Visualizations (19 figures)

Generated to `results/figures/` at 300 DPI:

- **EDA (10):** balance sheet overview, ending stocks/total supply trends, stocks-to-use ratio, production vs consumption, YoY changes, component shares, ACF/PACF diagnostics, cross-commodity correlations, structural breaks timeline
- **Revision Analysis (5):** revision funnels, magnitude distribution, revision by forecast month, cumulative revision paths, systematic bias
- **Models (4):** forecast overlay (actual vs predicted), residual diagnostics, model comparison bars, XGBoost feature importance

## Project Structure

```
forecasting-wasde/
├── configs/
│   └── commodities.yaml          # Commodity codes, attribute IDs, data source URLs
├── data/
│   ├── raw/psd/                  # PSD bulk CSV
│   ├── raw/wasde/                # 171 WASDE XML files
│   ├── processed/                # Parquet: balance sheets, revision history
│   └── features/                 # Feature matrix for ML
├── src/
│   ├── config.py                 # Paths, constants, YAML config loading
│   ├── data/
│   │   ├── psd_fetcher.py        # Download + cache PSD bulk CSV
│   │   ├── psd_processor.py      # Filter, aggregate to world totals, validate
│   │   ├── wasde_fetcher.py      # Download WASDE XMLs via ESMIS API
│   │   └── wasde_processor.py    # Parse XML, extract revision history
│   ├── features/
│   │   └── engineer.py           # 56 engineered features
│   └── models/
│       ├── sarima.py             # SARIMA with auto_arima + expanding window
│       ├── xgboost_model.py      # XGBoost with tuning + SHAP
│       └── evaluate.py           # Metrics computation + model comparison
├── scripts/
│   ├── 01_fetch_data.py          # Download PSD + WASDE data
│   ├── 02_process_data.py        # Process raw data to parquet
│   ├── 03_eda_analysis.py        # EDA: stats, stationarity, correlations
│   ├── 04_engineer_features.py   # Build feature matrix
│   ├── 05_train_sarima.py        # Train SARIMA models
│   ├── 06_train_xgboost.py       # Train XGBoost models
│   ├── 07_evaluate.py            # Cross-model evaluation
│   └── 08_visualize.py           # Generate all figures (separated from analysis)
├── results/
│   ├── eda/                      # Summary stats, stationarity tests, correlations
│   ├── models/                   # Predictions, metrics, SHAP values
│   └── figures/                  # All PNG plots (300 DPI)
├── Makefile                      # Pipeline orchestration
├── requirements.txt              # Pinned dependencies
└── STATUS.md                     # Current project state
```

## Setup

```bash
# Clone and create virtual environment
git clone <repo-url>
cd forecasting-wasde
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Or use the Makefile:

```bash
make setup
```

## Usage

### Full Pipeline (all 3 commodities, full history)

```bash
make all
```

This runs: fetch → process → eda → features → sarima → xgboost → evaluate → visualize

### Dev Mode (wheat only, faster iteration)

```bash
make dev
```

### Individual Steps

```bash
make fetch          # Download PSD + WASDE data
make process        # Process raw data
make eda            # Run EDA analysis
make features       # Engineer features
make sarima         # Train SARIMA models
make xgboost        # Train XGBoost models
make evaluate       # Compare models
make visualize      # Generate all plots
```

### Script-Level Options

```bash
# Fetch with options
python scripts/01_fetch_data.py --commodities wheat corn rice --wasde-start-year 2012

# Process specific commodities
python scripts/02_process_data.py --commodities wheat --start-year 2000

# Train on specific targets
python scripts/05_train_sarima.py --all --targets ending_stocks --min-train 25

# Visualize specific section
python scripts/08_visualize.py --section models
```

## Feature Engineering (56 features)

| Category | Features | Description |
|----------|----------|-------------|
| Lag features | 21 | 1-3 year lags for 7 balance sheet variables |
| Derived ratios | 6 | Stocks-to-use, production/consumption, import/export shares + lagged |
| YoY changes | 8 | Absolute and percentage changes for 4 key variables |
| Rolling statistics | 9 | 3yr and 5yr rolling means + 5yr linear trend for 3 variables |
| Cross-commodity | 3 | Other commodities' lagged ending stocks |
| WASDE revision | 9 | Historical avg revision %, std, and bias per commodity |

## Methodology

### Data Processing
- PSD country-level data aggregated to world totals (no pre-built world row exists)
- Balance sheet identity validated: Beginning Stocks + Production + Imports = Domestic Consumption + Exports + Ending Stocks
- WASDE XML parsed for 3 sub-reports (sr18=wheat, sr22=corn, sr24=rice), extracting World-level estimates

### Stationarity
- All series non-stationary in levels (ADF test, p > 0.05)
- All stationary after first differencing (ADF p < 0.001)
- SARIMA uses d=1 (auto-selected by pmdarima)

### Evaluation
- **Expanding-window time-series CV** — strict temporal ordering, no data leakage
- Minimum training window: 20 years
- Test period: 46 years (1980–2025)
- Metrics: RMSE, MAE, MAPE, direction accuracy, bias

### Models
- **Naive baseline:** predict last year's value
- **SARIMA:** `pmdarima.auto_arima` with automatic order selection, re-tuned every 10 steps
- **XGBoost:** hyperparameter tuning via TimeSeriesSplit CV, SHAP for interpretability, NaN features filled with training median

## Dependencies

- Python 3.11+
- pandas, numpy, pyarrow, scipy
- statsmodels, pmdarima (SARIMA)
- xgboost, scikit-learn, shap (ML)
- matplotlib, seaborn (visualization)
- requests, pyyaml, tqdm, openpyxl

See `requirements.txt` for pinned versions.

## Reproducibility

- Random seed: 42 (set in `src/config.py`, used across all models)
- All data downloaded from public USDA sources (no API key required)
- Deterministic pipeline via Makefile
- Results saved to files; visualization is fully separated from analysis
