# Walmart Weekly Sales — EDA & Forecasting

End-to-end exploratory analysis and forecasting pipeline for the **Walmart Recruiting – Store Sales Forecasting** dataset. The project covers data auditing, multi-table merging, time-series feature engineering, a Prophet baseline, a global gradient-boosted model (LightGBM) trained with the competition's holiday-weighted metric (WMAE), SHAP-based interpretation, and a statistical benchmark (AutoARIMA / SARIMA via `statsforecast`).

> Notebook: `Walmart_EDA_and_model.ipynb` (developed on Google Colab)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Pipeline Overview](#pipeline-overview)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Key Findings](#key-findings)
7. [Known Limitations & Next Steps](#known-limitations--next-steps)
8. [Repository Structure](#repository-structure)
9. [Setup & Usage](#setup--usage)
10. [Tech Stack](#tech-stack)

---

## Problem Statement

Forecast weekly sales for **45 Walmart stores × up to 81 departments** (~3,300 individual time series) over a 39-week horizon (2012-11-02 → 2013-07-26), where:

- Series are **heterogeneous**: different lengths, scales and seasonal strengths. Some have fewer than 52 weeks of history, so purely seasonal models cannot be fit on them.
- **Holiday weeks** (Super Bowl, Labor Day, Thanksgiving, Christmas) drive the largest spikes and are weighted **5×** in the evaluation metric.
- Promotional **MarkDown** data exists only after Nov 2011 and is heavily missing.

**Evaluation metric** — Weighted MAE:

```
WMAE = Σ wᵢ · |yᵢ − ŷᵢ| / Σ wᵢ ,   wᵢ = 5 if holiday week else 1
```

## Dataset

| File | Shape | Description |
|---|---|---|
| `train.csv` | 421,570 × 5 | Store, Dept, Date, Weekly_Sales, IsHoliday (2010-02-05 → 2012-10-26, 143 weeks) |
| `stores.csv` | 45 × 3 | Store Type (A/B/C) and Size |
| `features.csv` | 8,190 × 12 | Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment, IsHoliday |
| `test.csv` | — | Held-out period used for final predictions |

Source: [Kaggle – Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting).

**Audit findings**

- No missing values or duplicates in `train`.
- `Weekly_Sales` ranges from **−4,988.94 to 693,099.36** (mean ≈ 15,981, median ≈ 7,612). Negative values are product returns and are **kept** so revenue totals stay accurate.
- `MarkDown1–5` are missing in 64–74% of merged rows.

## Pipeline Overview

```
train.csv ─┐
stores.csv ─┼─► merge (store, date, isholiday) ─► clean / impute ─► feature engineering
features.csv┘                                                            │
                                                                         ▼
                    EDA  ◄──────────────────────────────────  modelling & evaluation
                                                              ├─ Prophet (aggregate + per series)
                                                              ├─ LightGBM (global, WMAE-weighted)
                                                              ├─ SHAP explainability
                                                              └─ AutoARIMA / SARIMA (statsforecast)
                                                                         │
                                                                         ▼
                                                       test-set inference → predictions CSV
```

## Methodology

### 1. Preprocessing
- Column names normalised (`strip → lower → snake_case`); `Date` parsed to `datetime`.
- Left-join `train` ⟵ `features` on `(store, date, isholiday)` ⟵ `stores` on `store` → 421,570 × 16.
- **Economic regressors** (`temperature`, `fuel_price`, `cpi`, `unemployment`): per-store forward-fill then back-fill, with global median as final fallback.
- **MarkDowns**: excluded from the Prophet baseline; for the ML model, `NaN → 0` (interpreted as "no promotion active").
- `type`, `store`, `dept` cast to `category`; `isholiday` cast to `int`.

### 2. Feature Engineering
Computed per `(store, dept)` series, sorted by date:

| Group | Features |
|---|---|
| Calendar | `year`, `month`, `week_of_year`, `day_of_week`, `is_weekend` |
| Lags | `lag_1, 2, 3, 4, 7, 14, 21, 52` |
| Rolling | `rolling_4`, `rolling_12` (mean of `shift(1)` so the current week is never included) |
| Exogenous | temperature, fuel price, CPI, unemployment, MarkDown1–5, store type, store size |
| Flags | `isholiday` |

### 3. Exploratory Analysis
- Sales by department (heavy-tailed; a few departments dominate revenue → single global scale is inadequate).
- Mean weekly sales by year and month-over-month seasonality profile.
- Pearson correlation of regressors with sales — all |r| < 0.03 (fuel_price −0.0001, temperature −0.0023, CPI −0.0209, unemployment −0.0259).
- Outlier screen: per-series z-score > 4. Outliers correspond to promotional/holiday events, so they are **flagged, not removed**.

### 4. Models

**a) Prophet baseline**
- Aggregate total weekly sales across all stores; yearly seasonality, 52-week forecast, component plots.
- Per-series Prophet for the top-5 revenue (store, dept) pairs — all in Dept 92 (Stores 14, 2, 20, 13, 4) — with US country holidays and a 26-week horizon.

**b) LightGBM (global model)**
- One model trained across all series, with `store`/`dept` as categorical features, so short series borrow strength from long ones.
- Objective `regression_l1` (aligns with MAE); `num_leaves=63`, `learning_rate=0.05`, `n_estimators=600`, `random_state=42`.
- **Holiday sample weights**: 5× for holiday weeks, normalised by the **mean** (not the sum) so the effective gradient scale stays suitable for tree splitting. Dividing by the sum shrinks weights to ~1/N and can stall splits.
- **Time-based split**: train `< 2012-04-01` (332,778 rows), validate on `≥ 2012-04-01` (88,792 rows).
- SHAP `TreeExplainer` for global feature attribution.

**c) Statistical benchmark**
- `statsforecast` `AutoARIMA` (non-seasonal) and `AutoARIMA(season_length=52)` fitted per series on the same train split, `freq='W-FRI'`, evaluated on MAE.

**d) Final inference**
- Refit LightGBM on all labelled data (with holiday weights), concatenate `train` + `test`, rebuild lag/rolling features per series, filter to the test window (`date ≥ 2012-11-02`), predict, and export `walmart_sales_predictions.csv`.

## Results

| Model | Scope | Eval window | MAE | WMAE | RMSE |
|---|---|---|---|---|---|
| Prophet (yearly seasonality) | Total sales, all stores | Last 52 weeks | 2,612,036 | — | 4,003,769 |
| LightGBM | Per store-dept rows | 2012-04-01 → 2012-10-26 | 827.96 | 835.91 | — |

Prophet per-series next-week point forecasts for the top-5 series ranged from ~$190K to ~$252K.

> ⚠️ The two rows are on **different scales** (aggregate chain total vs. individual store-dept rows) and are not directly comparable. See limitations below regarding the LightGBM figure.

## Key Findings

- **Seasonality and holidays dominate.** Macroeconomic regressors show near-zero linear correlation with sales; the Christmas/Thanksgiving peaks are the primary structure and Prophet captures them but under-shoots peak magnitude.
- **A global model suits this data.** Series lengths and scales vary widely; sharing parameters via categorical store/dept features and lag-based inputs is more practical than fitting thousands of independent local models.
- **Metric alignment matters.** Using L1 loss plus holiday-weighted samples directly optimises the WMAE target.
- **Returns and promo spikes are signal.** Negative sales and z > 4 outliers reflect genuine business events and are retained.

## Known Limitations & Next Steps

Documented here for transparency; these are the highest-value fixes before treating the metrics as production-grade.

1. **Target leakage via the `z` column.** The per-series z-score of `weekly_sales` is computed from the full series (including the target week) and is currently left in `X_train`/`X_test`. It must be dropped from the feature set; the reported LightGBM error is therefore optimistic.
2. **Train/inference lag mismatch.** In validation, lag features for the hold-out weeks are computed from true sales inside the hold-out window (effectively one-step-ahead). At final inference the test target is unknown, so short lags (`lag_1…lag_21`) and rolling means are `NaN` for most of the 39-week horizon. Fix: use only lags ≥ forecast horizon (e.g. `lag_52`, plus lags of ≥ 39 weeks), or forecast recursively / train direct multi-horizon models.
3. **Row-based shifts.** Lags use `shift(n)` on rows, not on calendar dates. Series with missing weeks will misalign (e.g. `lag_52` ≠ exactly 52 weeks). Reindex each series to a complete weekly grid first.
4. **Unused/leaky naming.** `lag_7/14/21` are 7/14/21 *weeks*, not days.
5. **Validation scheme.** A single date split is used. Expanding-window (rolling-origin) cross-validation across multiple holiday seasons is recommended, especially before hyperparameter tuning.
6. **Short-history series.** Series with < 52 weeks have no `lag_52`; consider a fallback (department/store-type averages or a naive seasonal fallback).
7. **MarkDown handling.** Zero-fill conflates "no promo" with "not recorded" (pre-Nov 2011). Add a `markdown_available` indicator.
8. **Benchmark completeness.** The AutoARIMA/SARIMA benchmark is wired up but results are not yet compared against LightGBM under an identical horizon and metric (WMAE).
9. **Prophet baseline.** Aggregate backtest uses only ~2 years of history for yearly seasonality (Prophet warns about under-identification); the narrative MAE in the notebook markdown should be reconciled with the computed value.
10. **Possible extensions:** Optuna tuning, quantile regression for prediction intervals, hierarchical reconciliation (store → dept), holiday-proximity features (weeks to/from each holiday), and a custom WMAE eval function for early stopping.

## Repository Structure

```
.
├── Walmart_EDA_and_model.ipynb     # full analysis and modelling notebook
├── data/                           # train.csv, test.csv, stores.csv, features.csv (not tracked)
├── outputs/
│   ├── walmart_cleaned.csv         # merged + engineered dataset
│   ├── walmart_forecast_agg.csv    # Prophet aggregate forecast (ds, yhat, yhat_lower, yhat_upper)
│   └── walmart_sales_predictions.csv   # final test-set predictions
└── README.md
```

## Setup & Usage

```bash
git clone [<your-repo-url>](https://github.com/Adi21032004/Walmart_Sales_Forecasting/edit/main/README.md)
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn prophet lightgbm shap statsforecast utilsforecast
```

1. Download the competition data from Kaggle and place the CSVs in `data/` (the notebook's Colab `files.upload()` cells can be replaced with `pd.read_csv("data/...")`).
2. Run `Walmart_EDA_and_model.ipynb` top to bottom.
3. Predictions are written to `walmart_sales_predictions.csv`.

## Tech Stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `Prophet` · `LightGBM` · `SHAP` · `statsforecast` (AutoARIMA) · `Google Colab`
