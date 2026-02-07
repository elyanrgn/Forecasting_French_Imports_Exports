# Macroeconometrics & Machine Learning – Forecasting French Exports and Imports

This repository implements a comprehensive benchmark of traditional macroeconometric models and modern machine learning methods for **forecasting French exports and imports** in a high‑dimensional setting.

The project is inspired by Medeiros et al. (2021), who show that machine learning models can outperform classical benchmarks for U.S. inflation. Here, the same philosophy is applied to **French trade flows**, using a rich Euro Area dataset.

> Project report: *Macroeconometrics and Machine Learning* (Rougon, Pliszczak, Chadeuf, 2026)

---

## Overview

- Target variables: **French exports and imports** (monthly, 2 series)
- Predictors: **552 macro‑financial indicators** for France and the Euro Area
- Sample: **305 monthly observations** after pre‑processing
- Objective:
  - Compare **traditional time‑series models** (ARIMA / ARMAX / SARIMAX)
  - **Linear ML models**: Ridge, Lasso, Elastic Net
  - **Factor models** (DFM + Ridge)
  - **Non‑linear ML**: Random Forest, XGBoost, Neural Network
- Evaluation:
  - **Out‑of‑sample rolling window** forecasts
  - Horizons: **h = 1, 3, 6, 12 months**
  - Loss functions: MAE and RMSE (often reported **relative to ARIMA**)

---

## Data

The empirical analysis relies on the **EA‑MD‑QD** database for the Euro Area and member countries (Barigozzi & Lissona, 2024), restricted to **France**:

- Sources: Eurostat, OECD, FRED, ECB, BIS…
- Mixed frequencies (monthly/quarterly) harmonized to **monthly** via **Chow–Lin** temporal disaggregation (regression‑based, AR(1) residuals).
- Stationarity:
  - Variables are differenced to reach I(0) (first or second differences).
  - Transformations are provided by the original EA‑MD‑QD code.
- Missing data and COVID‑19 period (2020–2021):
  - Treated as missing for real variables.
  - Imputed using an **EM algorithm** based on a **factor model** (McCracken & Ng, 2016).
  - Number of factors chosen by **Bai & Ng (2002)** information criteria.

Notation in the report:

- \( y \in \mathbb{R}^{305 \times 2} \): exports and imports
- \( X \in \mathbb{R}^{305 \times 552} \): covariates

> The raw EA‑MD‑QD data are not redistributed here. Please refer to the original authors for access.

---

## Models

### Benchmarks

- **ARMA(2,1)** on each target (no exogenous regressors)
- **ARMAX / SARIMAX** with selected exogenous variables

### Linear ML models

- **Ridge Regression**
- **Lasso**
- **Elastic Net**

Regularization parameters (and mixing parameter for Elastic Net) are tuned via **cross‑validation**, rather than BIC, to exploit standard ML tooling and improve computational efficiency.

### Factor model

- **Dynamic Factor Model (DFM + Ridge)**:
  - Extracts a small number of common factors via **PCA**
  - Uses these factors as regressors in a Ridge regression for exports/imports

### Non‑linear ML models

- **Random Forest**
- **XGBoost**
- **Feedforward Neural Network**

Hyperparameters are optimized with **Optuna** (Akiba et al., 2019), including:

- Tree depth, number of trees, learning rate, subsampling, etc. (RF, XGBoost)
- Number of layers, neurons, activation functions, regularization (NN)

---

## Forecasting Strategy

To mimic a **real‑time forecasting environment**, the project uses a **rolling window** scheme:

1. Choose a fixed training window length.
2. For each evaluation date \( t \):
   - Re‑estimate the model on the most recent window.
   - Produce direct forecasts at horizons **h = 1, 3, 6, 12** months ahead.
3. Roll the window forward by one month and repeat.
4. Aggregate out‑of‑sample errors (MAE, RMSE) across time.

This setup allows:

- Robust comparison across models
- Analysis of performance by **forecast horizon**
- Detection of potential structural breaks

---

## Repository Structure

Adapted example (update to match your actual layout):

```text
.
├── data/
│   ├── raw/              # EA‑MD‑QD input data (not tracked / or synthetic example)
│   └── processed/        # Monthly, cleaned, stationarized data (y, X)
├── src/
│   ├── data_prep.py      # Loading, Chow–Lin, EM imputation, transformations
│   ├── models_linear.py  # ARIMA / ARMAX / SARIMAX, Ridge, Lasso, Elastic Net
│   ├── models_nonlinear.py
|   ├── model_saving.py 
│   ├── factor_model.py   # DFM construction and DFM + Ridge
│   ├── rolling_eval.py   # Rolling‑window forecasting and evaluation
│   └── plots.py          # Tables and figures (MAE/RMSE, win rates, etc.)
├── outputs/
│   └── todo           # plot models performance 
├── models/
│   ├── neural_network_model.pkl    # Neural Network Mode savec
│   |── random_forest_model.pkl #RandomForest model saved
|   └── xgboost_model.pkl #Xgboost model saved
├── requirements.txt
└── README.md
