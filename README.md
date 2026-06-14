# GB Electricity Demand Forecaster

An interactive web app that forecasts Great Britain's national electricity
demand 24 hours ahead using a multivariate LSTM neural network trained on NESO
open data. Adjust temperature, day type, and wind generation to explore how
national demand responds.

**Live app:** https://gb-demand-forecaster-eceik67aicsv5qepuepnau.streamlit.app/

Built by Nasir.

---

## What this project does

Given the last 48 hours of electricity demand, weather, and calendar context,
the model predicts national demand for each of the next 24 hours. Day-ahead
forecasting is the most important horizon in the GB electricity market — it
underpins the day-ahead market and the system operator's planning — which is why
the app targets exactly that window.

The app frames the forecast as an interactive what-if tool: a temperature slider,
a weekday/weekend toggle, and a wind-generation control let anyone explore the
drivers of demand, while headline numbers are translated into human terms (peak
megawatts, and an approximate number of homes powered).

## Data sources

- **Electricity demand, embedded wind and solar generation** — NESO Historic
  Demand Data, used under the Open Government Licence v3.0.
- **Weather (temperature and wind speed)** — Open-Meteo historical weather API,
  combined as a population-weighted blend of London, Birmingham, Manchester and
  Glasgow to approximate a national signal.

## How it works

1. **Data pipeline.** Three years of half-hourly NESO demand data are loaded,
   audited for missing values, duplicate timestamps (autumn clock change) and
   gaps (spring clock change), cleaned, and resampled to hourly resolution.
2. **Feature engineering.** Weather is joined on, and calendar features are
   added: hour and month are encoded cyclically (sine/cosine) so the model
   understands their circular nature, alongside a weekday/weekend flag and
   recent-demand lag features.
3. **Leak-free preparation.** Data is split chronologically (70/15/15). Scalers
   are fitted on the training set only and applied to validation and test, to
   prevent data leakage. Sequences of 48 hours are built to predict the next 24.
4. **Models.** A naive seasonal baseline and a SARIMA model provide reference
   points; a stacked LSTM with dropout is the production model.

## Features the model uses

National demand history, embedded wind and solar generation, population-weighted
temperature and wind speed, and calendar signals (cyclically-encoded hour and
month, plus a weekday/weekend flag).

## Results

Evaluated on a held-out, most-recent test set (error in MW and MAPE):

| Model | MAE (MW) | RMSE (MW) | MAPE |
|-------|---------:|----------:|-----:|
| Naive (same hour yesterday) | 1,820 | 2,534 | 7.15% |
| SARIMA(1,0,1)(1,1,1,24) | 2,127 | 2,786 | 8.21% |
| **LSTM (48h to 24h, multivariate)** | **1,286** | **1,749** | **5.17%** |

The LSTM beats both reference models. SARIMA, being univariate, cannot see
weather and slightly underperforms the naive baseline — which is precisely the
motivation for a multivariate deep learning approach.

## Running locally

```bash
git clone https://github.com/<your-username>/gb-demand-forecaster.git
cd gb-demand-forecaster
pip install -r requirements.txt
streamlit run app.py
```

## Repository contents

- `app.py` — the Streamlit application
- `demand_lstm.keras` — the trained LSTM model
- `feat_scaler.joblib`, `target_scaler.joblib` — fitted scalers
- `feature_config.joblib` — feature names and model configuration
- `recent_window.csv` — the most recent 48-hour input window for live forecasts
- `requirements.txt` — pinned dependencies
- `.github/workflows/ci.yml` — continuous integration (lint and build check)

## Credits and licence

- Demand data: NESO, under the Open Government Licence v3.0.
- Weather data: Open-Meteo.
- Built by Nasir as an independent portfolio project.

This project is not affiliated with, or endorsed by, NESO. The forecasts are
illustrative and are not official NESO demand forecasts.