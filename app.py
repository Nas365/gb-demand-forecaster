"""
GB Electricity Demand Forecaster
A 24-hour-ahead national demand forecasting app powered by an LSTM trained
on NESO open data. Fetches the most recent data live, with a bundled
fallback window if the live sources are unavailable.

Built by Nasir.
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import streamlit as st

from features import FEATURE_ORDER, CONTINUOUS

st.set_page_config(page_title="GB Electricity Demand Forecaster", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1f3a 0%, #14365c 45%, #1d4e6f 100%);
        color: #f0f4f8;
    }
    .block-container { padding-top: 2.2rem; }
    h1, h2, h3, h4 { color: #ffffff; }
    .credit-band {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px; padding: 14px 18px; margin-bottom: 18px;
        font-size: 0.92rem; line-height: 1.5;
    }
    .metric-card {
        background: rgba(255,255,255,0.08);
        border-radius: 12px; padding: 16px; text-align: center;
    }
    .stSlider label, .stSelectbox label { color: #dbe6f0 !important; }
    a { color: #7fc4ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artefacts():
    model = tf.keras.models.load_model("demand_lstm.keras")
    feat_scaler = joblib.load("feat_scaler.joblib")
    target_scaler = joblib.load("target_scaler.joblib")
    config = joblib.load("feature_config.joblib")
    return model, feat_scaler, target_scaler, config


@st.cache_data(ttl=3600)
def get_window():
    try:
        from live_data import fetch_live_window
        window, last_time = fetch_live_window()
        return window, last_time, "live"
    except Exception as exc:
        df = pd.read_csv("recent_window.csv", index_col=0, parse_dates=True)
        return df, df.index[-1], f"fallback ({type(exc).__name__})"


model, feat_scaler, target_scaler, config = load_artefacts()
WINDOW = config["WINDOW"]
HORIZON = config["HORIZON"]

st.title("GB Electricity Demand Forecaster")
st.markdown(
    "<div class='credit-band'>"
    "An interactive 24-hour-ahead forecast of Great Britain's national "
    "electricity demand, built with a multivariate LSTM neural network. The app "
    "pulls the most recent demand and weather data and predicts the next 24 hours. "
    "Use the controls to explore how temperature, day type and wind generation "
    "shift demand."
    "<br><br>"
    "<b>Built by Nasir.</b> An independent portfolio project. Not affiliated with "
    "or endorsed by NESO. Forecasts are illustrative and are not official NESO "
    "demand forecasts."
    "</div>",
    unsafe_allow_html=True,
)

window_df, last_time, source = get_window()
forecast_start = last_time + pd.Timedelta(hours=1)
forecast_end = last_time + pd.Timedelta(hours=HORIZON)

if source == "live":
    st.success(
        f"Using live NESO data up to {last_time:%H:%M on %d %b %Y} (UTC). "
        f"Forecasting demand for the next 24 hours — "
        f"{forecast_start:%d %b}, midnight through 23:00 UTC."
    )
else:
    st.info(
        f"Live data unavailable - using bundled recent window through "
        f"{last_time:%Y-%m-%d %H:%M}. Forecasting the following 24 hours. ({source})"
    )

st.subheader("Scenario controls")
st.caption(
    "Explore what-if scenarios. These controls adjust the recent conditions the "
    "model sees, so you can test how demand would change if it were colder, "
    "warmer, a weekend, or windier than reality."
)
c1, c2, c3 = st.columns(3)
with c1:
    temp_shift = st.slider(
        "Temperature: warmer / colder than forecast (°C)",
        -25.0, 25.0, 0.0, 0.5,
        help="The model uses real recent temperature. Drag left to simulate a "
             "colder spell, right for a warmer one — then watch how predicted "
             "demand responds. Large shifts explore extremes the model saw "
             "rarely in training, so treat those as directional."
    )
with c2:
    day_type = st.selectbox("Day type", ["As recorded", "Force weekday", "Force weekend"])
with c3:
    wind_shift = st.slider("Wind generation adjustment (%)", -50, 50, 0, 5)


def build_forecast(w, temp_shift, wind_shift, day_type):
    w = w.copy()
    if "temp_c" in w.columns:
        w["temp_c"] = w["temp_c"] + temp_shift
    if "wind_mw" in w.columns:
        w["wind_mw"] = w["wind_mw"] * (1 + wind_shift / 100.0)
    if day_type == "Force weekday":
        w["is_weekend"] = 0
    elif day_type == "Force weekend":
        w["is_weekend"] = 1

    scaled = w[FEATURE_ORDER].copy()
    scaled[CONTINUOUS] = feat_scaler.transform(w[CONTINUOUS])
    X = scaled[FEATURE_ORDER].values.astype(np.float32).reshape(1, WINDOW, len(FEATURE_ORDER))
    pred_scaled = model.predict(X, verbose=0)
    return target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()


forecast = build_forecast(window_df, temp_shift, wind_shift, day_type)

peak, trough, avg = float(np.max(forecast)), float(np.min(forecast)), float(np.mean(forecast))
homes = peak * 2000

st.subheader("Next 24 hours")
m1, m2, m3 = st.columns(3)
m1.markdown(f"<div class='metric-card'><h3>{peak:,.0f} MW</h3>Peak demand</div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card'><h3>{avg:,.0f} MW</h3>Average demand</div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card'><h3>{homes/1e6:,.1f} M</h3>Homes at peak (approx.)</div>", unsafe_allow_html=True)

times = pd.date_range(forecast_start, periods=HORIZON, freq="h")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=times, y=forecast, mode="lines+markers",
    line=dict(color="#7fc4ff", width=3), name="Forecast demand",
))
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Time (UTC)", yaxis_title="Demand (MW)",
    height=420, margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("About this project - data, method, and how it works"):
    st.markdown(
        """
**What it is.** A 24-hour-ahead forecast of Great Britain's national electricity
demand. Given the last 48 hours of demand, weather and calendar context, a
trained LSTM predicts demand for each of the next 24 hours.

**Live data.** On load, the app fetches the most recent demand from NESO's Demand
Data Update feed and recent weather from Open-Meteo, rebuilds the model features,
and forecasts forward. If a live source is unavailable, it falls back to a bundled
recent window so the app always works.

**Why 24 hours.** Day-ahead forecasting is the central horizon in the GB
electricity market.

**Data sources.**
- Electricity demand, embedded wind and solar: NESO, Open Government Licence v3.0.
- Weather: Open-Meteo, population-weighted across London, Birmingham, Manchester,
  Glasgow.

**Models.** A naive seasonal baseline and a SARIMA model are reference points;
the LSTM is the production model, with the lowest error of the three on a
held-out, most-recent test set.

**Important.** Independent portfolio project. Not affiliated with or endorsed by
NESO; forecasts are illustrative, not official.

Built by Nasir.
        """
    )