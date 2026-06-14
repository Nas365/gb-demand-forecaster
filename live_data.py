"""
live_data.py — fetch the most recent data needed to forecast.

Pulls the latest ~48+ hours of demand (NESO) and weather (Open-Meteo), builds
features via the shared features.build_features, and returns the final 48-hour
window the model expects. Designed to fail safely: any error is raised to the
caller, which falls back to the bundled recent_window.csv.

Data sources:
  - NESO "Demand Data Update" (Open Government Licence v3.0)
  - Open-Meteo historical/forecast weather API
"""

import numpy as np
import pandas as pd
import requests

from features import build_features, FEATURE_ORDER

# NESO "Demand Data Update": previous month-start to today, half-hourly.
NESO_RESOURCE = "177f6fa4-ae49-4182-81ea-0c6b35f26ca6"
NESO_URL = "https://api.neso.energy/api/3/action/datastore_search"

# Population-weighted GB demand centres (same as the notebook).
CITIES = {
    "London":     {"lat": 51.51, "lon": -0.13, "weight": 0.40},
    "Birmingham": {"lat": 52.48, "lon": -1.90, "weight": 0.25},
    "Manchester": {"lat": 53.48, "lon": -2.24, "weight": 0.20},
    "Glasgow":    {"lat": 55.86, "lon": -4.25, "weight": 0.15},
}

WINDOW = 48


def _fetch_neso_demand():
    """Return recent half-hourly demand, wind, solar from NESO as hourly data."""
    params = {"resource_id": NESO_RESOURCE, "limit": 5000}
    r = requests.get(NESO_URL, params=params, timeout=30)
    r.raise_for_status()
    records = r.json()["result"]["records"]
    df = pd.DataFrame(records)

    # Build datetime from settlement date + period (period 1 -> 00:00).
    df["SETTLEMENT_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"])
    df["SETTLEMENT_PERIOD"] = pd.to_numeric(df["SETTLEMENT_PERIOD"], errors="coerce")
    df["datetime"] = df["SETTLEMENT_DATE"] + pd.to_timedelta(
        (df["SETTLEMENT_PERIOD"] - 1) * 30, unit="m"
    )

    rename = {
        "ND": "demand_mw",
        "EMBEDDED_WIND_GENERATION": "wind_mw",
        "EMBEDDED_SOLAR_GENERATION": "solar_mw",
    }
    df = df.rename(columns=rename)
    for c in ["demand_mw", "wind_mw", "solar_mw"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (df[["datetime", "demand_mw", "wind_mw", "solar_mw"]]
          .dropna(subset=["demand_mw"])
          .set_index("datetime")
          .sort_index())
    df = df[~df.index.duplicated(keep="first")]

    # Resample to hourly to match training.
    return df.resample("1h").mean()


def _fetch_weather():
    """Population-weighted hourly temperature and wind speed for GB.

    Uses the forecast API with past_days=10 so data is always current
    (the archive endpoint has a ~5-day lag and causes HTTPErrors for recent dates).
    """
    temp_w, wind_w = None, None
    for c in CITIES.values():
        params = {
            "latitude": c["lat"], "longitude": c["lon"],
            "hourly": "temperature_2m,wind_speed_10m",
            "past_days": 10,
            "forecast_days": 1,
            "timezone": "UTC",
        }
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params, timeout=30,
        )
        resp.raise_for_status()
        h = resp.json()["hourly"]
        idx = pd.to_datetime(h["time"])
        t = pd.Series(h["temperature_2m"], index=idx) * c["weight"]
        wd = pd.Series(h["wind_speed_10m"], index=idx) * c["weight"]
        temp_w = t if temp_w is None else temp_w.add(t, fill_value=0)
        wind_w = wd if wind_w is None else wind_w.add(wd, fill_value=0)
    return pd.DataFrame({"temp_c": temp_w, "wind_speed": wind_w})


def fetch_live_window():
    """Return the most recent 48-hour feature window (DataFrame, FEATURE_ORDER).

    Raises on any failure so the app can fall back to the bundled CSV.
    """
    demand = _fetch_neso_demand()
    if len(demand) < 200:  # need history for the 168h lag
        raise ValueError("Insufficient NESO history returned for lag features.")

    weather = _fetch_weather()

    merged = demand.join(weather, how="left")
    merged[["temp_c", "wind_speed"]] = (
        merged[["temp_c", "wind_speed"]].interpolate("time").ffill().bfill()
    )
    merged = merged.dropna(subset=["demand_mw", "wind_mw", "solar_mw"])

    feats = build_features(merged).dropna()
    if len(feats) < WINDOW:
        raise ValueError("Not enough rows after feature build for a 48h window.")

    window = feats.iloc[-WINDOW:]
    last_time = window.index[-1]
    return window, last_time
