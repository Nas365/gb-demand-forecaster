"""
features.py — shared feature engineering for the GB Demand Forecaster.

CRITICAL: this is the single source of truth for how the 13 model features are
built from raw (datetime-indexed) demand + weather data. The notebook and the
app must both use this exact function, or the model will receive inputs that
differ from what it was trained on and forecasts will be silently wrong.

Expected raw input: a DataFrame with a DatetimeIndex (hourly) and columns:
    demand_mw, wind_mw, solar_mw, temp_c, wind_speed
Returns: a DataFrame with the 13 engineered features, in FEATURE_ORDER.
"""

import numpy as np
import pandas as pd

# The exact feature order the model was trained on.
CONTINUOUS = [
    "demand_mw", "wind_mw", "solar_mw",
    "temp_c", "wind_speed",
    "demand_lag_24", "demand_lag_168", "demand_roll_24",
]
BOUNDED = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend",
]
FEATURE_ORDER = CONTINUOUS + BOUNDED
TARGET = "demand_mw"


def build_features(df):
    """Build the 13 model features from raw hourly demand + weather data.

    df must have a DatetimeIndex and columns:
        demand_mw, wind_mw, solar_mw, temp_c, wind_speed
    """
    out = df.copy().sort_index()

    # Calendar
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(np.int8)
    hour = out.index.hour
    month = out.index.month

    # Cyclical encodings — identical formulas to the notebook
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    out["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    out["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)

    # Lag and rolling features
    out["demand_lag_24"] = out["demand_mw"].shift(24).astype(np.float32)
    out["demand_lag_168"] = out["demand_mw"].shift(168).astype(np.float32)
    out["demand_roll_24"] = out["demand_mw"].rolling(24).mean().astype(np.float32)

    return out[FEATURE_ORDER]
