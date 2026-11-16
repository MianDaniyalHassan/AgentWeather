import numpy as np
import pandas as pd

BASE = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "us_aqi",
    "pm10",
    "pm2_5",
    "uv_index",
]

TARGET_SRC = {
    "target_aqi": "us_aqi",
    "target_temp": "temperature_2m",
    "target_precip": "precipitation",
    "target_humidity": "relative_humidity_2m",
    "target_uv": "uv_index",
    "target_wind_speed": "wind_speed_10m",
    "target_wind_dir": "wind_direction_10m",
}

ROLL_COLS = [
    "us_aqi",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
]

RAIN_THRESHOLD_MM = 0.1
MAX_GAP_HOURS = 6


def impute(df):
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.asfreq("h")
    df["n_imputed"] = df[BASE].isna().sum(axis=1)
    df[BASE] = df[BASE].interpolate(
        method="time", limit=MAX_GAP_HOURS, limit_direction="both"
    )
    return df.reset_index()


def build_features(df):
    df = df.copy()
    t = df["time"]
    df["hour"] = t.dt.hour
    df["day"] = t.dt.day
    df["month"] = t.dt.month
    df["dayofweek"] = t.dt.dayofweek

    rad = np.deg2rad(df["wind_direction_10m"])
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)

    for col in ROLL_COLS:
        df[f"{col}_diff_1h"] = df[col].diff(1)
        df[f"{col}_diff_24h"] = df[col].diff(24)
        df[f"{col}_roll_6h"] = df[col].rolling(6).mean()
        df[f"{col}_roll_24h"] = df[col].rolling(24).mean()

    for tgt, src in TARGET_SRC.items():
        df[tgt] = df[src].shift(-24)

    df["target_rain"] = np.where(
        df["target_precip"].isna(),
        np.nan,
        (df["target_precip"] >= RAIN_THRESHOLD_MM).astype(float),
    )
    return df


def feature_columns(df):
    return [c for c in df.columns if not c.startswith("target_")]
