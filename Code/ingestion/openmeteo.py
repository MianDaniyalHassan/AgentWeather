import pandas as pd
import requests
from config import LAT, LON, TIMEZONE

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]
AQ_VARS = ["us_aqi", "pm10", "pm2_5", "uv_index"]


def _fetch(url, hourly_vars, extra):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(hourly_vars),
        "timezone": TIMEZONE,
        **extra,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def fetch_range(start, end):
    extra = {"start_date": start, "end_date": end}
    weather = _fetch(ARCHIVE_URL, WEATHER_VARS, extra)
    air = _fetch(AQ_URL, AQ_VARS, extra)
    return weather.join(air, how="outer").reset_index()


def fetch_recent(past_days=3):
    extra = {"past_days": past_days, "forecast_days": 1}
    weather = _fetch(FORECAST_URL, WEATHER_VARS, extra)
    air = _fetch(AQ_URL, AQ_VARS, extra)
    return weather.join(air, how="outer").reset_index()


if __name__ == "__main__":
    df = fetch_range("2026-11-01", "2026-11-07")
    print(df.shape)
    print(df.isna().sum())
    print(df.head())
