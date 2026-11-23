import time
from functools import lru_cache

import hopsworks
from config import HOPSWORKS_KEY, HOPSWORKS_PROJECT

FG_VERSION = 3
PRECOMBINE = "ingested_at"


@lru_cache(maxsize=1)
def get_project():
    return hopsworks.login(api_key_value=HOPSWORKS_KEY, project=HOPSWORKS_PROJECT)


def get_feature_store():
    return get_project().get_feature_store()


def stamp(df):
    df = df.copy()
    df[PRECOMBINE] = int(time.time())
    return df


def get_feature_group(fs):
    return fs.get_or_create_feature_group(
        name="weather_features",
        version=FG_VERSION,
        primary_key=["time"],
        event_time="time",
        hudi_precombine_key=PRECOMBINE,
        time_travel_format="HUDI",
        stream=True,
        description="Hourly fused weather and AQI features with 24h-ahead targets",
    )


def get_raw_group(fs):
    return fs.get_or_create_feature_group(
        name="weather_raw",
        version=FG_VERSION,
        primary_key=["time"],
        event_time="time",
        hudi_precombine_key=PRECOMBINE,
        time_travel_format="HUDI",
        stream=True,
        description="Hourly raw weather and AQI observations",
    )


def get_aqicn_group(fs):
    return fs.get_or_create_feature_group(
        name="aqicn_live",
        version=1,
        primary_key=["time"],
        event_time="time",
        hudi_precombine_key=PRECOMBINE,
        time_travel_format="HUDI",
        stream=True,
        description="Hourly AQICN station AQI reading for validation and fallback",
    )


def _log_group(fs, name, primary_key, description):
    return fs.get_or_create_feature_group(
        name=name,
        version=1,
        primary_key=primary_key,
        event_time="time",
        hudi_precombine_key=PRECOMBINE,
        time_travel_format="HUDI",
        stream=True,
        description=description,
    )


def get_ledger_group(fs):
    return _log_group(fs, "forecast_ledger", ["time", "metric"],
                      "Every 24h-ahead forecast: origin time, target time, metric, value, model")


def get_agent_calls_group(fs):
    return _log_group(fs, "agent_calls", ["call_id"],
                      "Every LLM call: prompt, response, latency, status. Failed rows form the failure log")


def get_drift_group(fs):
    return _log_group(fs, "drift_log", ["time", "metric", "drift_type"],
                      "Drift events: metric, drift type, score, threshold, action taken")


def get_registry_group(fs):
    return _log_group(fs, "registry_log", ["time", "target"],
                      "Model registry decisions per training run")


def get_alerts_group(fs):
    return _log_group(fs, "alerts", ["time", "metric"],
                      "Hazard alerts per forecast origin with text for public, sensitive groups and outdoor workers")


def get_reports_group(fs):
    return _log_group(fs, "weather_reports", ["time"], "News-style weather reports")


def get_reference_group(fs):
    return _log_group(fs, "drift_reference", ["feature", "month"],
                      "Training-set feature distributions per calendar month, used for data drift")
