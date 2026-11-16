import time

import hopsworks
from config import HOPSWORKS_KEY, HOPSWORKS_PROJECT

FG_VERSION = 3
PRECOMBINE = "ingested_at"


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