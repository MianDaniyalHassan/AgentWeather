from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).parent / "cache" / "features.parquet"

NON_FEATURES = {"time", "day", "ingested_at"}
TEST_FRAC = 0.2   # same test window as the EDA naive baseline
VAL_FRAC = 0.1
GAP = 24          # hours dropped at each boundary so targets never overlap the next split


def load(refresh=False):
    """Read weather_features from Hopsworks (cached locally as parquet)."""
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    from ingestion.store import get_feature_group, get_feature_store

    df = get_feature_group(get_feature_store()).read()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    CACHE.parent.mkdir(exist_ok=True)
    df.to_parquet(CACHE)
    return df


def feature_cols(df):
    return [
        c for c in df.columns
        if not c.startswith("target_") and c not in NON_FEATURES
    ]


def split(df):
    """Time-based train / val / test on rows that have every target."""
    target_cols = [c for c in df.columns if c.startswith("target_")]
    lab = df.dropna(subset=target_cols).reset_index(drop=True)
    n = len(lab)
    test_start = int(n * (1 - TEST_FRAC))
    val_start = int(n * (1 - TEST_FRAC - VAL_FRAC))
    train = lab.iloc[: val_start - GAP]
    val = lab.iloc[val_start : test_start - GAP]
    test = lab.iloc[test_start:]
    return train, val, test


def wind_dir_xy(deg):
    """Wind direction as (sin, cos) so models don't see a 359 -> 0 jump."""
    rad = np.deg2rad(np.asarray(deg, dtype=float))
    return np.column_stack([np.sin(rad), np.cos(rad)])


def xy_to_wind_dir(xy):
    return np.rad2deg(np.arctan2(xy[:, 0], xy[:, 1])) % 360
