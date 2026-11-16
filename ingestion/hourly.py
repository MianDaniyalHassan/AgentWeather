import argparse

import pandas as pd

from ingestion.aqicn import fetch_current, local_hour
from ingestion.features import build_features, feature_columns, impute
from ingestion.openmeteo import fetch_recent
from ingestion.store import (
    get_aqicn_group,
    get_feature_group,
    get_feature_store,
    get_raw_group,
    stamp,
)


def build_recent(past_days):
    raw = fetch_recent(past_days)
    raw = raw[raw["time"] <= pd.Timestamp(local_hour())]
    raw = raw.drop_duplicates(subset="time", keep="last").sort_values("time")
    df = build_features(impute(raw))
    df = df.dropna(subset=feature_columns(df)).reset_index(drop=True)
    return raw.reset_index(drop=True), df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--past-days", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    raw, df = build_recent(a.past_days)

    aq = None
    try:
        aq = fetch_current()
    except Exception as e:
        print(f"aqicn failed: {e}")

    print(f"raw rows: {len(raw)} ({raw['time'].min()} -> {raw['time'].max()})")
    print(f"feature rows: {len(df)} ({df['time'].min()} -> {df['time'].max()})")
    print(f"rows without target: {df['target_aqi'].isna().sum()}")
    print(f"max n_imputed: {df['n_imputed'].max()}")
    if aq is not None:
        print(aq.T)

    if a.dry_run:
        return

    fs = get_feature_store()
    get_feature_group(fs).insert(stamp(df), write_options={"wait_for_job": True})
    get_raw_group(fs).insert(stamp(raw), write_options={"wait_for_job": False})
    if aq is not None:
        get_aqicn_group(fs).insert(stamp(aq), write_options={"wait_for_job": False})
    print("inserted")


if __name__ == "__main__":
    main()