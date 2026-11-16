import argparse
from datetime import date, timedelta

import pandas as pd

from ingestion.features import build_features, feature_columns, impute
from ingestion.openmeteo import fetch_range
from ingestion.store import get_feature_group, get_feature_store, get_raw_group, stamp


def chunks(start, end, days=90):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur.isoformat(), nxt.isoformat()
        cur = nxt + timedelta(days=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=(date.today() - timedelta(days=7)).isoformat())
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    start = date.fromisoformat(a.start)
    end = date.fromisoformat(a.end)

    frames = []
    for s, e in chunks(start, end):
        print(f"fetching {s} -> {e}")
        frames.append(fetch_range(s, e))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset="time", keep="last").sort_values("time")

    df = build_features(impute(raw))
    feats = feature_columns(df)
    before = len(df)
    df = df.dropna(subset=feats).reset_index(drop=True)

    print(f"raw rows: {len(raw)}")
    print(f"feature rows kept: {len(df)} (dropped {before - len(df)})")
    print(f"rows without target (forecast-only): {df['target_aqi'].isna().sum()}")
    print(f"columns: {df.shape[1]}")
    print(df.isna().sum()[lambda s: s > 0])

    if a.dry_run:
        return

    fs = get_feature_store()
    get_raw_group(fs).insert(stamp(raw), write_options={"wait_for_job": True})
    get_feature_group(fs).insert(stamp(df), write_options={"wait_for_job": True})
    print("inserted")


if __name__ == "__main__":
    main()