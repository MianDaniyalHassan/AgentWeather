"""Reference distributions for data drift (brief section 6, Drift Agent).

Saved at train time to the `drift_reference` feature group. For each monitored
feature and calendar month it holds the share of training rows in each bin.
Bin edges are quantiles of the whole training split, so every month uses the
same bins. The Drift Agent compares live features with the months they fall in,
so normal seasonal change does not count as drift.

    python -m training.reference     # build from the cached features and write
"""
import json

import numpy as np
import pandas as pd

from ingestion.features import BASE

BINS = 5
EPS = 0.005


def proportions(x, edges):
    idx = np.searchsorted(edges, np.asarray(x, dtype=float), side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def psi(ref, live):
    ref = np.clip(ref, EPS, None)
    live = np.clip(live, EPS, None)
    return float(np.sum((live - ref) * np.log(live / ref)))


def build(train):
    rows = []
    months = train["time"].dt.month
    for col in BASE:
        qs = np.linspace(0, 1, BINS + 1)[1:-1]
        edges = np.unique(np.quantile(train[col].dropna(), qs))
        for month, g in train.groupby(months):
            rows.append(
                {
                    "feature": col,
                    "month": int(month),
                    "edges": json.dumps(edges.tolist()),
                    "props": json.dumps(proportions(g[col].dropna(), edges).tolist()),
                    "n": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def write(train):
    from ingestion.store import get_feature_store, get_reference_group, insert_retry

    ref = build(train).assign(time=pd.Timestamp.now())
    insert_retry(get_reference_group(get_feature_store()), ref)
    return ref


def main():
    from training import data

    train, _, _ = data.split(data.load())
    ref = write(train)
    print(f"wrote {len(ref)} reference rows "
          f"({ref['feature'].nunique()} features x {ref['month'].nunique()} months)")


if __name__ == "__main__":
    main()
