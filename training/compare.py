"""Naive baseline vs Ridge vs Random Forest for every target.

Run from the repo root:
    python -m training.compare            # uses cached features if present
    python -m training.compare --refresh  # re-read from Hopsworks
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ingestion.features import RAIN_THRESHOLD_MM, TARGET_SRC
from training import data, metrics

OUT = Path(__file__).parent / "results"


def regressors():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=0
        ),
    }


def classifiers():
    return {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=0,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    df = data.load(refresh=a.refresh)
    train, val, test = data.split(df)
    feats = data.feature_cols(df)
    parts = {"val": val, "test": test}
    print(f"features: {len(feats)}")
    print(f"train {len(train)} ({train['time'].min()} -> {train['time'].max()})")
    print(f"val   {len(val)} ({val['time'].min()} -> {val['time'].max()})")
    print(f"test  {len(test)} ({test['time'].min()} -> {test['time'].max()})")

    reg_rows = []
    for tgt, src in TARGET_SRC.items():
        is_wd = tgt == "target_wind_dir"
        score = metrics.wind_dir if is_wd else metrics.regression

        for sp, part in parts.items():
            reg_rows.append(
                {"target": tgt, "model": "naive", "split": sp,
                 **score(part[tgt].values, part[src].values)}
            )

        y_train = data.wind_dir_xy(train[tgt]) if is_wd else train[tgt].values
        for name, model in regressors().items():
            print(f"fitting {name} -> {tgt}")
            model.fit(train[feats], y_train)
            for sp, part in parts.items():
                p = model.predict(part[feats])
                if is_wd:
                    p = data.xy_to_wind_dir(p)
                reg_rows.append(
                    {"target": tgt, "model": name, "split": sp,
                     **score(part[tgt].values, p)}
                )

    rain_rows = []
    for sp, part in parts.items():
        naive = (part["precipitation"] >= RAIN_THRESHOLD_MM).astype(int)
        rain_rows.append(
            {"target": "target_rain", "model": "naive", "split": sp,
             **metrics.rain(part["target_rain"].astype(int), naive)}
        )
    y_train = train["target_rain"].astype(int)
    for name, model in classifiers().items():
        print(f"fitting {name} -> target_rain")
        model.fit(train[feats], y_train)
        yv = val["target_rain"].astype(int)
        val_proba = model.predict_proba(val[feats])[:, 1]
        grid = [i / 100 for i in range(5, 60, 5)]
        best_t = max(
            grid, key=lambda t: metrics.rain(yv, (val_proba >= t).astype(int))["f1"]
        )
        for sp, part in parts.items():
            proba = model.predict_proba(part[feats])[:, 1]
            rain_rows.append(
                {"target": "target_rain", "model": name, "split": sp,
                 "threshold": best_t,
                 **metrics.rain(part["target_rain"].astype(int),
                                (proba >= best_t).astype(int), proba)}
            )

    reg = pd.DataFrame(reg_rows)
    rn = pd.DataFrame(rain_rows)
    OUT.mkdir(exist_ok=True)
    reg.to_csv(OUT / "classical_regression.csv", index=False)
    rn.to_csv(OUT / "classical_rain.csv", index=False)

    pd.set_option("display.width", 200)
    for sp in ("val", "test"):
        print(f"\n=== regression, {sp} ===")
        t = reg[reg["split"] == sp].pivot(index="target", columns="model", values="rmse")
        print(t.round(3).to_string())
    print("\n=== rain (val + test) ===")
    print(rn.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
