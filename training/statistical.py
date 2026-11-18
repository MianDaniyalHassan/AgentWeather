"""Statistical family: SARIMAX with a 24h season, one model per target.

Run from the repo root:
    python -m training.statistical
Uses the cached features from `python -m training.compare --refresh`.

How it works
- Parameters are fitted on the last FIT_HOURS of the train split only.
- The fitted model is then applied to the whole labeled series with the Kalman
  filter. The state at hour t only uses data up to t, so the 24h-ahead forecast
  for every origin is Z @ T^24 @ state_t. No refitting, no leakage.
- Wind direction is modeled as (sin, cos) and converted back to degrees.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ingestion.features import TARGET_SRC
from training import data, metrics

ORDER = (1, 0, 1)
SEASONAL = (1, 0, 0, 24)
FIT_HOURS = 24 * 60
HORIZON = 24
OUT = Path(__file__).parent / "results"

CLIP = {
    "target_precip": (0, None),
    "target_humidity": (0, 100),
    "target_uv": (0, None),
    "target_wind_speed": (0, None),
}


def _model(y):
    return SARIMAX(
        y,
        order=ORDER,
        seasonal_order=SEASONAL,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )


def forecast_series(series, n_train):
    """24h-ahead forecast for every origin in `series` (numpy array).

    Fits on the tail of series[:n_train], filters over the whole series.
    """
    mu = series[:n_train].mean()
    y = series - mu
    fit_start = max(0, n_train - FIT_HOURS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = _model(y[fit_start:n_train]).fit(disp=False, maxiter=100).params
        mod = _model(y)
        res = mod.smooth(params)
    T = mod.ssm["transition", :, :, 0]
    Z = mod.ssm["design", :, :, 0]
    state = res.filter_results.filtered_state  # (k_states, nobs)
    pred = (Z @ np.linalg.matrix_power(T, HORIZON) @ state).ravel()
    return pred + mu


def main():
    df = data.load()
    train, val, test = data.split(df)
    tcols = [c for c in df.columns if c.startswith("target_")]
    lab = df.dropna(subset=tcols).reset_index(drop=True)
    n_train = int(lab["time"].isin(train["time"]).sum())
    print(f"labeled rows {len(lab)} | train rows {n_train}")

    rows = []
    for tgt, src in TARGET_SRC.items():
        print(f"fitting sarimax -> {tgt}")
        if tgt == "target_wind_dir":
            rad = np.deg2rad(lab[src].values.astype(float))
            s = forecast_series(np.sin(rad), n_train)
            c = forecast_series(np.cos(rad), n_train)
            pred = np.rad2deg(np.arctan2(s, c)) % 360
            score = metrics.wind_dir
        else:
            pred = forecast_series(lab[src].values.astype(float), n_train)
            lo, hi = CLIP.get(tgt, (None, None))
            pred = np.clip(pred, lo, hi)
            score = metrics.regression
        pred = pd.Series(pred, index=lab["time"])
        for name, part in (("val", val), ("test", test)):
            rows.append(
                {"target": tgt, "model": "sarimax", "split": name,
                 **score(part[tgt].values, pred.loc[part["time"]].values)}
            )

    out = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "statistical_regression.csv", index=False)
    pd.set_option("display.width", 200)
    for name in ("val", "test"):
        print(f"\n=== SARIMAX regression, {name} ===")
        print(out[out["split"] == name].set_index("target")[["rmse", "mae", "r2"]].round(3).to_string())


if __name__ == "__main__":
    main()
