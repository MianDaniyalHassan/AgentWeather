"""Drift Agent (brief section 6): runs hourly after the Forecast and Alert agents.

Data drift   PSI of the last 14 days of each base feature against the training
             split, using the calendar months the live window falls in.
Model drift  recent forecast error (ledger vs actuals) against the validation
             error stored with the registered model.

Drift is confirmed when a model metric crosses its threshold, or when at least
PSI_MIN_GROUPS groups of related features cross the PSI threshold (AQI, PM10 and
PM2.5 move together, so they count once). Then the agent alerts the team and
dispatches the training workflow, at most once per COOLDOWN_H.
Every check is one row in drift_log.

    python -m agents.drift --backtest   # how often would data drift fire on history?
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import brier_score_loss

from ingestion.features import BASE
from training import metrics, reference

DATA_WINDOW_H = 24 * 14
MODEL_WINDOW_H = 24 * 7
MIN_ORIGINS = 48          # scored forecasts needed before model drift is judged
# Provisional. Hourly data is strongly autocorrelated, so 14 days is only ~14
# independent days and PSI is biased upward (about bins-1 over that count). The
# textbook 0.25 fired on 60% of 2026 windows, so these are set from --backtest.
PSI_THRESHOLD = 0.5
PSI_MIN_GROUPS = 2
GROUP = {
    "us_aqi": "air", "pm10": "air", "pm2_5": "air",
    "temperature_2m": "thermal", "relative_humidity_2m": "thermal", "uv_index": "thermal",
    "wind_speed_10m": "wind", "wind_direction_10m": "wind",
    "precipitation": "rain",
}
# recent error / validation error. Precipitation and rain sit higher because
# their test error was already 1.5x and 2.2x the validation error (season shift).
RATIO = {"default": 1.5, "precip": 2.0, "rain": 3.0}
COOLDOWN_H = 24
NOTIFIED = {"retrain_triggered", "dispatch_failed", "alert_only"}
METRICS = ["aqi", "temp", "precip", "humidity", "uv", "wind_speed", "wind_dir", "rain"]
WORKFLOW = "train.yml"


def check(metric, drift_type, score, threshold):
    return {
        "metric": metric,
        "drift_type": drift_type,
        "score": float(score),
        "threshold": float(threshold),
        "action": "none",
    }


def data_drift(history, ref):
    live = history.tail(DATA_WINDOW_H)
    share = live["time"].dt.month.value_counts(normalize=True)
    rows = []
    for col in BASE:
        r = ref[ref["feature"] == col].set_index("month")
        months = [m for m in share.index if m in r.index]
        if not months:
            continue
        w = np.array([share[m] for m in months])
        w = w / w.sum()
        edges = np.array(json.loads(r.loc[months[0], "edges"]))
        ref_p = sum(wi * np.array(json.loads(r.loc[m, "props"])) for wi, m in zip(w, months))
        live_p = reference.proportions(live[col].dropna(), edges)
        rows.append(check(col, "data", reference.psi(ref_p, live_p), PSI_THRESHOLD))
    return rows


def model_drift(history, ledger, val_err):
    since = history["time"].max() - pd.Timedelta(hours=MODEL_WINDOW_H)
    led = ledger[ledger["time"] > since]
    rows, short = [], []
    for m, ref_err in val_err.items():
        col = f"target_{m}"
        d = led[led["metric"] == m].merge(history[["time", col]], on="time").dropna(subset=[col])
        if len(d) < MIN_ORIGINS:
            short.append(f"{m}:{len(d)}")
            continue
        y, p = d[col].to_numpy(float), d["forecast"].to_numpy(float)
        if m == "wind_dir":
            err = metrics.wind_dir(y, p)["rmse"]
        elif m == "rain":
            err = brier_score_loss(y.astype(int), p)
        else:
            err = float(np.sqrt(np.mean((y - p) ** 2)))
        rows.append(check(m, "model", err / max(ref_err, 1e-9), RATIO.get(m, RATIO["default"])))
    if short:
        print(f"model drift waiting for {MIN_ORIGINS} scored forecasts ({', '.join(short)})")
    return rows


def validation_errors(project):
    """Validation error stored with the current registered model of each target."""
    mr = project.get_model_registry()
    out = {}
    for m in METRICS:
        best = max(mr.get_models(f"best_{m}"), key=lambda x: x.version)
        out[m] = float(best.training_metrics["val_brier" if m == "rain" else "val_rmse"])
    return out


def recently_triggered(fs, now):
    from ingestion.store import get_drift_group

    try:
        df = get_drift_group(fs).read()
    except Exception:
        return False
    if df.empty:
        return False
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    recent = df[df["action"].isin(NOTIFIED) & (df["time"] > now - pd.Timedelta(hours=COOLDOWN_H))]
    return not recent.empty


def notify(text):
    print(f"DRIFT ALERT: {text}")
    url = os.getenv("ALERT_WEBHOOK_URL")
    if url:
        try:
            requests.post(url, json={"text": text, "content": text}, timeout=15)
        except Exception as e:
            print(f"webhook failed: {e}")


def dispatch():
    token = os.getenv("GH_TOKEN")
    if not token:
        return "alert_only"
    repo = os.getenv("GITHUB_REPOSITORY", "MianDaniyalHassan/AgentWeather")
    try:
        r = requests.post(
            f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json={"ref": os.getenv("GITHUB_REF_NAME", "main")},
            timeout=30,
        )
        r.raise_for_status()
        return "retrain_triggered"
    except Exception as e:
        print(f"dispatch failed: {e}")
        return "dispatch_failed"


def apply_actions(rows, fs, now):
    over = [r for r in rows if r["score"] >= r["threshold"]]
    data_over = [r for r in over if r["drift_type"] == "data"]
    confirmed = [r for r in over if r["drift_type"] == "model"]
    if len({GROUP[r["metric"]] for r in data_over}) >= PSI_MIN_GROUPS:
        confirmed += data_over
    else:
        for r in data_over:
            r["action"] = "watch"
    if not confirmed:
        return "none"
    if recently_triggered(fs, now):
        action = "cooldown"
    else:
        action = dispatch()
        notify("; ".join(
            f"{r['drift_type']} drift on {r['metric']}: {r['score']:.2f} (threshold {r['threshold']})"
            for r in confirmed
        ) + f". Action: {action}.")
    for r in confirmed:
        r["action"] = action
    return action


def run(history, project):
    from ingestion.aqicn import local_hour
    from ingestion.store import get_ledger_group, get_reference_group

    fs = project.get_feature_store()
    rows = []
    try:
        rows += data_drift(history, get_reference_group(fs).read())
    except Exception as e:
        print(f"data drift skipped (run `python -m training.reference` first?): {e}")
    try:
        ledger = get_ledger_group(fs).read()
        ledger["time"] = pd.to_datetime(ledger["time"]).dt.tz_localize(None)
        rows += model_drift(history, ledger, validation_errors(project))
    except Exception as e:
        print(f"model drift skipped: {e}")
    return rows, apply_actions(rows, fs, pd.Timestamp(local_hour()))


def windows(df, ref, step=24):
    """PSI per feature for every 14-day window of df, one window per `step` hours."""
    out = []
    for end in range(DATA_WINDOW_H, len(df) + 1, step):
        win = df.iloc[end - DATA_WINDOW_H : end]
        out.append({"end": win["time"].iloc[-1], **{r["metric"]: r["score"] for r in data_drift(win, ref)}})
    return pd.DataFrame(out).set_index("end")


def groups_over(bt, thr):
    return sum((bt[[c for c in bt if GROUP[c] == g]] >= thr).any(axis=1) for g in set(GROUP.values()))


def report(bt, label):
    thrs = [0.25, 0.5, 0.75, 1.0]
    print(f"\n=== {label}: {len(bt)} windows ===")
    print(f"share of windows over PSI {PSI_THRESHOLD}, per feature:")
    print((bt >= PSI_THRESHOLD).mean().round(3).to_string())
    table = {
        f"PSI >= {t}": {f">= {k} groups": f"{(groups_over(bt, t) >= k).mean():.1%}" for k in (2, 3)}
        for t in thrs
    }
    print("\nshare of windows where drift would fire:")
    print(pd.DataFrame(table).to_string())


def backtest():
    """Calibration. 2026 windows against the training reference (what production sees),
    and a null test: 2025 windows against a 2024-only reference, both inside the
    training split, where no real change was injected."""
    from training import data

    train, val, test = data.split(data.load())
    live = windows(pd.concat([val, test]).reset_index(drop=True), reference.build(train))
    report(live, "validation + test windows vs training reference")
    year = train["time"].dt.year
    if (year == 2025).any():
        null = windows(train[year == 2025].reset_index(drop=True), reference.build(train[year == 2024]))
        report(null, "null test: 2025 windows vs 2024 reference")
    path = Path(__file__).resolve().parent.parent / "training" / "results" / "drift_backtest.csv"
    path.parent.mkdir(exist_ok=True)
    live.round(4).to_csv(path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args()
    backtest() if a.backtest else ap.print_help()