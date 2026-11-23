"""Save, load and run trained models. Shared by train.py and the Forecast Agent.

A saved model is a folder with manifest.json plus its artifacts:
    naive    manifest only (forecast = current value)
    sklearn  manifest + model.joblib
    lstm     manifest + lstm.pt + lstm_meta.json

predict(folder, df) takes the feature table (sorted, hourly, contiguous) and
returns a Series indexed by `time`. Rain returns a probability, wind direction
returns degrees. LSTM rows without a full 24h window are simply absent.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ingestion.features import TARGET_SRC
from training import data

TARGETS = [
    "target_aqi", "target_temp", "target_precip", "target_humidity",
    "target_uv", "target_wind_speed", "target_wind_dir", "target_rain",
]

CLIP = {
    "target_precip": (0, None),
    "target_humidity": (0, 100),
    "target_uv": (0, None),
    "target_wind_speed": (0, None),
    "target_rain": (0, 1),
}


def save(folder, manifest, estimator=None):
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    if estimator is not None:
        joblib.dump(estimator, d / "model.joblib")
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))


def manifest(folder):
    return json.loads((Path(folder) / "manifest.json").read_text())


def predict(folder, df):
    d = Path(folder)
    m = manifest(d)
    tgt, kind = m["target"], m["kind"]
    t = df["time"]

    if kind == "naive":
        return pd.Series(df[TARGET_SRC[tgt]].values, index=t)

    if kind == "sklearn":
        est = joblib.load(d / "model.joblib")
        X = df[m["features"]]
        if tgt == "target_rain":
            out = est.predict_proba(X)[:, 1]
        elif tgt == "target_wind_dir":
            out = data.xy_to_wind_dir(est.predict(X))
        else:
            out = est.predict(X)
        return pd.Series(out, index=t)

    if kind == "lstm":
        import torch

        from training.lstm import Net

        meta = json.loads((d / "lstm_meta.json").read_text())
        feats, seq, reg = meta["features"], meta["seq"], meta["reg_targets"]
        mu = pd.Series(meta["x_mean"], index=feats)
        sd = pd.Series(meta["x_std"], index=feats)
        X = ((df[feats] - mu) / sd).values.astype(np.float32)
        W = sliding_window_view(X, seq, axis=0).transpose(0, 2, 1)

        net = Net(len(feats))
        net.load_state_dict(torch.load(d / "lstm.pt"))
        net.eval()
        with torch.no_grad():
            out = torch.cat(
                [net(torch.from_numpy(np.ascontiguousarray(W[i : i + 2048])))
                 for i in range(0, len(W), 2048)]
            ).numpy()
        raw = out[:, :-1] * np.array(meta["y_std"]) + np.array(meta["y_mean"])
        if tgt == "target_rain":
            v = 1 / (1 + np.exp(-out[:, -1]))
        elif tgt == "target_wind_dir":
            v = data.xy_to_wind_dir(raw[:, len(reg) : len(reg) + 2])
        else:
            v = raw[:, reg.index(tgt)]
        return pd.Series(v, index=t.iloc[seq - 1 :])

    raise ValueError(f"unknown model kind: {kind}")


def load_registry_models(project):
    mr = project.get_model_registry()
    folders = {}
    for tgt in TARGETS:
        models = mr.get_models("best_" + tgt.removeprefix("target_"))
        folders[tgt] = max(models, key=lambda m: m.version).download()
    return folders


def forecast(folders, df):
    df = df.sort_values("time").reset_index(drop=True)
    gaps = int(df["time"].diff().dropna().ne(pd.Timedelta(hours=1)).sum())
    if gaps:
        raise ValueError(f"{gaps} non-hourly gaps in feature rows")
    last = df["time"].iloc[-1]
    out = {}
    for tgt, folder in folders.items():
        v = float(predict(folder, df).loc[last])
        lo, hi = CLIP.get(tgt, (None, None))
        if lo is not None:
            v = max(v, lo)
        if hi is not None:
            v = min(v, hi)
        out[tgt] = v
    thr = manifest(folders["target_rain"])["rain_threshold"]
    return {
        "origin_time": last,
        "target_time": last + pd.Timedelta(hours=24),
        "forecast": out,
        "rain_expected": out["target_rain"] >= thr,
    }
