"""LSTM (deep learning family): one network, 24h window of features -> all targets.

Run from the repo root:
    python -m training.lstm
Uses the cached features from `python -m training.compare --refresh`.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from ingestion.features import TARGET_SRC
from training import data, metrics

SEQ = 24
HIDDEN = 64
BATCH = 256
MAX_EPOCHS = 40
PATIENCE = 6
REG = [t for t in TARGET_SRC if t != "target_wind_dir"]  # 6 plain regression targets
OUT = Path(__file__).parent / "results"
MODELS = Path(__file__).parent / "models"


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.lstm = nn.LSTM(n_in, HIDDEN, batch_first=True)
        self.drop = nn.Dropout(0.2)
        # 6 regression + wind dir (sin, cos) + 1 rain logit
        self.head = nn.Linear(HIDDEN, len(REG) + 2 + 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1]))


def make_targets(lab):
    reg = lab[REG].values.astype(np.float32)
    wd = data.wind_dir_xy(lab["target_wind_dir"]).astype(np.float32)
    return np.hstack([reg, wd]), lab["target_rain"].values.astype(np.float32)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    df = data.load()
    train, val, test = data.split(df)
    feats = data.feature_cols(df)

    tcols = [c for c in df.columns if c.startswith("target_")]
    lab = df.dropna(subset=tcols).reset_index(drop=True)
    gaps = int(lab["time"].diff().dropna().ne(pd.Timedelta(hours=1)).sum())
    print(f"labeled rows: {len(lab)} | non-hourly gaps: {gaps}")

    # scale features with train statistics only
    mu = train[feats].mean()
    sd = train[feats].std().replace(0, 1)
    X = ((lab[feats] - mu) / sd).values.astype(np.float32)

    # window i ends at row i+SEQ-1, so drop the first SEQ-1 rows of lab
    W = sliding_window_view(X, SEQ, axis=0).transpose(0, 2, 1)
    lab = lab.iloc[SEQ - 1 :].reset_index(drop=True)
    y, rain = make_targets(lab)

    idx = {
        n: lab["time"].isin(p["time"]).values
        for n, p in (("train", train), ("val", val), ("test", test))
    }
    y_mu = y[idx["train"]].mean(0)
    y_sd = y[idx["train"]].std(0)
    y_sd[y_sd == 0] = 1

    def tensors(name):
        m = idx[name]
        return (
            torch.from_numpy(np.ascontiguousarray(W[m])),
            torch.from_numpy(((y[m] - y_mu) / y_sd).astype(np.float32)),
            torch.from_numpy(rain[m]),
        )

    xtr, ytr, rtr = tensors("train")
    xva, yva, rva = tensors("val")

    net = Net(len(feats))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    def loss_fn(out, yy, rr):
        return mse(out[:, :-1], yy) + bce(out[:, -1], rr)

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), BATCH):
            b = perm[i : i + BATCH]
            opt.zero_grad()
            loss_fn(net(xtr[b]), ytr[b], rtr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(xva), yva, rva))
        print(f"epoch {epoch:02d} val_loss {vl:.4f}")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stop")
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(name):
        xt = tensors(name)[0]
        with torch.no_grad():
            out = torch.cat([net(xt[i : i + 2048]) for i in range(0, len(xt), 2048)]).numpy()
        return out[:, :-1] * y_sd + y_mu, 1 / (1 + np.exp(-out[:, -1]))

    reg_rows, rain_rows = [], []
    preds = {n: predict(n) for n in ("val", "test")}

    grid = [i / 100 for i in range(5, 60, 5)]
    yv = lab.loc[idx["val"], "target_rain"].astype(int).values
    pv = preds["val"][1]
    best_t = max(grid, key=lambda t: metrics.rain(yv, (pv >= t).astype(int))["f1"])

    for name in ("val", "test"):
        part = lab[idx[name]]
        reg, proba = preds[name]
        for j, tgt in enumerate(REG):
            reg_rows.append(
                {"target": tgt, "model": "lstm", "split": name,
                 **metrics.regression(part[tgt].values, reg[:, j])}
            )
        wd = data.xy_to_wind_dir(reg[:, len(REG) : len(REG) + 2])
        reg_rows.append(
            {"target": "target_wind_dir", "model": "lstm", "split": name,
             **metrics.wind_dir(part["target_wind_dir"].values, wd)}
        )
        yr = part["target_rain"].astype(int).values
        rain_rows.append(
            {"target": "target_rain", "model": "lstm", "split": name,
             "threshold": best_t,
             **metrics.rain(yr, (proba >= best_t).astype(int), proba)}
        )

    reg_df, rain_df = pd.DataFrame(reg_rows), pd.DataFrame(rain_rows)
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    reg_df.to_csv(OUT / "lstm_regression.csv", index=False)
    rain_df.to_csv(OUT / "lstm_rain.csv", index=False)
    torch.save(best_state, MODELS / "lstm.pt")
    (MODELS / "lstm_meta.json").write_text(
        json.dumps(
            {"features": feats, "seq": SEQ, "hidden": HIDDEN, "reg_targets": REG,
             "rain_threshold": best_t,
             "x_mean": mu.tolist(), "x_std": sd.tolist(),
             "y_mean": y_mu.tolist(), "y_std": y_sd.tolist()}
        )
    )

    pd.set_option("display.width", 200)
    for name in ("val", "test"):
        print(f"\n=== LSTM regression, {name} ===")
        print(reg_df[reg_df["split"] == name].set_index("target")[["rmse", "mae", "r2"]].round(3).to_string())
    print("\n=== LSTM rain ===")
    print(rain_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
