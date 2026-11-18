"""Training pipeline (brief section 5): runs daily and when the Drift Agent asks.

    python -m training.train --refresh        # CI: re-read Hopsworks, train, register
    python -m training.train --no-registry    # local: train and score, do not register
    python -m training.train --skip-fit       # reuse the results CSVs already on disk

Steps
1. Fit every family: naive, Ridge/RF/Logistic (compare.py), LSTM (lstm.py),
   SARIMAX (statistical.py). All use the same time-based split.
2. Per target, pick the best DEPLOYABLE model on validation (naive, Ridge, RF,
   Logistic, LSTM). SARIMAX is evaluated and reported but not deployed.
3. Save the winner as a model folder (see predict.py).
4. Registry: score the current registered model on TODAY's validation set (the
   window moves every day, so old scores are not comparable) and register the
   new model only if it is better. Every decision goes to registry_log.csv.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from training import data, metrics, predict
from training.compare import classifiers, regressors

ROOT = Path(__file__).parent
RES = ROOT / "results"
MODELS = ROOT / "models"
BEST = MODELS / "best"
DEPLOYABLE = {"naive", "ridge", "random_forest", "logistic", "lstm"}
TARGETS = [
    "target_aqi", "target_temp", "target_precip", "target_humidity",
    "target_uv", "target_wind_speed", "target_wind_dir", "target_rain",
]


def run_scripts(refresh):
    for mod in ("compare", "lstm", "statistical"):
        cmd = [sys.executable, "-m", f"training.{mod}"]
        if mod == "compare" and refresh:
            cmd.append("--refresh")
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT.parent)


def pick_winners():
    reg = pd.concat(
        [pd.read_csv(RES / f) for f in
         ("classical_regression.csv", "lstm_regression.csv", "statistical_regression.csv")
         if (RES / f).exists()]
    )
    rain = pd.concat(
        [pd.read_csv(RES / f) for f in ("classical_rain.csv", "lstm_rain.csv")]
    )
    win = {}
    for tgt in TARGETS[:-1]:
        val = reg[(reg["target"] == tgt) & (reg["split"] == "val")].set_index("model")["rmse"]
        dep = val[val.index.isin(DEPLOYABLE)]
        win[tgt] = (dep.idxmin(), float(dep.min()), val.idxmin())
    val = rain[(rain["split"] == "val") & rain["brier"].notna()].set_index("model")["brier"]
    dep = val[val.index.isin(DEPLOYABLE)]
    win["target_rain"] = (dep.idxmin(), float(dep.min()), val.idxmin())
    return win


def build(tgt, name, train, feats):
    """Create the model folder for `name` on `tgt`; returns the folder."""
    d = BEST / tgt
    if d.exists():
        shutil.rmtree(d)
    m = {"target": tgt, "model": name, "features": feats, "trained_at": int(time.time())}
    if name == "naive":
        predict.save(d, {**m, "kind": "naive"})
    elif name == "lstm":
        d.mkdir(parents=True)
        shutil.copy(MODELS / "lstm.pt", d / "lstm.pt")
        shutil.copy(MODELS / "lstm_meta.json", d / "lstm_meta.json")
        predict.save(d, {**m, "kind": "lstm"})
    else:
        if tgt == "target_rain":
            est = classifiers()[name]
            est.fit(train[feats], train[tgt].astype(int))
        else:
            est = regressors()[name]
            y = data.wind_dir_xy(train[tgt]) if tgt == "target_wind_dir" else train[tgt].values
            est.fit(train[feats], y)
        predict.save(d, {**m, "kind": "sklearn"}, est)
    return d


def score(folder, lab, part, tgt):
    """Metrics of a saved model on `part`. Lower primary is better."""
    p = predict.predict(folder, lab).reindex(part["time"]).values
    y = part[tgt].values
    if tgt == "target_rain":
        yi = y.astype(int)
        grid = [i / 100 for i in range(5, 60, 5)]
        thr = max(grid, key=lambda t: metrics.rain(yi, (p >= t).astype(int))["f1"])
        s = metrics.rain(yi, (p >= thr).astype(int), p)
        return {**s, "threshold": thr, "primary": s["brier"]}
    s = metrics.wind_dir(y, p) if tgt == "target_wind_dir" else metrics.regression(y, p)
    return {**s, "primary": s["rmse"]}


def incumbent(mr, name):
    try:
        models = mr.get_models(name)
        if models:
            return max(models, key=lambda m: m.version).download()
    except Exception as e:
        print(f"  no usable incumbent for {name}: {e}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-read features from Hopsworks")
    ap.add_argument("--skip-fit", action="store_true", help="reuse results CSVs on disk")
    ap.add_argument("--no-registry", action="store_true", help="do not touch the Model Registry")
    a = ap.parse_args()

    if not a.skip_fit:
        run_scripts(a.refresh)

    df = data.load()
    train, val, test = data.split(df)
    feats = data.feature_cols(df)
    tcols = [c for c in df.columns if c.startswith("target_")]
    lab = df.dropna(subset=tcols).reset_index(drop=True)

    mr = None
    if not a.no_registry:
        from ingestion.store import get_project

        mr = get_project().get_model_registry()

    winners = pick_winners()
    log = []
    for tgt in TARGETS:
        name, csv_val, overall = winners[tgt]
        print(f"\n[{tgt}] winner {name} (val {csv_val:.4f})"
              + (f" | overall best on val was {overall}, not deployed" if overall != name else ""))
        folder = build(tgt, name, train, feats)
        sv, st = score(folder, lab, val, tgt), score(folder, lab, test, tgt)
        m = predict.manifest(folder)
        if tgt == "target_rain":
            m["rain_threshold"] = sv["threshold"]
        predict.save(folder, m)
        print(f"  re-scored: val {sv['primary']:.4f} | test {st['primary']:.4f}")

        row = {"time": pd.Timestamp.now().isoformat(timespec="seconds"), "target": tgt,
               "candidate": name, "cand_val": sv["primary"], "cand_test": st["primary"],
               "incumbent_val": np.nan, "action": "local_only"}
        if mr is not None:
            reg_name = "best_" + tgt.removeprefix("target_")
            inc = incumbent(mr, reg_name)
            inc_val = np.nan
            if inc is not None:
                try:
                    inc_val = score(inc, lab, val, tgt)["primary"]
                except Exception as e:
                    print(f"  incumbent could not be scored ({e}); replacing it")
            row["incumbent_val"] = inc_val
            if np.isnan(inc_val) or sv["primary"] < inc_val:
                keys = {f"val_{k}": v for k, v in sv.items() if k != "primary"}
                keys.update({f"test_{k}": v for k, v in st.items() if k != "primary"})
                keys = {k: float(v) for k, v in keys.items() if v == v}
                mdl = mr.python.create_model(
                    name=reg_name, metrics=keys,
                    description=f"{name} for {tgt}; lower val {'brier' if tgt == 'target_rain' else 'rmse'} is better",
                )
                mdl.save(str(folder))
                row["action"] = "registered"
            else:
                row["action"] = "kept_incumbent"
            print(f"  registry: {row['action']} (incumbent val {inc_val})")
        log.append(row)

    out = pd.DataFrame(log)
    RES.mkdir(exist_ok=True)
    path = RES / "registry_log.csv"
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    pd.set_option("display.width", 200)
    print("\n", out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
