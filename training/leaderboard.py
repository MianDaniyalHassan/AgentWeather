"""Merge every results CSV and pick the best model per target on VALIDATION.

Run from the repo root after compare.py, lstm.py and statistical.py:
    python -m training.leaderboard

Rules
- Regression targets: lowest validation RMSE wins (naive included).
- Rain: lowest validation Brier score among probabilistic models
  (naive has no probabilities, so it is shown only as a reference).
- Test scores are reported for the chosen model but never used to choose it.
"""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "results"


def load(pattern):
    files = sorted(OUT.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    reg = load("*_regression.csv")
    rain = load("*_rain.csv")

    best = {}
    rows = []
    for tgt, g in reg.groupby("target"):
        val = g[g["split"] == "val"].set_index("model")["rmse"]
        test = g[g["split"] == "test"].set_index("model")["rmse"]
        winner = val.idxmin()
        naive_test = test["naive"]
        rows.append(
            {
                "target": tgt,
                "best_model": winner,
                "val_rmse": round(val[winner], 3),
                "test_rmse": round(test[winner], 3),
                "naive_test_rmse": round(naive_test, 3),
                "test_gain_vs_naive_%": round((1 - test[winner] / naive_test) * 100, 1),
                "beats_naive_on_test": bool(test[winner] < naive_test),
            }
        )
        best[tgt] = {"model": winner, "val_rmse": float(val[winner])}

    prob = rain[(rain["model"] != "naive") & rain["brier"].notna()]
    val = prob[prob["split"] == "val"].set_index("model")["brier"]
    winner = val.idxmin()
    t = prob[(prob["split"] == "test") & (prob["model"] == winner)].iloc[0]
    rows.append(
        {
            "target": "target_rain",
            "best_model": winner,
            "val_rmse": round(val[winner], 3),  # brier, not rmse
            "test_rmse": round(t["brier"], 3),
            "naive_test_rmse": float("nan"),
            "test_gain_vs_naive_%": float("nan"),
            "beats_naive_on_test": None,
        }
    )
    best["target_rain"] = {"model": winner, "val_brier": float(val[winner])}

    board = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("(target_rain rows show Brier score in the rmse columns)\n")
    print(board.to_string(index=False))
    board.to_csv(OUT / "leaderboard.csv", index=False)
    (OUT / "best_models.json").write_text(json.dumps(best, indent=2))
    print(f"\nwrote {OUT / 'leaderboard.csv'} and best_models.json")


if __name__ == "__main__":
    main()
