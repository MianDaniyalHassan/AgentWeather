import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def ang_diff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def regression(y, p):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
    }


def wind_dir(y, p):
    d = ang_diff(p, y)
    return {
        "rmse": float(np.sqrt((d**2).mean())),
        "mae": float(np.abs(d).mean()),
        "r2": float("nan"),
    }


def rain(y, pred, proba=None):
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float("nan"),
        "brier": float("nan"),
    }
    if proba is not None and len(set(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["brier"] = float(brier_score_loss(y, proba))
    return out
