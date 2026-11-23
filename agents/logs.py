import time
import uuid
from pathlib import Path

import pandas as pd

from ingestion.aqicn import local_hour
from ingestion.store import (
    get_agent_calls_group,
    get_alerts_group,
    get_drift_group,
    get_feature_store,
    get_ledger_group,
    get_reports_group,
    stamp,
)
from training import predict


def ledger_frame(result, folders):
    rows = []
    for tgt, value in result["forecast"].items():
        folder = folders[tgt]
        rows.append(
            {
                "time": result["origin_time"],
                "target_time": result["target_time"],
                "metric": tgt.removeprefix("target_"),
                "forecast": float(value),
                "model": predict.manifest(folder)["model"],
                "model_version": int(Path(folder).name.rsplit("_", 1)[1]),
            }
        )
    return pd.DataFrame(rows)


def write_ledger(result, folders):
    df = ledger_frame(result, folders)
    get_ledger_group(get_feature_store()).insert(stamp(df), write_options={"wait_for_job": False})
    return df


def write_drift(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["time"] = pd.Timestamp(local_hour())
    get_drift_group(get_feature_store()).insert(stamp(df), write_options={"wait_for_job": False})


def write_alerts(origin, alerts, text, provider):
    df = pd.DataFrame(alerts).assign(
        time=origin,
        public=text["public"],
        sensitive=text["sensitive"],
        workers=text["workers"],
        provider=provider,
    )
    get_alerts_group(get_feature_store()).insert(stamp(df), write_options={"wait_for_job": False})


def write_report(origin, text, provider):
    df = pd.DataFrame([{"time": origin, "report": text, "provider": provider}])
    get_reports_group(get_feature_store()).insert(stamp(df), write_options={"wait_for_job": False})


class CallLog:
    def __init__(self):
        self.rows = []

    def call(self, agent, provider, prompt, fn):
        t0 = time.perf_counter()
        response, status, error = None, "ok", ""
        try:
            response = fn()
        except Exception as e:
            status, error = "failed", str(e)
        self.rows.append(
            {
                "call_id": uuid.uuid4().hex,
                "time": pd.Timestamp(local_hour()),
                "agent": agent,
                "provider": provider,
                "prompt": prompt,
                "response": response or "",
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "status": status,
                "error": error,
            }
        )
        return response

    def flush(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        get_agent_calls_group(get_feature_store()).insert(stamp(df), write_options={"wait_for_job": False})
        self.rows = []
