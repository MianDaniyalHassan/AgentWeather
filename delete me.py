import pandas as pd
from agents.logs import CallLog, write_drift, write_ledger
from ingestion.store import get_feature_group, get_feature_store, get_project
from training import predict

df = get_feature_group(get_feature_store()).read()
df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
df = df.sort_values("time").tail(48)

folders = predict.load_registry_models(get_project())
result = predict.forecast(folders, df)
print(write_ledger(result, folders))

log = CallLog()
log.call("test", "dummy", "ping", lambda: "pong")
log.call("test", "dummy", "ping", lambda: 1 / 0)
log.flush()

write_drift([{"metric": "aqi", "drift_type": "data", "score": 0.0, "threshold": 0.2, "action": "none"}])