from ingestion.store import (
    get_agent_calls_group,
    get_alerts_group,
    get_drift_group,
    get_feature_store,
    get_ledger_group,
    get_registry_group,
    get_reference_group,
    get_reports_group,
)

GROUPS = {
    "forecast_ledger": get_ledger_group,
    "registry_log": get_registry_group,
    "drift_reference": get_reference_group,
    "drift_log": get_drift_group,
    "alerts": get_alerts_group,
    "weather_reports": get_reports_group,
    "agent_calls": get_agent_calls_group,
}

fs = get_feature_store()
for name, getter in GROUPS.items():
    try:
        df = getter(fs).read()
        latest = df["time"].max() if "time" in df.columns and len(df) else "n/a"
        print(f"{name}: {len(df)} rows, latest {latest}")
    except Exception as e:
        print(f"{name}: no data yet ({type(e).__name__})")
