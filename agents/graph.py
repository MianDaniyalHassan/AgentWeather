import json
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents import drift, logs, rules
from agents.llm import generate
from ingestion.store import get_alerts_group, get_feature_group, get_project
from training import predict

REPORT_HOUR = 7
HISTORY_HOURS = 24 * 15
AUDIENCES = ("public", "sensitive", "workers")
LABELS = {
    "aqi": "air quality index",
    "temp": "temperature",
    "precip": "precipitation",
    "humidity": "humidity",
    "uv": "UV index",
    "wind_speed": "wind speed",
    "rain": "rain",
}


class State(TypedDict, total=False):
    folders: dict
    forecast: dict
    history: object
    alerts: list
    alert_text: dict
    alert_provider: str
    new_alert: bool
    report: str
    report_provider: str
    drift: list
    drift_action: str


def compass(deg):
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int((deg + 22.5) // 45) % 8]


def readable(result):
    f = result["forecast"]
    t = result["target_time"]
    return "\n".join(
        [
            f"Valid for: {t:%A %d %B %Y, %H:%M} (Lahore local time)",
            f"Temperature: {f['target_temp']:.1f} degrees Celsius",
            f"Humidity: {f['target_humidity']:.0f} percent",
            f"Wind: {f['target_wind_speed']:.1f} km/h from the {compass(f['target_wind_dir'])}",
            f"UV index: {f['target_uv']:.1f}",
            f"Air quality index (AQI): {f['target_aqi']:.1f}",
            f"Precipitation: {f['target_precip']:.1f} mm",
            f"Chance of rain: {f['target_rain'] * 100:.0f} percent",
        ]
    )


def readable_alerts(alerts):
    return [
        {**a, "metric": LABELS.get(a["metric"], a["metric"]), "forecast_value": round(a["forecast_value"], 1)}
        for a in alerts
    ]


def alert_prompt(result, alerts):
    return (
        "You write weather and air quality alerts for Lahore, Pakistan. "
        f"The forecast is valid for {result['target_time']:%A %d %B %Y, %H:%M}. "
        f"Triggered hazards: {json.dumps(readable_alerts(alerts))}. "
        "Reply with only a JSON object with the keys public, sensitive and workers. "
        "public is advice for the general public. "
        "sensitive is advice for children, the elderly and people with heart or lung conditions. "
        "workers is advice for outdoor workers. "
        "Each value is 1 to 2 plain sentences and uses only the numbers given."
    )


def alert_template(alerts):
    s = "; ".join(
        f"{LABELS.get(a['metric'], a['metric'])} {a['forecast_value']:.1f} ({a['level']})" for a in alerts
    )
    return {
        "public": f"Hazard forecast for the next 24 hours: {s}. Limit time outdoors where possible.",
        "sensitive": f"Hazard forecast for the next 24 hours: {s}. Children, the elderly and people with heart or lung conditions should stay indoors and keep medication close.",
        "workers": f"Hazard forecast for the next 24 hours: {s}. Outdoor workers should shorten shifts, take regular breaks and use protective gear.",
    }


def report_prompt(result, alerts):
    return (
        "Write a short news-style weather report for Lahore, Pakistan, 100 to 130 words, "
        "plain text, no headings, no bullet points. This is a forecast, not a measurement.\n"
        f"{readable(result)}\n"
        f"Rain expected: {'yes' if result['rain_expected'] else 'no'}.\n"
        f"Active alerts: {json.dumps(readable_alerts(alerts))}.\n"
        "Use only the facts above, rounded as written, in natural language. "
        "Do not add advice, locations, times or events that are not listed. "
        "Do not describe cloud cover or sky conditions. "
        "Use the weekday exactly as given."
    )


def report_template(result, alerts):
    f = result["forecast"]
    text = (
        f"Lahore forecast for {result['target_time']:%d %b %H:%M}: "
        f"temperature {f['target_temp']:.1f} C, humidity {f['target_humidity']:.0f}%, "
        f"wind {f['target_wind_speed']:.1f} km/h from the {compass(f['target_wind_dir'])}, "
        f"UV index {f['target_uv']:.1f}, AQI {f['target_aqi']:.1f}, "
        f"precipitation {f['target_precip']:.2f} mm "
        f"({'rain expected' if result['rain_expected'] else 'no rain expected'})."
    )
    if alerts:
        text += " Active alerts: " + "; ".join(f"{LABELS.get(a['metric'], a['metric'])} ({a['level']})" for a in alerts) + "."
    return text


def previous_alerts(origin):
    try:
        fs = get_project().get_feature_store()
        df = get_alerts_group(fs).read()
    except Exception:
        return set()
    if df.empty:
        return set()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    last = df[df["time"] == origin - pd.Timedelta(hours=1)]
    return set(zip(last["metric"], last["level"]))


def forecast_node(state):
    project = get_project()
    fs = project.get_feature_store()
    df = get_feature_group(fs).read()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.sort_values("time")
    history = df.tail(HISTORY_HOURS).reset_index(drop=True)
    df = df.tail(48)
    folders = predict.load_registry_models(project)
    result = predict.forecast(folders, df)
    logs.write_ledger(result, folders)
    return {"forecast": result, "folders": folders, "history": history}


def make_alert_node(log):
    def node(state):
        result = state["forecast"]
        thr = predict.manifest(state["folders"]["target_rain"])["rain_threshold"]
        alerts = rules.evaluate(result, thr)
        if not alerts:
            return {"alerts": [], "new_alert": False}
        prev = previous_alerts(result["origin_time"])
        new = any((a["metric"], a["level"]) not in prev for a in alerts)
        text, provider = generate(
            log, "alert", alert_prompt(result, alerts), alert_template(alerts), keys=AUDIENCES
        )
        logs.write_alerts(result["origin_time"], alerts, text, provider)
        return {"alerts": alerts, "alert_text": text, "alert_provider": provider, "new_alert": new}

    return node


def make_report_node(log):
    def node(state):
        result, alerts = state["forecast"], state["alerts"]
        text, provider = generate(
            log, "report", report_prompt(result, alerts), report_template(result, alerts)
        )
        logs.write_report(result["origin_time"], text, provider)
        return {"report": text, "report_provider": provider}

    return node


def drift_node(state):
    try:
        rows, action = drift.run(state["history"], get_project())
    except Exception as e:
        print(f"drift agent failed: {e}")
        return {"drift": [], "drift_action": "error"}
    logs.write_drift(rows)
    return {"drift": rows, "drift_action": action}


def route(state):
    daily = state["forecast"]["origin_time"].hour == REPORT_HOUR
    return "report" if daily or state.get("new_alert") else "drift"


def build(log):
    g = StateGraph(State)
    g.add_node("forecast", forecast_node)
    g.add_node("alert", make_alert_node(log))
    g.add_node("report", make_report_node(log))
    g.add_node("drift", drift_node)
    g.add_edge(START, "forecast")
    g.add_edge("forecast", "alert")
    g.add_conditional_edges("alert", route, {"report": "report", "drift": "drift"})
    g.add_edge("report", "drift")
    g.add_edge("drift", END)
    return g.compile()
