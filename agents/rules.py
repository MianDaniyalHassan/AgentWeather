HIGH = {
    "target_aqi": [(201, "severe"), (151, "warning"), (101, "advisory")],
    "target_temp": [(45, "severe"), (40, "warning"), (35, "advisory")],
    "target_precip": [(7.6, "warning"), (2.5, "advisory")],
    "target_humidity": [(95, "advisory")],
    "target_uv": [(11, "severe"), (8, "warning"), (6, "advisory")],
    "target_wind_speed": [(60, "severe"), (40, "warning"), (30, "advisory")],
}
LOW = {
    "target_temp": [(0, "warning"), (5, "advisory")],
    "target_humidity": [(15, "advisory")],
}


def _first(value, rules, above):
    for lim, level in rules:
        if (value >= lim) if above else (value <= lim):
            return lim, level
    return None


def evaluate(result, rain_threshold):
    f = result["forecast"]
    out = []
    for table, above in ((HIGH, True), (LOW, False)):
        for tgt, rl in table.items():
            hit = _first(f[tgt], rl, above)
            if hit:
                out.append(
                    {
                        "metric": tgt.removeprefix("target_"),
                        "level": hit[1],
                        "forecast_value": round(f[tgt], 4),
                        "threshold": hit[0],
                        "direction": "above" if above else "below",
                    }
                )
    if result["rain_expected"] and not any(a["metric"] == "precip" for a in out):
        out.append(
            {
                "metric": "rain",
                "level": "advisory",
                "forecast_value": round(f["target_rain"], 4),
                "threshold": rain_threshold,
                "direction": "above",
            }
        )
    return out
