from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import AQICN_STATION, AQICN_TOKEN, LAT, LON, TIMEZONE

if AQICN_STATION:
    URL = f"https://api.waqi.info/feed/@{AQICN_STATION}/"
else:
    URL = f"https://api.waqi.info/feed/geo:{LAT};{LON}/"


def local_hour():
    now = datetime.now(ZoneInfo(TIMEZONE))
    return now.replace(minute=0, second=0, microsecond=0, tzinfo=None)


def fetch_current():
    r = requests.get(URL, params={"token": AQICN_TOKEN}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "ok":
        raise RuntimeError(f"AQICN error: {body.get('data')}")
    d = body["data"]
    return pd.DataFrame(
        [
            {
                "time": pd.Timestamp(local_hour()),
                "aqi_aqicn": float(d["aqi"]),
                "station": d["city"]["name"],
            }
        ]
    )


if __name__ == "__main__":
    out = fetch_current()
    print(out.T)
    print(URL)