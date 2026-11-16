from math import asin, cos, radians, sin, sqrt

import requests

from config import AQICN_TOKEN, LAT, LON


def km(lat1, lon1, lat2, lon2):
    p1, p2 = radians(lat1), radians(lat2)
    dp = p2 - p1
    dl = radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


rows = []
for kw in ["lahore", "lahore pakistan"]:
    r = requests.get(
        "https://api.waqi.info/search/",
        params={"token": AQICN_TOKEN, "keyword": kw},
        timeout=30,
    )
    r.raise_for_status()
    for s in r.json().get("data", []):
        geo = s.get("station", {}).get("geo") or [None, None]
        dist = km(LAT, LON, geo[0], geo[1]) if geo[0] is not None else 9999
        rows.append(
            (
                dist,
                s["uid"],
                s["station"]["name"],
                s.get("aqi"),
                s.get("time", {}).get("stime"),
            )
        )

seen = set()
for dist, uid, name, aqi, stime in sorted(rows):
    if uid in seen:
        continue
    seen.add(uid)
    print(f"{dist:7.1f} km | uid {uid} | aqi {aqi} | {stime} | {name}")
