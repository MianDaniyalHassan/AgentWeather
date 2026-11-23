from agents.graph import build
from agents.logs import CallLog


def main():
    log = CallLog()
    try:
        out = build(log).invoke({})
    finally:
        log.flush()
    print(out["forecast"])
    print(out.get("alerts"))
    print(out.get("alert_text"), out.get("alert_provider"))
    print(out.get("report"), out.get("report_provider"))
    checks = out.get("drift", [])
    print("drift:", out.get("drift_action"), f"({len(checks)} checks)")
    for r in checks:
        if r["action"] != "none":
            print("  ", r)


if __name__ == "__main__":
    main()
