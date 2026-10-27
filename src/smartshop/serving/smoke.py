"""Smoke test against a running API (used on the container in CI).

python -m smartshop.serving.smoke --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def _call(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--wait", type=float, default=60)
    args = parser.parse_args()
    deadline = time.time() + args.wait
    while True:
        try:
            ready = _call(f"{args.url}/ready")
            break
        except Exception:  # noqa: BLE001
            if time.time() > deadline:
                print("API never became ready", file=sys.stderr)
                return 1
            time.sleep(1)
    session = {
        "session_id": "smoke-1",
        "member_id": 2,
        "duration_min": 18,
        "events": [
            {"t": 30 + 60 * i, "action": "scan", "product_id": 1 + 7 * i, "unit_price": 2.5}
            for i in range(10)
        ]
        + [{"t": 700, "action": "delete", "product_id": 8, "unit_price": 2.5}],
    }
    first = _call(f"{args.url}/decide", session | {"draw": 0.999})
    floor = _call(f"{args.url}/decide", session | {"draw": 0.0})
    trust = _call(f"{args.url}/members/2/trust")
    checks = {
        "ready": ready["status"] == "ready",
        "high draw never rescans": first["decision"] == "no_rescan",
        "zero draw always rescans (the random floor)": floor["decision"] == "full_rescan"
        and floor["audit_sample"],
        "probability within floor and cap": 0.02 <= first["rescan_probability"] <= 0.5,
        "trust endpoint": trust["member_id"] == 2,
    }
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
