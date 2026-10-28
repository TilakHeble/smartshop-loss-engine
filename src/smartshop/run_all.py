"""Rebuild everything from the raw CSVs with one command (works on Windows: no `make` needed).

    python -m smartshop.run_all              # everything except the stress tests (~15 min)
    python -m smartshop.run_all --stress     # also the stress tests (~30 min more)
    python -m smartshop.run_all --from risk  # resume from a step

Steps, in order: ingest, dbt, simulate, basket, risk, policy, [stress], mitigation, weekly,
report. Each runs as its own process, exactly as the Makefile target of the same name.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from smartshop.config import DBT_DIR

PY = [sys.executable, "-m"]
STEPS = {
    "ingest": PY + ["smartshop.ingest.load_raw"],
    "dbt": ["dbt", "build", "--profiles-dir", "."],
    "simulate": PY + ["smartshop.simulate.simulator"],
    "basket": PY + ["smartshop.basket.model"],
    "risk": PY + ["smartshop.risk.model"],
    "policy": PY + ["smartshop.policy.search"],
    "stress": PY + ["smartshop.experiments.sensitivity"],
    "mitigation": PY + ["smartshop.reporting.mitigation"],
    "weekly": PY + ["smartshop.pipeline.weekly"],
    "report": PY + ["smartshop.reporting.snapshot"],
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from", dest="start", choices=list(STEPS), default="ingest")
    parser.add_argument("--stress", action="store_true", help="include the stress tests")
    args = parser.parse_args()
    names = list(STEPS)[list(STEPS).index(args.start) :]
    if not args.stress:
        names = [n for n in names if n != "stress"]
    for name in names:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        cwd = DBT_DIR if name == "dbt" else None
        if subprocess.run(STEPS[name], cwd=cwd).returncode != 0:
            print(
                f"\nStep '{name}' failed. Fix it, then: python -m smartshop.run_all --from {name}"
            )
            return 1
        print(f"=== {name} done in {time.time() - t0:.0f}s ===", flush=True)
    print("\nAll done. Dashboard: streamlit run dashboards/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
