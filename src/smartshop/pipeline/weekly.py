"""The weekly job: rebuild the bundle, check it, and publish it only if every gate passes.

    1. build     retrain the risk model on all random full rescans so far, refresh member
                 profiles and trust, recalibrate the policy to today's colleague time
    2. drift     PSI of the latest cycle against the live bundle's training profile
    3. gates     release checks (below); any failure keeps the live bundle
    4. publish   atomic swap into artifacts/ (the API hot-reloads), previous bundle archived
    5. record    runs/<timestamp>.json with every check, whatever the outcome

Release gates:
    enough labels          at least MIN_LABELS random full rescans to train on
    budget                 expected colleague time within 2% of today's
    floor and cap          every probability between the random floor and the cap
    drift                  no feature in PSI alert (>= 0.25) against the live bundle
    channel parity         neither app nor handset shoppers rescanned more than 1.25x the
                           rest (the four-fifths rule, one-sided: being rescanned less is
                           not a harm); groups under MIN_GROUP sessions are reported, not gated
    new-user parity        new SmartShop users not rescanned more than 1.25x established ones
    score sanity           the new model ranks the latest cycle like the live one
                           (Spearman >= 0.7), so a broken retrain cannot slip through

    python -m smartshop.pipeline.weekly [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from smartshop.config import ROOT, WAREHOUSE_PATH
from smartshop.monitoring.drift import compare, overall
from smartshop.serving.bundle import DEFAULT_DIR, Bundle, build_bundle

log = logging.getLogger(__name__)
RUNS_DIR = ROOT / "runs"
ARCHIVE_DIR = ROOT / "artifacts_archive"
MIN_LABELS = 300
BUDGET_TOLERANCE = 0.02
PARITY_MIN = 0.8
MIN_GROUP = 100
RANK_MIN = 0.7


def excess_rate(p: pd.Series, group: pd.Series) -> float:
    """Largest ratio of a group's mean rescan probability to everyone else's.

    Only groups with at least MIN_GROUP sessions count; NaN when none qualifies.
    """
    worst = np.nan
    for g in group.dropna().unique():
        inside = group.eq(g)
        if inside.sum() < MIN_GROUP or (~inside).sum() < MIN_GROUP:
            continue
        ratio = p[inside].mean() / p[~inside].mean()
        worst = ratio if np.isnan(worst) else max(worst, ratio)
    return float(worst)


def gates(new: Bundle, live: Bundle | None) -> tuple[dict, pd.DataFrame | None]:
    m, r = new.manifest, new.recent
    pol = m["policy"]
    budget = m["budget"]["colleague_seconds_per_session"]
    used = float((r.p * r.cost_s).mean())
    new_user = r.prior_ss_sessions.fillna(0).eq(0)
    checks = {
        "enough labels": (m["training_labels"]["random_full_rescans"], ">=", MIN_LABELS),
        "budget": (used / budget - 1, "abs <=", BUDGET_TOLERANCE),
        "floor and cap": (
            float(r.p.min()) >= pol["floor"] - 1e-9 and float(r.p.max()) <= pol["cap"] + 1e-9,
            "==",
            True,
        ),
        "channel parity": (excess_rate(r.p, r.channel), "<= or too few", 1 / PARITY_MIN),
        "new-user parity": (excess_rate(r.p, new_user), "<= or too few", 1 / PARITY_MIN),
    }
    drift = None
    if live is not None:
        cur = r.assign(risk_score=live.model.predict_proba(r[live.features])[:, 1])
        drift = compare(live.manifest["drift_reference"], cur)
        checks["drift"] = (overall(drift), "!=", "alert")
        rho = pd.Series(r.risk_score.to_numpy()).corr(
            cur.risk_score.reset_index(drop=True), method="spearman"
        )
        checks["score sanity"] = (float(rho), ">=", RANK_MIN)
    ops = {
        ">=": lambda a, b: a >= b,
        "abs <=": lambda a, b: abs(a) <= b,
        "<= or too few": lambda a, b: bool(np.isnan(a) or a <= b),
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    out = {
        k: {"value": v, "rule": f"{op} {t}", "pass": bool(ops[op](v, t))}
        for k, (v, op, t) in checks.items()
    }
    return out, drift


def publish(new: Bundle, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if (out_dir / "manifest.json").exists():
        old = json.loads((out_dir / "manifest.json").read_text())["bundle_version"]
        ARCHIVE_DIR.mkdir(exist_ok=True)
        dest = ARCHIVE_DIR / old
        if not dest.exists():
            shutil.copytree(out_dir, dest)
    return new.save(out_dir)


def run(
    con, out_dir: Path = DEFAULT_DIR, cutoff_day: int | None = None, dry_run: bool = False
) -> dict:
    out_dir = Path(out_dir)
    live = Bundle.load(out_dir) if (out_dir / "manifest.json").exists() else None
    new = build_bundle(con, cutoff_day)
    checks, drift = gates(new, live)
    ok = all(c["pass"] for c in checks.values())
    published = ok and not dry_run
    if published:
        publish(new, out_dir)
    record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "candidate": new.manifest["bundle_version"],
        "live_before": live.manifest["bundle_version"] if live else None,
        "published": published,
        "dry_run": dry_run,
        "checks": checks,
        "drift": drift.to_dict("records") if drift is not None else None,
        "budget": new.manifest["budget"],
    }
    RUNS_DIR.mkdir(exist_ok=True)
    name = record["run_at"].replace(":", "").replace("-", "")[:15] + "Z.json"
    (RUNS_DIR / name).write_text(json.dumps(record, indent=2, default=_jsonable))
    return record


def _jsonable(o):
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def main() -> None:
    from smartshop.warehouse import connect

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--warehouse", default=str(WAREHOUSE_PATH))
    parser.add_argument("--out", default=str(DEFAULT_DIR))
    parser.add_argument("--cutoff-day", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rec = run(connect(args.warehouse), Path(args.out), args.cutoff_day, args.dry_run)
    for name, c in rec["checks"].items():
        log.info("%-16s %-5s %s (%s)", name, "PASS" if c["pass"] else "FAIL", c["value"], c["rule"])
    log.info(
        "candidate %s: %s", rec["candidate"], "published" if rec["published"] else "NOT published"
    )


if __name__ == "__main__":
    main()
