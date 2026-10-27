"""Week 10 reporting run: plays the recommended policy and writes everything the dashboard,
the fairness check, the model card and the business case use.

    models/fairness.csv            the full fairness check (one row per attribute)
    models/fairness_groups.csv     honest-stop rate of every group
    models/trial_sizing.csv        stores per arm for the store-level trial
    dashboards/snapshot/snapshot.json   aggregate-only snapshot for the Streamlit dashboard

The snapshot holds aggregates only (no member, household or basket identifiers), so the
dashboard can be deployed publicly without the data.

    python -m smartshop.reporting.snapshot      (about 3 minutes)
"""

from __future__ import annotations

import glob
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from smartshop.config import ROOT
from smartshop.experiments.measure import estimates
from smartshop.reporting.fairness import fairness_table, group_rates, session_attributes
from smartshop.reporting.trial import measurement_cv, sizing_table

log = logging.getLogger(__name__)
SNAPSHOT_PATH = ROOT / "dashboards" / "snapshot" / "snapshot.json"
N_RUNS = 10  # headline, fairness, operations
N_MEASURE = 30  # loss measurement: an error's spread needs more runs (same seeds as week 8)
EFFECT = 0.12


def _weekly(g, real_runs, today_fn) -> pd.DataFrame:
    s = g.sessions
    week = (s.day - s.day.min()) // 7 + 1
    rows = []
    for w in sorted(week.unique()):
        gw = g.subset((week == w).to_numpy())
        t = today_fn(gw)
        per_run = [r[r.week == w] for r in real_runs]
        rows.append(
            {
                "week": int(w),
                "sessions": len(gw),
                "loss_before_rescans": float(t["loss_before_rescans"]),
                "today_recovered": float(t["recovered"]),
                "today_honest_stops": float(t["honest_stops"]),
                "policy_recovered": float(np.mean([r.found_value.sum() for r in per_run])),
                "policy_honest_stops": float(
                    np.mean([(r.rescanned & r.honest).sum() for r in per_run])
                ),
            }
        )
    return pd.DataFrame(rows)


def build() -> dict:
    from smartshop.basket.data import TEST_DAYS, prepare
    from smartshop.policy.game import build_game, outcome
    from smartshop.policy.search import POLICY_PATH
    from smartshop.policy.trust import RECOMMENDED_TRUST, history_events, run_cycles
    from smartshop.risk.model import MODEL_PATH, load_features
    from smartshop.warehouse import connect

    con = connect()
    prepare(con)
    feats = load_features(con)
    bundle = joblib.load(MODEL_PATH)
    params = json.loads(POLICY_PATH.read_text())["recommended"]
    g = build_game(con, feats, bundle["model"], bundle["features"], TEST_DAYS)
    events = history_events(con, TEST_DAYS[0])
    cfg = g.cfg
    today_rate = cfg.status_quo_full_rate + cfg.status_quo_partial_rate
    today = outcome(g, None, today=True)
    log.info("game: %s sessions; running %s realisations", len(g), N_RUNS)

    cycles, reals = [], []
    est_rows = []
    for seed in range(N_MEASURE):
        c, r = run_cycles(g, events, params, RECOMMENDED_TRUST, seed=seed)
        r["week"] = (r.day - g.sessions.day.min()) // 7 + 1
        est_rows.append(estimates(r, params["floor"], cfg.full_detect_nonscan))
        if seed < N_RUNS:
            cycles.append(c.assign(run=seed))
            reals.append(r)
    cycles = pd.concat(cycles)
    tot = cycles.groupby("run")[
        ["loss_before_rescans", "recovered", "loss_remaining", "honest_stops", "colleague_hours"]
    ].sum()

    # ---- fairness (mean probability per session over the runs) ----
    attrs = session_attributes(con, feats)
    per_session = (
        pd.concat(reals)
        .groupby("basket_id")
        .agg(p=("p", "mean"), honest=("honest", "first"))
        .reset_index()
        .merge(attrs, on="basket_id")
    )
    fair = fairness_table(per_session, today_rate)
    groups = group_rates(per_session, today_rate)
    fair.to_csv(ROOT / "models" / "fairness.csv")
    groups.to_csv(ROOT / "models" / "fairness_groups.csv", index=False)

    # ---- measurement over the same runs ----
    est = pd.DataFrame(est_rows)
    methods = [
        "naive (average of rescans x sessions)",
        "inverse-probability weighted",
        "random floor only",
    ]
    truth = est.truth.mean()
    measurement = [
        {
            "method": m,
            "bias": float(est[m].mean() / truth - 1),
            "error": float(est[m].std() / truth),
        }
        for m in methods
    ]
    floor_error = next(x["error"] for x in measurement if x["method"] == "random floor only")

    # ---- trial sizing: both arms measured from their random full rescans (2% floor) ----
    sizing = sizing_table(EFFECT, floor_error, len(g))
    sizing.to_csv(ROOT / "models" / "trial_sizing.csv")

    # ---- operations: rescans by hour of day ----
    hours = attrs.set_index("basket_id").hour_of_day
    days = g.sessions.day.nunique()
    stores = g.sessions.merge(feats[["basket_id", "store_id"]], on="basket_id").store_id.nunique()
    pol_hour = (
        pd.concat([r[r.rescanned] for r in reals]).basket_id.map(hours).value_counts()
        / N_RUNS
        / days
    )
    sess_hour = g.sessions.basket_id.map(hours).value_counts() / days
    hourly = (
        pd.DataFrame(
            {
                "sessions_per_day": sess_hour,
                "today_rescans_per_day": sess_hour * today_rate,
                "policy_rescans_per_day": pol_hour,
            }
        )
        .fillna(0)
        .sort_index()
    )

    # ---- risk bands: where the rescans go and what they find ----
    allr = pd.concat(reals)
    allr["band"] = pd.cut(
        allr.p,
        [0, params["floor"] + 1e-9, 0.1, 0.3, 1.0],
        labels=["floor (2%)", "2-10%", "10-30%", "30-50%"],
    )
    bands = allr.groupby("band", observed=False).agg(
        sessions=("p", "size"),
        rescans=("rescanned", "sum"),
        found=("found_value", "sum"),
        honest_share=("honest", "mean"),
        mean_p=("p", "mean"),
    )
    bands["sessions"] /= bands.sessions.sum()
    bands["rescans"] /= bands.rescans.sum()
    bands["found"] /= bands.found.sum()

    # ---- the deployed policy curve, by trust level ----
    from smartshop.serving.bundle import smooth_probability

    grid = np.linspace(0.01, 0.99, 99)
    curve = {
        f"trust {m}": np.maximum(
            params["floor"],
            smooth_probability(grid, params["t"], params["floor"], params["cap"], params["tau"])
            * m,
        ).tolist()
        for m in (1.0, 0.75, 0.5, 0.25)
    }

    # ---- latest weekly job (if run): gates and drift only ----
    runs = sorted(glob.glob(str(ROOT / "runs" / "*.json")))
    weekly_job = None
    if runs:
        rec = json.loads(open(runs[-1]).read())
        weekly_job = {k: rec[k] for k in ["run_at", "candidate", "published", "checks"]}
        weekly_job["drift"] = (rec.get("drift") or [])[:10]

    sens = pd.read_csv(ROOT / "models" / "sensitivity.csv")
    weekly = _weekly(g, reals, lambda gw: outcome(gw, None, today=True))

    snap = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "note": "Simulated SmartShop sessions on dunnhumby Complete Journey baskets (US $). "
        "Aggregates only.",
        "period": {
            "days": int(days),
            "sessions": int(len(g)),
            "stores": int(stores),
            "runs": N_RUNS,
        },
        "policy": params,
        "trust_rule": json.loads(POLICY_PATH.read_text()).get("trust_rule"),
        "headline": {
            "today": {
                k: float(today[k])
                for k in ["loss_remaining", "recovered", "honest_stops", "colleague_hours"]
            },
            "policy": {
                k: float(tot[k].mean())
                for k in ["loss_remaining", "recovered", "honest_stops", "colleague_hours"]
            },
            "policy_min": {
                k: float(tot[k].min()) for k in ["loss_remaining", "recovered", "honest_stops"]
            },
            "policy_max": {
                k: float(tot[k].max()) for k in ["loss_remaining", "recovered", "honest_stops"]
            },
            "loss_before_rescans": float(today["loss_before_rescans"]),
        },
        "weekly": weekly.to_dict("records"),
        "trust_cycles": cycles.groupby("cycle")[["members_trusted", "trust_discount_to_thieves"]]
        .mean()
        .reset_index()
        .to_dict("records"),
        "bands": bands.reset_index().assign(band=lambda d: d.band.astype(str)).to_dict("records"),
        "curve": {"score": grid.tolist(), **curve},
        "hourly": hourly.reset_index(names="hour").to_dict("records"),
        "fairness": fair.reset_index().to_dict("records"),
        "fairness_groups": groups.to_dict("records"),
        "today_rescan_rate": today_rate,
        "measurement": measurement,
        "trial": {
            "effect": EFFECT,
            "sessions_per_store": 20_000,
            "measurement_cv": measurement_cv(20_000, floor_error, len(g)),
            "sizing": sizing.reset_index().to_dict("records"),
        },
        "sensitivity": sens.to_dict("records"),
        "weekly_job": weekly_job,
    }
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(_clean(snap), indent=1, default=_jsonable, allow_nan=False))
    return snap


def _clean(o):
    """Strict JSON: NaN becomes null (the dashboard and any other reader can parse it)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    return o


def _jsonable(o):
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return str(o)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = build()
    t, p = s["headline"]["today"], s["headline"]["policy"]
    log.info(
        "loss remaining %.0f -> %.0f, recovered %.0f -> %.0f, honest stops %.0f -> %.0f",
        t["loss_remaining"],
        p["loss_remaining"],
        t["recovered"],
        p["recovered"],
        t["honest_stops"],
        p["honest_stops"],
    )


if __name__ == "__main__":
    main()
