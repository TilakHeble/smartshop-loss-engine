"""v1.4: partial checks that escalate, and when to use them instead of a full check.

How checks work in store: when a SmartShop shopper scans the till barcode to pay, the till may
ask for a check. A **partial check** covers about 20-25% of the basket; if it finds anything
unscanned, the check device suggests a **full check** of everything. The till can also ask for a
full check directly. Until v1.4 the project modelled a partial check as a fixed 3 items that stopped
there, which made today's checks look weaker than they are.

This compares, at today's colleague time, on the test period, as the policy would run (shoppers
respond, trust over four cycles, 10 runs):

- today: 10% partial and 2% full checks at random (partial checks now escalate);
- v1.1: risk-based, full checks only;
- v1.4: risk-based, the same overall check rate, but only the highest-risk shoppers above the
  random floor get a full check; the rest get a partial check that escalates.

The share of risk-based checks that are full is chosen on the tuning period.

    python -m smartshop.reporting.checks        (about 10 minutes)
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd

from smartshop.config import ROOT

log = logging.getLogger(__name__)
OUT = ROOT / "models" / "checks.csv"
OUT_TUNE = ROOT / "models" / "checks_tuning.csv"
FULL_SHARES = (0.0, 0.02, 0.05, 0.1, 0.2, 1.0)  # share of sessions scoring "high"
N_RUNS = 10


def _mix_params(params: dict, s_full: float | None) -> dict:
    base = {k: params[k] for k in ("floor", "cap", "tau", "t")}
    if s_full is None:
        return {**base, "kind": "smooth"}
    return {**base, "kind": "mix", "s_full": float(s_full)}


def run() -> dict:
    from smartshop.basket.data import TEST_DAYS, TRAIN_DAYS, prepare
    from smartshop.experiments.measure import estimates
    from smartshop.policy.game import build_game, outcome, today_hours
    from smartshop.policy.neutral import SizeNeutraliser, neutralised_game
    from smartshop.policy.search import POLICY_PATH, make_policy, recalibrate
    from smartshop.policy.trust import RECOMMENDED_TRUST, history_events, run_cycles
    from smartshop.risk.model import MODEL_PATH, load_features
    from smartshop.warehouse import connect

    con = connect()
    prepare(con)
    feats = load_features(con)
    bundle = joblib.load(MODEL_PATH)
    policy_json = json.loads(POLICY_PATH.read_text())
    params = policy_json["recommended"]
    strength = params.get("size_neutral_strength", 0.5)
    g_tune_raw = build_game(con, feats, bundle["model"], bundle["features"], TRAIN_DAYS)
    g_test_raw = build_game(con, feats, bundle["model"], bundle["features"], TEST_DAYS)
    neutral = SizeNeutraliser.fit(
        g_tune_raw.sessions.v0_score.to_numpy(), g_tune_raw.sessions.v0_scanned_units.to_numpy()
    )
    g_tune = neutralised_game(g_tune_raw, neutral, strength)
    g_test = neutralised_game(g_test_raw, neutral, strength)

    # today's checks: as modelled before v1.4 (3 items, no escalation), and as they really work
    old_cfg = replace(
        g_test_raw.cfg,
        partial_share=0.0,
        partial_min_items=3,
        escalate_partial=False,
        partial_detect_nonscan=0.97,
        partial_detect_deliberate=0.15,
    )
    g_old = build_game(con, feats, bundle["model"], bundle["features"], TEST_DAYS, cfg=old_cfg)
    today_old = outcome(g_old, None, today=True)
    today = outcome(g_test, None, today=True)

    # 1. tuning period: how many risk-based checks should be full? At today's colleague time,
    # and on a short-staffed shift with half of it (when cheaper partial checks might pay).
    tune_scores = g_tune.sessions.v0_score.to_numpy()
    hours_tune = today_hours(g_tune)

    def s_full_for(share: float) -> float | None:
        if share == 1.0:
            return None
        if share == 0.0:
            return float(tune_scores.max() + 1)  # nobody: every risk-based check is partial
        return float(np.quantile(tune_scores, 1 - share))

    rows = []
    for staffing, hours in (("today", hours_tune), ("half", 0.5 * hours_tune)):
        for share in FULL_SHARES:
            p = recalibrate(g_tune, _mix_params(params, s_full_for(share)), hours)
            o = outcome(g_tune, make_policy(p))
            rows.append({"colleague_time": staffing, "full_share": share, **o})
            log.info(
                "tuning, %s time, full share %.2f: loss %.0f", staffing, share, o["loss_remaining"]
            )
    tune = pd.DataFrame(rows)
    tune.to_csv(OUT_TUNE, index=False)
    # ties (within 0.1%) go to more full checks: simpler for colleagues, and they feed training
    at_today = tune[tune.colleague_time == "today"]
    near = at_today[at_today.loss_remaining <= at_today.loss_remaining.min() * 1.001]
    best_share = float(near.full_share.max())

    # 2. test period, as the policy would run: v1.1 (full only) vs v1.4 (best mix, if different)
    events = history_events(con, TEST_DAYS[0])
    arms = {"v1.1: full checks only": _mix_params(params, None)}
    if best_share < 1.0:
        arms["v1.4: partial for medium risk, full for high"] = _mix_params(
            params, s_full_for(best_share)
        )
    out_rows = [
        {"arm": "Today, as modelled before v1.4 (3 items, no escalation)", **today_old},
        {"arm": "Today: 10% partial (escalating) + 2% full, at random", **today},
    ]
    for name, p in arms.items():
        cyc, est = [], []
        for seed in range(N_RUNS):
            c, r = run_cycles(g_test, events, p, RECOMMENDED_TRUST, seed=seed)
            cyc.append(c.sum(numeric_only=True))
            est.append(estimates(r, p["floor"], g_test.cfg.full_detect_nonscan))
            if seed == 0:
                kinds = r[r.rescanned].kind.value_counts(normalize=True).to_dict()
        tot = pd.DataFrame(cyc).mean()
        e = pd.DataFrame(est)
        out_rows.append(
            {
                "arm": name,
                **tot.to_dict(),
                "share of checks that are full": kinds.get("full", 0.0),
                "loss estimate bias": e["inverse-probability weighted"].mean() / e.truth.mean() - 1,
            }
        )
        log.info("test, %s: loss %.0f", name, tot.loss_remaining)
    table = pd.DataFrame(out_rows).set_index("arm")
    t0 = table.loc["Today: 10% partial (escalating) + 2% full, at random"]
    table["loss vs today"] = table.loss_remaining / t0.loss_remaining - 1
    table["recovery vs today"] = table.recovered / t0.recovered
    table["honest stops vs today"] = table.honest_stops / t0.honest_stops - 1
    table["recovered per colleague hour"] = table.recovered / table.colleague_hours
    table.to_csv(OUT)

    # 3. record the choice in the policy (v1.4)
    rec = dict(params)
    rec["full_share"] = best_share
    policy_json["recommended"] = rec
    policy_json["v1_4_note"] = (
        "Partial checks (about 20-25% of the basket) escalate to a full check when they find "
        "anything. full_share: the share of sessions, by size-adjusted score, whose risk-based "
        "check is full; the rest are partial. The random floor is always a full check. "
        "Chosen on the tuning period (models/checks_tuning.csv)."
    )
    POLICY_PATH.write_text(json.dumps(policy_json, indent=2))
    return {"table": table, "tune": tune, "best_share": best_share}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = run()
    pd.set_option("display.width", 250)
    cols = [
        "loss_remaining",
        "loss vs today",
        "recovered",
        "recovery vs today",
        "honest_stops",
        "honest stops vs today",
        "colleague_hours",
        "recovered per colleague hour",
        "share of checks that are full",
        "loss estimate bias",
    ]
    log.info("\n%s", out["table"][[c for c in cols if c in out["table"]]].round(3).to_string())
    log.info("chosen full share: %s", out["best_share"])


if __name__ == "__main__":
    main()
