"""Can the age gap be closed, and what would it cost? (follow-up to the week 10 fairness check)

The gap comes from basket size: younger family households buy bigger baskets, and bigger
baskets score higher for everyone. The mitigation compares each session only with sessions of a
similar size: its score is replaced by its rank within its basket-size band, mapped back onto
the overall score distribution, so the policy's volume and shape are unchanged. A strength
between 0 (the policy as is) and 1 (fully size-neutral) blends the two on the log-odds scale.

Uses only what the store observes (units scanned), never demographics. Every setting is run as
the policy would operate: shoppers respond, trust updates over four cycles, today's colleague
time, 10 runs.

    python -m smartshop.reporting.mitigation        (about 6 minutes)
"""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from smartshop.config import ROOT
from smartshop.policy.neutral import SizeNeutraliser, neutralised_game
from smartshop.reporting.fairness import fairness_table, session_attributes

log = logging.getLogger(__name__)
OUT = ROOT / "models" / "fairness_mitigation.csv"
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_RUNS = 10


def run() -> pd.DataFrame:
    from smartshop.basket.data import TEST_DAYS, TRAIN_DAYS, prepare
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
    # fitted on the tuning period, applied to the test period
    g_tune = build_game(con, feats, bundle["model"], bundle["features"], TRAIN_DAYS)
    neutral = SizeNeutraliser.fit(
        g_tune.sessions.v0_score.to_numpy(), g_tune.sessions.v0_scanned_units.to_numpy()
    )
    g = build_game(con, feats, bundle["model"], bundle["features"], TEST_DAYS)
    events = history_events(con, TEST_DAYS[0])
    today = outcome(g, None, today=True)
    today_rate = g.cfg.status_quo_full_rate + g.cfg.status_quo_partial_rate
    attrs = session_attributes(con, feats)

    # choose on the tuning period (5 runs), report the test period
    ev_tune = history_events(con, TRAIN_DAYS[0])
    today_tune = outcome(g_tune, None, today=True)
    tune = {}
    for strength in STRENGTHS:
        gs = neutralised_game(g_tune, neutral, strength)
        c = pd.DataFrame(
            [
                run_cycles(gs, ev_tune, params, RECOMMENDED_TRUST, seed=s)[0].sum(numeric_only=True)
                for s in range(5)
            ]
        ).mean()
        tune[strength] = (
            1 - c.loss_remaining / today_tune["loss_remaining"],
            c.honest_stops / today_tune["honest_stops"],
        )

    rows = []
    for strength in STRENGTHS:
        gs = neutralised_game(g, neutral, strength)
        cyc, reals = [], []
        for seed in range(N_RUNS):
            c, r = run_cycles(gs, events, params, RECOMMENDED_TRUST, seed=seed)
            cyc.append(c.sum(numeric_only=True))
            reals.append(r)
        tot = pd.DataFrame(cyc).mean()
        per_session = (
            pd.concat(reals)
            .groupby("basket_id")
            .agg(p=("p", "mean"), honest=("honest", "first"))
            .reset_index()
            .merge(attrs, on="basket_id")
        )
        fair = fairness_table(per_session, today_rate)
        row = {
            "strength": strength,
            "tuning: loss reduction vs today": tune[strength][0],
            "tuning: honest stops vs today": tune[strength][1],
            "loss remaining": tot.loss_remaining,
            "loss reduction vs today": 1 - tot.loss_remaining / today["loss_remaining"],
            "recovered": tot.recovered,
            "recovery multiple": tot.recovered / today["recovered"],
            "honest stops": tot.honest_stops,
            "colleague hours": tot.colleague_hours,
        }
        for attr in [
            "age group",
            "household size",
            "income level",
            "channel",
            "new SmartShop user (<5 sessions)",
        ]:
            row[f"{attr}: four-fifths ratio"] = fair.loc[attr, "four-fifths ratio"]
            row[f"{attr}: verdict"] = fair.loc[attr, "verdict"]
        rows.append(row)
        log.info(
            "strength %.1f: loss -%.1f%%, recovery x%.1f, age ratio %.2f (%s)",
            strength,
            100 * row["loss reduction vs today"],
            row["recovery multiple"],
            row["age group: four-fifths ratio"],
            row["age group: verdict"],
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()


if __name__ == "__main__":
    main()
