"""v1.5: genuine cancellations, cancel-and-walk and dead devices: what they cost, what helps.

Colleagues carry no device, and customers pick their own handset, so v1.5 uses only what the
store has (policy/exits.py, ops/handsets.py):

models/handsets.csv          dead handsets per 1,000 sessions: today, A, B, A+B (5 fleets)
models/handset_screen.csv    idea B: how long a handset must charge before it stops asking
models/exits.csv             today vs ideas 1 and 1+2, by world
models/exits_fairness.csv    idea 2: honest members given a full check, by group
models/exits_summary.json    headline numbers for the dashboard and docs

python -m smartshop.reporting.exits      (about 3 minutes)
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import pandas as pd

from smartshop.config import ROOT
from smartshop.ops.handsets import (
    POLICIES,
    RANDOM,
    SCREEN,
    SCREEN_RETIRE,
    FleetConfig,
    compare,
    simulate,
    store_timeline,
)
from smartshop.policy.exits import (
    FLAG_ONE,
    FLAG_TWO,
    TODAY,
    ExitConfig,
    flag_rates_by_group,
    run_arms,
)

log = logging.getLogger(__name__)
OUT = ROOT / "models"
TEST = (501, 711)
SCREEN_GRID = (15.0, 30.0, 60.0, 90.0, 120.0)


def sessions_frame(con) -> pd.DataFrame:
    """Test-period sessions, sorted by member then time, with what a full check would find."""
    lo, hi = TEST
    return con.sql(
        f"""
        SELECT o.basket_id, o.member_id, o.day + o.start_minute / 1440.0 AS t, o.hour_of_day,
               o.channel, m.gt_type, t.gt_true_value AS value, t.gt_true_units AS units,
               t.gt_nonscan_value AS nonscan_value, t.gt_overscan_value AS overscan_value,
               t.gt_deliberate_value AS deliberate_value
        FROM sim_obs.sessions o
        JOIN sim_truth.sessions t USING (basket_id)
        JOIN sim_truth.members m ON m.member_id = o.member_id
        WHERE o.day BETWEEN {lo} AND {hi}
        ORDER BY o.member_id, t
        """
    ).df()


def run() -> dict:
    from smartshop.basket.data import prepare
    from smartshop.reporting.fairness import session_attributes
    from smartshop.risk.model import load_features
    from smartshop.warehouse import connect

    con = connect()
    # 1. handsets: one busy store, real start times and shop lengths, customers pick freely
    lo, hi = TEST
    hs = con.sql(
        "SELECT day, start_minute, duration_min FROM sim_obs.sessions "
        f"WHERE channel = 'handset' AND day BETWEEN {lo} AND {hi}"
    ).df()
    timeline = store_timeline(hs)
    # idea B: how long on the stand before a handset stops saying "take another"? (5 fleets)
    grid = pd.DataFrame(
        [
            simulate(timeline, SCREEN, FleetConfig(seed=k, screen_minutes=m)) | {"minutes": m}
            for m in SCREEN_GRID
            for k in range(5)
        ]
    )
    grid = grid.groupby("minutes")["failures per 1,000 sessions"].mean().reset_index()
    grid.to_csv(OUT / "handset_screen.csv", index=False)
    best_minutes = float(grid.loc[grid["failures per 1,000 sessions"].idxmin(), "minutes"])
    cfg = FleetConfig(screen_minutes=best_minutes)
    fleet = compare(timeline, cfg)
    seeds = pd.concat([compare(timeline, replace(cfg, seed=k)).assign(seed=k) for k in range(5)])
    spread = seeds.groupby("policy")["failures per 1,000 sessions"].agg(["min", "max"])
    fleet = fleet.merge(spread, left_on="policy", right_index=True)
    fleet.to_csv(OUT / "handsets.csv", index=False)
    rate = fleet.set_index("policy")["failures per 1,000 sessions"] / 1000

    # 2. cancellations, walk-outs and dead devices
    s = sessions_frame(con)
    base = ExitConfig(
        handset_failure_today=float(rate[RANDOM]),
        handset_failure_v15=float(rate[SCREEN_RETIRE]),
    )
    worlds = {
        "Base assumptions": base,
        "Half as many walk-outs": base.walk_scaled(0.5),
        "Twice as many walk-outs": base.walk_scaled(2.0),
        "Idea 2 puts off 25% of walk-outs": replace(base, flag_deters=0.25),
        "Idea 2 puts off 50% of walk-outs": replace(base, flag_deters=0.5),
        "Fewer genuine shoppers finish at the point (40%)": replace(base, genuine_finish=0.4),
        "More genuine shoppers finish at the point (90%)": replace(base, genuine_finish=0.9),
        "More genuine cancellations (6%)": replace(base, genuine_cancel_rate=0.06),
        "Harder to tailgate (50% / 70%)": replace(
            base, tailgate_success_quiet=0.5, tailgate_success_peak=0.7
        ),
        "Gates flag two people per receipt (idea 3)": replace(base, gate_counts_people=True),
    }
    rows = []
    for name, cc in worlds.items():
        r = run_arms(s, cc).reset_index()
        r.insert(0, "world", name)
        rows.append(r)
        log.info("%s done", name)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT / "exits.csv", index=False)

    # 3. idea 2: which honest members get a full check, by group
    prepare(con)
    feats = load_features(con)
    attrs = session_attributes(con, feats)[
        ["member_id", "channel", "age_group", "household_size", "income_level"]
    ].rename(columns={"channel": "app or handset"})
    fair = pd.concat(
        [flag_rates_by_group(s, attrs, base, arm).assign(rule=arm) for arm in (FLAG_ONE, FLAG_TWO)]
    )
    fair.to_csv(OUT / "exits_fairness.csv", index=False)

    # 4. headline: today's loss that checks never see
    b = table[table.world == "Base assumptions"].set_index("arm")
    check_loss_today = 41130.30  # today's random checks, test period (v1.4)
    summary = {
        "sessions": int(len(s)),
        "walk_out_loss_today": float(b.loc[TODAY, "walk-out loss"]),
        "walk_out_share_of_all_loss_today": float(
            b.loc[TODAY, "walk-out loss"] / (b.loc[TODAY, "walk-out loss"] + check_loss_today)
        ),
        "screen_minutes": best_minutes,
        "handsets": fleet.to_dict("records"),
        "policies": list(POLICIES),
    }
    (OUT / "exits_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return {"table": table, "fleet": fleet, "fairness": fair, "summary": summary, "grid": grid}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = run()
    pd.set_option("display.width", 250)
    log.info("\n%s", out["grid"].round(2).to_string())
    log.info("\n%s", out["fleet"].round(2).to_string())
    b = out["table"][out["table"].world == "Base assumptions"].set_index("arm")
    log.info("\n%s", b.T.round(2).to_string())
    t = out["table"].pivot(index="world", columns="arm", values="walk-out loss")
    log.info("\nwalk-out loss by world:\n%s", t.round(0).to_string())
    log.info("\n%s", out["fairness"].round(3).to_string())


if __name__ == "__main__":
    main()
