"""v1.3: genuine cancellations, cancel-and-walk and dead devices: what they cost, what helps.

models/handsets.csv          dead handsets per 1,000 sessions by how handsets are handed out
models/exits.csv             today vs v1.3 (help every cancellation / higher-risk only), by world
models/exits_fairness.csv    who a "higher-risk" help prompt goes to, among genuine cancellers
models/exits_summary.json    headline numbers for the dashboard and docs

python -m smartshop.reporting.exits      (about 3 minutes)
"""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from smartshop.config import ROOT
from smartshop.ops.handsets import (
    LONGEST_FIRST_RETIRE,
    RANDOM,
    FleetConfig,
    compare,
    store_timeline,
)
from smartshop.policy.exits import (
    TODAY,
    ExitConfig,
    prompt_rates_by_group,
    run_arms,
)

log = logging.getLogger(__name__)
OUT = ROOT / "models"


def sessions_frame(con):
    """Test-period sessions with type, basket value, units, hour, channel and v1.1 risk score."""
    from smartshop.basket.data import TEST_DAYS, TRAIN_DAYS, prepare
    from smartshop.policy.game import build_game
    from smartshop.policy.neutral import SizeNeutraliser, neutralised_game
    from smartshop.policy.search import POLICY_PATH
    from smartshop.risk.model import MODEL_PATH, load_features

    prepare(con)
    feats = load_features(con)
    bundle = joblib.load(MODEL_PATH)
    params = json.loads(POLICY_PATH.read_text())["recommended"]
    g = build_game(con, feats, bundle["model"], bundle["features"], TEST_DAYS)
    g_tune = build_game(con, feats, bundle["model"], bundle["features"], TRAIN_DAYS)
    neutral = SizeNeutraliser.fit(
        g_tune.sessions.v0_score.to_numpy(), g_tune.sessions.v0_scanned_units.to_numpy()
    )
    g = neutralised_game(g, neutral, params.get("size_neutral_strength", 0.5))
    # a cancel-and-walk scans honestly, so its scan log is the honest version's (v0)
    s = g.sessions[["basket_id", "member_id", "gt_type", "v0_score"]].rename(
        columns={"v0_score": "score"}
    )
    truth = con.sql(
        "SELECT basket_id, gt_true_value AS value, gt_true_units AS units FROM sim_truth.sessions"
    ).df()
    obs = con.sql("SELECT basket_id, hour_of_day, channel FROM sim_obs.sessions").df()
    return s.merge(truth, on="basket_id").merge(obs, on="basket_id"), feats


def run() -> dict:
    from smartshop.reporting.fairness import session_attributes
    from smartshop.warehouse import connect

    con = connect()
    # 1. handsets: one busy store, real start times and shop lengths
    hs = con.sql(
        "SELECT day, start_minute, duration_min FROM sim_obs.sessions "
        "WHERE channel = 'handset' AND day BETWEEN 501 AND 711"
    ).df()
    timeline = store_timeline(hs)
    fleet = compare(timeline)
    seeds = pd.concat([compare(timeline, FleetConfig(seed=k)).assign(seed=k) for k in range(5)])
    spread = seeds.groupby("policy")["failures per 1,000 sessions"].agg(["min", "max"])
    fleet = fleet.merge(spread, left_on="policy", right_index=True)
    fleet.to_csv(OUT / "handsets.csv", index=False)
    rate = fleet.set_index("policy")["failures per 1,000 sessions"] / 1000

    # 2. cancellations, walk-outs and dead devices
    s, feats = sessions_frame(con)
    base = ExitConfig(
        handset_failure_today=float(rate[RANDOM]),
        handset_failure_v13=float(rate[LONGEST_FIRST_RETIRE]),
    )
    worlds = {
        "Base assumptions": base,
        "Half as many walk-outs": base.walk_scaled(0.5),
        "Twice as many walk-outs": base.walk_scaled(2.0),
        "Colleagues stretched (free 40% / 30%)": replace_cfg(
            base, colleague_free_quiet=0.4, colleague_free_peak=0.3
        ),
        "Harder to tailgate (50% / 70%)": replace_cfg(
            base, tailgate_success_quiet=0.5, tailgate_success_peak=0.7
        ),
        "Gates flag two people per receipt": replace_cfg(base, gate_counts_people=True),
        "Fewer genuine cancellations (1%)": replace_cfg(base, genuine_cancel_rate=0.01),
        "More genuine cancellations (6%)": replace_cfg(base, genuine_cancel_rate=0.06),
        "Prompts only for the top 25%": replace_cfg(base, risk_share=0.25),
    }
    rows = []
    for name, cc in worlds.items():
        r = run_arms(s, cc).reset_index()
        r.insert(0, "world", name)
        rows.append(r)
        log.info("%s done", name)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT / "exits.csv", index=False)

    # 3. who a "higher-risk" prompt goes to, among genuine cancellers
    attrs = session_attributes(con, feats)[
        [
            "basket_id",
            "channel",
            "new_smartshop_user",
            "age_group",
            "household_size",
            "income_level",
        ]
    ]
    fair = prompt_rates_by_group(s, attrs.rename(columns={"channel": "app or handset"}), base)
    fair.to_csv(OUT / "exits_fairness.csv", index=False)

    # 4. headline: today's loss that rescans never see
    b = table[table.world == "Base assumptions"].set_index("arm")
    rescan_loss_today = 41639.17  # today's random rescans, test period (week 8)
    summary = {
        "sessions": int(len(s)),
        "walk_out_loss_today": float(b.loc[TODAY, "walk-out loss"]),
        "walk_out_share_of_all_loss_today": float(
            b.loc[TODAY, "walk-out loss"] / (b.loc[TODAY, "walk-out loss"] + rescan_loss_today)
        ),
        "handsets": fleet.to_dict("records"),
    }
    (OUT / "exits_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return {"table": table, "fleet": fleet, "fairness": fair, "summary": summary}


def replace_cfg(cc: ExitConfig, **kw) -> ExitConfig:
    from dataclasses import replace

    return replace(cc, **kw)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = run()
    pd.set_option("display.width", 250)
    log.info("\n%s", out["fleet"].round(2).to_string())
    b = out["table"][out["table"].world == "Base assumptions"].set_index("arm")
    log.info("\n%s", b.T.round(2).to_string())
    log.info("\n%s", out["fairness"].round(3).to_string())


if __name__ == "__main__":
    main()
