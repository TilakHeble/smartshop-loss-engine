"""v1.2: what cancelled SmartShop sessions cost, and what closing the loophole recovers.

For today's random rescans and the v1.1 policy, under today's rules (cancelling before the till
barcode skips the decision: pay at a staffed till) and under closed rules (a cancellation is an
exit with the same decision; app-logged crashes get only the random floor). Under both, once the
barcode is scanned the session cannot be cancelled. Same test period and colleague time as the
v1.1 stress tests (one cycle, no trust).

Also: whether loss can still be measured without bias from the rescans that were completed, and
whether genuine cancellers are treated like everyone else.

    python -m smartshop.reporting.cancellations     (about 6 minutes)
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from smartshop.config import ROOT
from smartshop.policy.cancel import (
    CLOSED,
    OPEN,
    CancelConfig,
    fit_hours,
    outcome_with_cancellations,
)

log = logging.getLogger(__name__)
OUT = ROOT / "models" / "cancellations.csv"
OUT_MEASURE = ROOT / "models" / "cancellations_measurement.csv"
N_MEASURE = 30


def realised_estimates(g, policy, rules: str, cc: CancelConfig, seed: int) -> dict:
    """One realisation: who cancelled, who was selected and rescanned; the loss estimate.

    The store weights each completed rescan by one over the probability it gave that session
    (it knows that probability, not who cancelled to skip the check).
    """
    from smartshop.policy.cancel import cancel_matrix, effective_rescan, walk_matrix
    from smartshop.policy.game import responses

    cfg = g.cfg
    n_v, n = len(g.levels) + 1, len(g.sessions)
    kappa_all, walk_all = cancel_matrix(g, cc), walk_matrix(g, cc)
    p_all = np.vstack([policy(g.sessions[f"v{k}_score"].to_numpy()) for k in range(n_v)])
    choice = responses(g, None, q_matrix=effective_rescan(p_all, kappa_all, walk_all, rules))
    cols = np.arange(n)
    p, kappa, walk = p_all[choice, cols], kappa_all[choice, cols], walk_all[choice, cols]
    net_loss = g.matrix("net_loss")[choice, cols]
    missed = np.maximum(g.matrix("full_find")[choice, cols], 0) / cfg.full_detect_nonscan
    has_missing = g.matrix("nonscan_units")[choice, cols] > 0

    rng = np.random.default_rng(10_000 + seed)
    u_cancel, u_genuine, u_app, u_sel, u_walk, u_det = rng.random((6, n))
    cancelled = u_cancel < kappa
    genuine = cancelled & (u_genuine < cc.midshop_rate / np.maximum(kappa, 1e-12))
    crash = genuine & (u_app < cc.app_error_share)
    if rules == OPEN:
        decided, p_used = ~cancelled, p
    else:
        decided = np.ones(n, bool)
        p_used = np.where(crash, np.minimum(p, cfg.status_quo_full_rate), p)
    selected = decided & (u_sel < p_used)
    walked = selected & (u_walk < walk)
    completed = selected & ~walked
    found = completed & has_missing & (u_det < cfg.full_detect_nonscan)
    found_value = np.where(found, missed, 0.0)
    weighted = (found_value[completed] / p_used[completed]).sum() / cfg.full_detect_nonscan
    return {
        "truth": net_loss.clip(min=0).sum(),
        "estimate": weighted,
        "selected": selected.sum(),
        "cancelled": cancelled.sum(),
        "walked out": walked.sum(),
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    from smartshop.basket.data import TEST_DAYS, TRAIN_DAYS, prepare
    from smartshop.policy.game import build_game, outcome, today_hours
    from smartshop.policy.neutral import SizeNeutraliser, neutralised_game
    from smartshop.policy.search import POLICY_PATH, make_policy, recalibrate
    from smartshop.risk.model import MODEL_PATH, load_features
    from smartshop.warehouse import connect

    con = connect()
    prepare(con)
    feats = load_features(con)
    bundle = joblib.load(MODEL_PATH)
    params = json.loads(POLICY_PATH.read_text())["recommended"]
    g_raw = build_game(con, feats, bundle["model"], bundle["features"], TEST_DAYS)
    g_tune = build_game(con, feats, bundle["model"], bundle["features"], TRAIN_DAYS)
    neutral = SizeNeutraliser.fit(
        g_tune.sessions.v0_score.to_numpy(), g_tune.sessions.v0_scanned_units.to_numpy()
    )
    g = neutralised_game(g_raw, neutral, params.get("size_neutral_strength", 0.5))
    hours = today_hours(g)
    policy = make_policy(recalibrate(g, params, hours))

    base = CancelConfig()
    worlds = {
        "Cancellations as assumed": base,
        "Cancelling to skip half as likely": base.scaled(0.5),
        "Cancelling to skip 1.5x as likely": base.scaled(1.5),
        "Fewer genuine cancellations (1%)": CancelConfig(midshop_rate=0.01),
        "More genuine cancellations (6%)": CancelConfig(midshop_rate=0.06),
        "Nobody walks out when selected": CancelConfig(walk_away_when_selected=0.0),
        "30% walk out when selected": CancelConfig(walk_away_when_selected=0.3),
    }
    no_cancel = CancelConfig(midshop_rate=0.0, cancel_when_stealing={}, walk_away_when_selected=0.0)
    ref_today, ref_v11 = outcome(g, None, today=True), outcome(g, policy)

    rows = [
        {"world": "No cancellations (reference)", "arm": "Today", "rules": "-", **_pick(ref_today)},
        {
            "world": "No cancellations (reference)",
            "arm": "Policy v1.1",
            "rules": "-",
            **_pick(ref_v11),
        },
    ]
    check = outcome_with_cancellations(g, policy, OPEN, no_cancel)
    assert np.isclose(check["loss_remaining"], ref_v11["loss_remaining"], rtol=1e-9)
    from smartshop.policy.game import smooth_policy

    def v11(t):
        return smooth_policy(t, params["floor"], params["cap"], params["tau"])

    fitted = {}
    for name, cc in worlds.items():
        # every arm gets today's colleague time under today's rules in this world
        budget = outcome_with_cancellations(g, None, OPEN, cc, today=True)["colleague_hours"]
        for rules in (OPEN, CLOSED):
            k = fit_hours(
                lambda x, _r=rules, _c=cc: outcome_with_cancellations(
                    g, None, _r, _c, today=True, today_scale=x
                ),
                budget,
                0.2,
                2.0,
                decreasing=False,
            )
            o = outcome_with_cancellations(g, None, rules, cc, today=True, today_scale=k)
            rows.append({"world": name, "arm": "Today", "rules": rules, "volume": k, **o})
            t = fit_hours(
                lambda x, _r=rules, _c=cc: outcome_with_cancellations(g, v11(x), _r, _c),
                budget,
                -12.0,
                12.0,
            )
            o = outcome_with_cancellations(g, v11(t), rules, cc)
            rows.append({"world": name, "arm": "Policy v1.1", "rules": rules, "volume": t, **o})
            fitted[(name, rules)] = v11(t)
        log.info("%s done", name)
    table = pd.DataFrame(rows)
    today_open = table[(table.arm == "Today") & (table.rules == OPEN)].set_index("world")
    table["loss vs today (open rules)"] = table.apply(
        lambda r: (
            r.loss_remaining / today_open.loc[r.world, "loss_remaining"] - 1
            if r.world in today_open.index
            else np.nan
        ),
        axis=1,
    )
    table.to_csv(OUT, index=False)

    meas = []
    for rules in (OPEN, CLOSED):
        pol = fitted[("Cancellations as assumed", rules)]
        est = pd.DataFrame([realised_estimates(g, pol, rules, base, s) for s in range(N_MEASURE)])
        meas.append(
            {
                "rules": rules,
                "bias": est.estimate.mean() / est.truth.mean() - 1,
                "error": est.estimate.std() / est.truth.mean(),
                "rescans requested": est.selected.mean(),
                "cancellations": est.cancelled.mean(),
                "walked out": est["walked out"].mean(),
            }
        )
    meas = pd.DataFrame(meas)
    meas.to_csv(OUT_MEASURE, index=False)
    return table, meas


def _pick(o: dict) -> dict:
    return {
        k: o[k]
        for k in [
            "loss_remaining",
            "colleague_hours",
            "honest_stops",
            "adaptive_stealing",
            "opportunists_stealing",
            "loss_before_rescans",
        ]
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    table, meas = run()
    pd.set_option("display.width", 250)
    cols = [
        "world",
        "arm",
        "rules",
        "loss_remaining",
        "loss vs today (open rules)",
        "honest_stops",
        "genuine_cancellers_stopped",
        "hidden_loss_escaping_rescans",
        "colleague_hours",
        "adaptive_stealing",
    ]
    log.info("\n%s", table[cols].round(3).to_string())
    log.info("\n%s", meas.round(3).to_string())


if __name__ == "__main__":
    main()
