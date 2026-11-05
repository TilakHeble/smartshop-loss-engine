"""v1.4: how big should a partial check be? One share for every basket, or by basket size?

A partial check covers part of the basket and escalates to a full check if it finds anything
unscanned. A bigger sample finds more per check but takes longer, so at the same colleague time
fewer shoppers can be checked (less deterrence, fewer honest shoppers stopped). This searches the
sample rule at today's colleague time, with today's random checks (2% full, and as many partial
checks as the time allows), against shoppers who respond to the check rate:

- a fixed number of items (3 or 5: how the project modelled it before v1.4);
- one share for every basket (10% to 75%, at least 2 items);
- a share by basket size: small (up to 10 items), medium (11-30), large (over 30).

The rule is chosen on the tuning period and confirmed on the test period. Partial checks only
matter for random checks: when the till targets a shopper a full check is better
(reporting/checks.py), which is re-tested here with the chosen rule.

    python -m smartshop.reporting.sample_size        (about 15 minutes)
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd

from smartshop.config import ROOT

log = logging.getLogger(__name__)
OUT_TUNE = ROOT / "models" / "sample_size_tuning.csv"
OUT_TEST = ROOT / "models" / "sample_size.csv"
OUT_RULE = ROOT / "models" / "sample_size_rule.json"
FULL_RANDOM = 0.02  # today's random full checks (and the policy's floor), kept as they are
REF = "20% of every basket"
DETERRENCE = {
    "thieves fear any check alike": 1.0,
    "a partial check deters half as much": 0.5,
    "deterrence in proportion to items checked": "share",
}
BANDS = (10, 30)  # small: up to 10 items; medium: 11-30; large: over 30
SHARES = (0.1, 0.2, 0.3, 0.5)


def rules(base) -> dict:
    """Candidate sample rules as SimConfig variants, keyed by a readable name."""
    out = {
        "3 items (before v1.4)": replace(base, partial_share=0.0, partial_min_items=3),
        "5 items": replace(base, partial_share=0.0, partial_min_items=5),
    }
    for s in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75):
        out[f"{s:.0%} of every basket"] = replace(base, partial_share=s, partial_bands=None)
    for a, b, c in itertools.product(SHARES, repeat=3):
        if a == b == c:
            continue  # same as a single share
        out[f"by size {a:.0%}/{b:.0%}/{c:.0%}"] = replace(
            base, partial_bands=((BANDS[0], a), (BANDS[1], b), (10**9, c))
        )
    return out


def random_checks(rate: float):
    """Today's way: 2% full and `rate` partial checks, at random."""
    return lambda score: (np.full(len(score), FULL_RANDOM), np.full(len(score), rate))


def at_budget(g, hours: float) -> tuple[float, dict]:
    """The random partial rate that uses `hours` of colleague time, and its outcome."""
    from smartshop.policy.game import outcome

    lo, hi = 0.0, 0.98 - FULL_RANDOM
    for _ in range(30):
        mid = (lo + hi) / 2
        if outcome(g, random_checks(mid))["colleague_hours"] > hours:
            hi = mid
        else:
            lo = mid
    return lo, outcome(g, random_checks(lo))


def evaluate(g, candidates: dict, hours: float) -> pd.DataFrame:
    from smartshop.policy.game import use_partial_rule

    rows = []
    for name, cfg in candidates.items():
        gr = use_partial_rule(g, name, cfg)
        rate, o = at_budget(gr, hours)
        s = gr.sessions
        rows.append(
            {
                "rule": name,
                "partial check rate": rate,
                "items per partial check": s.v0_partial_units.mean(),
                "partial checks that escalate": s.v0_partial_hit.mean(),
                **o,
            }
        )
    t = pd.DataFrame(rows).set_index("rule")
    base = t.loc[REF]
    t["loss vs 20% rule"] = t.loss_remaining / base.loss_remaining - 1
    t["recovered vs 20% rule"] = t.recovered / base.recovered - 1
    t["honest stops vs 20% rule"] = t.honest_stops / base.honest_stops - 1
    return t.sort_values("loss_remaining")


def run() -> dict:
    from smartshop.basket.data import TEST_DAYS, TRAIN_DAYS, prepare
    from smartshop.policy.game import build_game, outcome, today_hours, use_partial_rule
    from smartshop.policy.neutral import SizeNeutraliser, neutralised_game
    from smartshop.policy.search import POLICY_PATH, make_policy, recalibrate
    from smartshop.reporting.checks import _mix_params
    from smartshop.risk.model import MODEL_PATH, load_features
    from smartshop.simulate.config import SimConfig
    from smartshop.warehouse import connect

    con = connect()
    prepare(con)
    feats = load_features(con)
    bundle = joblib.load(MODEL_PATH)
    m, cols = bundle["model"], bundle["features"]
    cand = rules(SimConfig())

    # 1. tuning period: every rule at today's colleague time (today's 20% rule sets the budget),
    # under each assumption about how much a partial check deters a would-be thief
    g_tune = build_game(con, feats, m, cols, TRAIN_DAYS, partial_rules=cand)
    tables = []
    for label, w in DETERRENCE.items():
        cw = {k: replace(c, partial_deterrence=w) for k, c in cand.items()}
        hours = today_hours(use_partial_rule(g_tune, REF, cw[REF]))
        t = evaluate(g_tune, cw, hours).assign(deterrence=label)
        t["regret"] = t.loss_remaining / t.loss_remaining.min() - 1
        tables.append(t)
        log.info(
            "tuning, %s: best %s (%.2f%% vs 20%% rule)",
            label,
            t.index[0],
            100 * t["loss vs 20% rule"].iloc[0],
        )
    tune = pd.concat(tables)
    tune.to_csv(OUT_TUNE)
    # loss barely depends on the rule, and which rule is best depends on an assumption nobody
    # has measured, so choose the rule whose worst case (across assumptions) is closest to best
    worst = tune.groupby(level=0).regret.max().sort_values()
    best_any = worst.index[0]
    singles = worst[worst.index.str.endswith("every basket")]
    best_fixed = singles.index[0]
    hours = today_hours(use_partial_rule(g_tune, REF, cand[REF]))

    # 2. test period: confirm, under each assumption
    keep = list(
        dict.fromkeys(
            [
                "3 items (before v1.4)",
                "10% of every basket",
                REF,
                "30% of every basket",
                "50% of every basket",
                best_fixed,
                best_any,
                *worst.index[:3],
            ]
        )
    )
    g_test = build_game(con, feats, m, cols, TEST_DAYS, partial_rules={k: cand[k] for k in keep})
    tests = []
    for label, w in DETERRENCE.items():
        cw = {k: replace(cand[k], partial_deterrence=w) for k in keep}
        h = today_hours(use_partial_rule(g_test, REF, cw[REF]))
        t = evaluate(g_test, cw, h).assign(deterrence=label)
        t["regret"] = t.loss_remaining / t.loss_remaining.min() - 1
        tests.append(t)
    test = pd.concat(tests)
    test.to_csv(OUT_TEST)

    # 3. with the chosen rule, should targeted checks still be full? (tuning period)
    policy_json = json.loads(POLICY_PATH.read_text())
    params = policy_json["recommended"]
    neutral = SizeNeutraliser.fit(
        g_tune.sessions.v0_score.to_numpy(), g_tune.sessions.v0_scanned_units.to_numpy()
    )
    gn = neutralised_game(
        use_partial_rule(g_tune, best_any, cand[best_any]),
        neutral,
        params.get("size_neutral_strength", 0.5),
    )
    sc = gn.sessions.v0_score.to_numpy()
    mix = {}
    for share in (0.0, 0.05, 1.0):
        s_full = (
            None
            if share == 1.0
            else (float(sc.max() + 1) if share == 0 else float(np.quantile(sc, 1 - share)))
        )
        p = recalibrate(gn, _mix_params(params, s_full), hours)
        mix[share] = outcome(gn, make_policy(p))["loss_remaining"]
    targeted = {
        f"top {k:.0%} full" if k < 1 else "all full": v / mix[1.0] - 1 for k, v in mix.items()
    }

    chosen = cand[best_any]
    rule = {
        "rule": best_any,
        "bands": [list(b) for b in chosen.partial_bands] if chosen.partial_bands else None,
        "share": None if chosen.partial_bands else chosen.partial_share,
        "min_items": chosen.partial_min_items,
        "tuning: worst-case loss vs the best rule": float(worst[best_any]),
        "test: loss vs 20% rule, by assumption": test.loc[best_any]
        .set_index("deterrence")["loss vs 20% rule"]
        .to_dict(),
        "best single share (tuning)": best_fixed,
        "best single share, worst case vs best rule": float(singles.iloc[0]),
        "targeted checks with this rule, loss vs all full": targeted,
    }
    OUT_RULE.write_text(json.dumps(rule, indent=2))
    return {"tune": tune, "test": test, "rule": rule}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = run()
    pd.set_option("display.width", 250)
    cols = [
        "deterrence",
        "partial check rate",
        "items per partial check",
        "partial checks that escalate",
        "loss vs 20% rule",
        "regret",
        "recovered",
        "honest stops vs 20% rule",
    ]
    t = out["tune"]
    single = t[~t.index.str.startswith("by size")]
    log.info("\ntuning, single rules:\n%s", single[cols].round(4).to_string())
    log.info(
        "\ntuning, worst-case regret (top 10):\n%s",
        t.groupby(level=0).regret.max().sort_values().head(10).round(4).to_string(),
    )
    log.info("\ntest:\n%s", out["test"][cols].round(4).to_string())
    log.info("\n%s", json.dumps(out["rule"], indent=2))


if __name__ == "__main__":
    main()
