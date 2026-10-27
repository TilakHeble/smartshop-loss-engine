"""Solve the store's side of the rescan game and compare policies.

Every policy uses the same colleague time as today's random rescans, expressed per session so a
policy fitted on the earlier (tuning) period can be applied unchanged to the later (test) period.
All parameters are fitted on the tuning period; the test period only evaluates.

Run:
    python -m smartshop.policy.search
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import pandas as pd

from smartshop.basket.data import TEST_DAYS, TRAIN_DAYS, prepare
from smartshop.config import ROOT, WAREHOUSE_PATH
from smartshop.policy.game import (
    GameData,
    build_game,
    fit_to_budget,
    outcome,
    random_policy,
    smooth_policy,
    threshold_policy,
    today_hours,
)
from smartshop.risk.model import MODEL_PATH as RISK_PATH
from smartshop.risk.model import load_features
from smartshop.warehouse import connect

log = logging.getLogger(__name__)
POLICY_PATH = ROOT / "models" / "policy.json"
FLOOR = 0.02  # random full rescans for every session: unbiased measurement and training labels

FAMILIES = {
    "Random full rescans": dict(kind="random"),
    "Threshold + random floor": dict(kind="threshold", floor=FLOOR),
    "Randomised, no cap": dict(kind="smooth", floor=FLOOR, cap=1.0),
    "Randomised, cap 0.5 (recommended)": dict(kind="smooth", floor=FLOOR, cap=0.5),
    "Randomised, cap 0.3": dict(kind="smooth", floor=FLOOR, cap=0.3),
}
TAUS = (0.05, 0.25, 0.5, 1.0)


def make_policy(params: dict):
    k = params["kind"]
    if k == "random":
        return random_policy(params["rate"])
    if k == "threshold":
        return threshold_policy(params["t"], params["floor"])
    return smooth_policy(params["t"], params["floor"], params["cap"], params["tau"])


def fit_family(g: GameData, family: dict, hours: float) -> dict:
    """Best parameters of one family on the tuning period, within the colleague-time budget."""
    k = family["kind"]
    if k == "random":
        _, x = fit_to_budget(g, lambda x: random_policy(1 - x), hours, 0, 1)
        return {**family, "rate": 1 - x}
    if k == "threshold":
        _, t = fit_to_budget(g, lambda t: threshold_policy(t, family["floor"]), hours, 0, 1)
        return {**family, "t": t}
    best = None
    for tau in TAUS:
        make = lambda t, tau=tau: smooth_policy(t, family["floor"], family["cap"], tau)  # noqa: E731
        pol, t = fit_to_budget(g, make, hours, -12, 12, iters=30)
        loss = outcome(g, pol)["loss_remaining"]
        if best is None or loss < best[0]:
            best = (loss, {**family, "tau": tau, "t": t})
    return best[1]


def recalibrate(g: GameData, params: dict, hours: float) -> dict:
    """Keep the policy's shape (floor, cap, smoothness); set its volume to this period's budget.

    A store sets rescan volume to the colleagues it has each week; only the shape is learned.
    """
    k = params["kind"]
    if k == "random":
        _, x = fit_to_budget(g, lambda x: random_policy(1 - x), hours, 0, 1)
        return {**params, "rate": 1 - x}
    if k == "threshold":
        _, t = fit_to_budget(g, lambda t: threshold_policy(t, params["floor"]), hours, 0, 1)
        return {**params, "t": t}
    make = lambda t: smooth_policy(t, params["floor"], params["cap"], params["tau"])  # noqa: E731
    _, t = fit_to_budget(g, make, hours, -12, 12, iters=30)
    return {**params, "t": t}


def compare(g: GameData, fitted: dict[str, dict], hours: float | None = None) -> pd.DataFrame:
    """Expected outcomes of today's rescans and each fitted policy on one period.

    With `hours`, each policy's volume is recalibrated to that colleague-time budget first.
    """
    rows = [{"policy": "Today: random partial + full", **outcome(g, None, today=True)}]
    for name, params in fitted.items():
        p = recalibrate(g, params, hours) if hours else params
        rows.append({"policy": name, **outcome(g, make_policy(p))})
    return pd.DataFrame(rows).set_index("policy")


def solve(con, base_features, model, cols):
    g_tune = build_game(con, base_features, model, cols, TRAIN_DAYS)
    g_test = build_game(con, base_features, model, cols, TEST_DAYS)
    # Budget: today's colleague time per session, applied to each period.
    per_session = today_hours(g_tune) / len(g_tune)
    fitted = {n: fit_family(g_tune, f, per_session * len(g_tune)) for n, f in FAMILIES.items()}
    return g_tune, g_test, fitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", default=str(WAREHOUSE_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    con = connect(args.warehouse)
    prepare(con)
    base = load_features(con)
    bundle = joblib.load(RISK_PATH)
    _, g_test, fitted = solve(con, base, bundle["model"], bundle["features"])
    table = compare(g_test, fitted, hours=today_hours(g_test))
    log.info("\n%s", table.round(2).to_string())
    rec = fitted["Randomised, cap 0.5 (recommended)"]
    POLICY_PATH.write_text(
        json.dumps(
            {"recommended": rec, "all": fitted, "test_period": table.round(4).to_dict()},
            indent=2,
            default=float,
        )
    )
    log.info("solved in %.0fs -> %s", time.time() - t0, POLICY_PATH)


if __name__ == "__main__":
    main()
