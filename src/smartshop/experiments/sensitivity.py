"""Stress tests: does the conclusion survive if the simulator's assumptions are wrong?

Each scenario changes one assumption, then reruns the whole chain as a store would: simulate
the world, retrain the risk model on that world's random full rescans, refit the policy shape on
the tuning period, and compare policies on the test period at today's colleague time.

To keep each run to minutes, the risk model is the week 5 configuration without the basket-gap
feature (it needs minutes of basket scoring per world); the base scenario is rerun the same way,
so every comparison is like for like.

Run:
    python -m smartshop.experiments.sensitivity        # writes models/sensitivity.csv
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from smartshop.basket.data import TEST_DAYS, TRAIN_DAYS, prepare
from smartshop.config import ROOT, WAREHOUSE_PATH
from smartshop.policy.game import build_game, outcome, responses, today_hours
from smartshop.policy.neutral import SIZE_NEUTRAL_STRENGTH, SizeNeutraliser, neutralised_game
from smartshop.policy.search import FLOOR, fit_family, make_policy, recalibrate
from smartshop.risk.features import build
from smartshop.risk.model import FEATURE_SETS, LABELS, TRAIN_LAST_DAY, candidates, observed_labels
from smartshop.simulate.config import SimConfig
from smartshop.simulate.simulator import simulate, write
from smartshop.warehouse import connect

log = logging.getLogger(__name__)
OUT = ROOT / "models" / "sensitivity.csv"
BASE = SimConfig()


def scenarios() -> dict[str, SimConfig]:
    b = BASE
    return {
        "Base assumptions": b,
        "Half as much theft": replace(b, skip_base={k: v * 0.5 for k, v in b.skip_base.items()}),
        "60% more theft": replace(b, skip_base={k: v * 1.6 for k, v in b.skip_base.items()}),
        "50% more honest mistakes": replace(
            b,
            accidental_nonscan={k: v * 1.5 for k, v in b.accidental_nonscan.items()},
            accidental_overscan={k: v * 1.5 for k, v in b.accidental_overscan.items()},
        ),
        "Weaker signals (deletions say nothing)": replace(
            b, intent_delete_rate=b.honest_delete_rate, scan_then_delete_share=0.0
        ),
        "Cautious thieves (penalty $100)": replace(b, adaptive_penalty=100.0),
        "Bold thieves (penalty $10)": replace(b, adaptive_penalty=10.0),
        "Thieves overrate detection (0.9)": replace(b, adaptive_perceived_detection=0.9),
        "Opportunists easily deterred": replace(b, opportunist_deterrence=10.0),
        "Partial checks see concealed items (50%)": replace(b, partial_detect_deliberate=0.5),
    }


def unseen_scenarios() -> dict[str, SimConfig]:
    """A theft method the risk model was not designed around: barcode switching.

    Each runs twice: retrained (the store keeps learning from random full rescans) and frozen
    (model and policy shape from the base world, only rescan volume recalibrated, as the weekly
    job does): a new method appearing after deployment.
    """
    return {
        "Barcode switching: half of theft": replace(BASE, switch_share=0.5),
        "Barcode switching: all theft": replace(BASE, switch_share=1.0),
    }


def run_scenario(
    name: str, cfg: SimConfig, workdir: Path, frozen: dict | None = None, keep: dict | None = None
) -> dict:
    """One scenario. `frozen` = {"model", "params"} reuses a base-world model and policy shapes;
    `keep` (a dict) receives this run's model and fitted shapes."""
    t0 = time.time()
    path = workdir / "scenario.duckdb"
    shutil.copy(WAREHOUSE_PATH, path)
    con = duckdb.connect(str(path), read_only=True)
    tables = simulate(con, cfg)
    con.close()
    write(tables, path)

    con = connect(path)
    prepare(con)
    feats = build(con)
    cols = FEATURE_SETS["session+history"]
    if frozen:
        model = frozen["model"]
    else:
        lab = feats[feats.day <= TRAIN_LAST_DAY].merge(observed_labels(con), on="basket_id")
        model = candidates()["logistic"].fit(lab[cols], LABELS["found anything"](lab))
    g_tune = None if frozen else build_game(con, feats, model, cols, TRAIN_DAYS, cfg=cfg)
    g_test = build_game(con, feats, model, cols, TEST_DAYS, cfg=cfg)
    con.close()
    path.unlink()

    hours_tune = None if frozen else today_hours(g_tune)
    hours_test = today_hours(g_test)
    today = outcome(g_test, None, today=True)
    out = {"scenario": name}
    fams = {
        "threshold": dict(kind="threshold", floor=FLOOR),
        "recommended": dict(kind="smooth", floor=FLOOR, cap=0.5),
        "randomised, no cap": dict(kind="smooth", floor=FLOOR, cap=1.0),
    }
    res = {}
    shapes = {}
    for k, fam in fams.items():
        shapes[k] = frozen["params"][k] if frozen else fit_family(g_tune, fam, hours_tune)
        params = recalibrate(g_test, shapes[k], hours_test)
        res[k] = (params, outcome(g_test, make_policy(params)))
    # v1.1: the same recommended shape on size-neutral scores (bands fitted on the tuning period,
    # or on the base world when the model is not retrained)
    if frozen:
        neutral = frozen["neutral"]
    else:
        neutral = SizeNeutraliser.fit(
            g_tune.sessions.v0_score.to_numpy(), g_tune.sessions.v0_scanned_units.to_numpy()
        )
    g_v11 = neutralised_game(g_test, neutral, SIZE_NEUTRAL_STRENGTH)
    v11 = outcome(g_v11, make_policy(recalibrate(g_v11, shapes["recommended"], hours_test)))
    if keep is not None:
        keep.update(model=model, params=shapes, neutral=neutral)
    rec = res["recommended"][1]
    thr_params, thr = res["threshold"]
    # what a static evaluation (shoppers keep today's behaviour) would predict for the threshold
    s = g_test.sessions
    cols_i = np.arange(len(s))
    ch = responses(g_test, None, q_today=cfg.status_quo_partial_rate + cfg.status_quo_full_rate)
    pol = make_policy(thr_params)
    static_rec = (
        pol(g_test.matrix("score")[ch, cols_i]) * g_test.matrix("full_find")[ch, cols_i]
    ).sum()
    out.update(
        {
            "today: loss remaining": today["loss_remaining"],
            "recommended: loss remaining": rec["loss_remaining"],
            "loss reduction": 1 - rec["loss_remaining"] / today["loss_remaining"],
            "recovery multiple": rec["recovered"] / today["recovered"],
            "honest stops reduction": 1 - rec["honest_stops"] / today["honest_stops"],
            "threshold: loss remaining": thr["loss_remaining"],
            "randomised no cap: loss remaining": res["randomised, no cap"][1]["loss_remaining"],
            "randomising gain vs threshold": 1
            - res["randomised, no cap"][1]["loss_remaining"] / thr["loss_remaining"],
            "static evaluation overstates recovery by": static_rec / thr["recovered"] - 1,
            "adaptive stealing, today": today["adaptive_stealing"],
            "adaptive stealing, recommended": rec["adaptive_stealing"],
            "minutes": (time.time() - t0) / 60,
        }
    )
    out["today: loss before rescans"] = today["loss_before_rescans"]
    out["model retrained"] = not frozen
    out["v1.1: loss remaining"] = v11["loss_remaining"]
    out["v1.1: loss reduction"] = 1 - v11["loss_remaining"] / today["loss_remaining"]
    out["v1.1: recovery multiple"] = v11["recovered"] / today["recovered"]
    out["v1.1: honest stops reduction"] = 1 - v11["honest_stops"] / today["honest_stops"]
    # how much deliberate loss the top 12% of risk scores holds, when thieves take their usual
    # haul (the last version of reality): does the ranking still point at theft?
    k = len(g_test.levels)
    score = g_test.sessions[f"v{k}_score"].to_numpy()
    dv = g_test.sessions[f"v{k}_deliberate_value"].to_numpy()
    top = score >= np.quantile(score, 0.88)
    out["deliberate value in top 12% of scores"] = dv[top].sum() / max(dv.sum(), 1e-9)
    log.info(
        "%-45s loss -%.1f%%  x%.1f recovery  (%.1f min)",
        name,
        100 * out["loss reduction"],
        out["recovery multiple"],
        out["minutes"],
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="scenario names to run (default: all)")
    parser.add_argument(
        "--unseen-only",
        action="store_true",
        help="run the base world and the barcode-switching tests only",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # keep rows from earlier runs; rerun scenarios replace their row
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["scenario"])
    rows = {r["scenario"]: r for r in old.to_dict("records")}

    def save():
        pd.DataFrame(list(rows.values())).to_csv(OUT, index=False)

    with tempfile.TemporaryDirectory(dir=ROOT / "data" / "processed") as d:
        base_fit: dict = {}
        for name, cfg in scenarios().items():
            if name != "Base assumptions" and (
                args.unseen_only or (args.only and name not in args.only)
            ):
                continue
            rows[name] = run_scenario(name, cfg, Path(d), keep=base_fit if cfg == BASE else None)
            save()
        if args.only and not args.unseen_only:
            return
        for name, cfg in unseen_scenarios().items():
            rows[name + " (model retrained)"] = run_scenario(
                name + " (model retrained)", cfg, Path(d)
            )
            save()
            rows[name + " (model not retrained)"] = run_scenario(
                name + " (model not retrained)", cfg, Path(d), frozen=base_fit
            )
            save()
    log.info("-> %s", OUT)


if __name__ == "__main__":
    main()
