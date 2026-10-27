"""Basket model on the synthetic warehouse (runs in CI)."""

import shutil

import numpy as np
import pandas as pd
import pytest

from smartshop.basket import model as basket_model
from smartshop.basket.data import TRAIN_DAYS, prepare
from smartshop.basket.evaluate import (
    evaluate_budget,
    greedy_fill,
    missing_sessions_in_test_period,
    missing_truth,
    true_units,
)
from smartshop.basket.features import FEATURES, build_features, masked_training_pairs
from smartshop.simulate.config import SimConfig
from smartshop.simulate.simulator import simulate, write
from smartshop.warehouse import connect


@pytest.fixture(scope="module")
def con(sample_warehouse, tmp_path_factory):
    path = tmp_path_factory.mktemp("basket") / "sim.duckdb"
    shutil.copy(sample_warehouse, path)
    c = connect(path)
    tables = simulate(c, SimConfig(seed=3))
    c.close()
    write(tables, path)
    c = connect(path)
    prepare(c, min_baskets=3)
    yield c
    c.close()


def test_history_only_records_what_was_scanned(con):
    obs = con.sql(
        """
        SELECT b.basket_id, SUM(b.units) AS observed
        FROM obs_basket_comm b JOIN obs_trips t USING (basket_id)
        WHERE t.is_smartshop GROUP BY 1
        """
    ).df()
    truth = con.sql("SELECT * FROM sim_truth.sessions").df()
    d = truth.merge(obs, on="basket_id", how="left").fillna({"observed": 0})
    scanned = d.gt_true_units - d.gt_nonscan_units + d.gt_overscan_units
    assert (d.observed == scanned).all()
    assert (d.observed < d.gt_true_units)[d.gt_nonscan_units > d.gt_overscan_units].all()


def test_masked_pairs(con):
    pairs, visible = masked_training_pairs(con, TRAIN_DAYS, hide_rate=0.3, seed=0)
    assert pairs.label.isin([0, 1]).all() and pairs.label.sum() > 0
    assert (pairs.weight >= 1).all()
    assert not pairs.duplicated(["basket_id", "commodity"]).any()
    # visible units never exceed the true basket
    true = con.sql("SELECT basket_id, commodity, units FROM obs_basket_comm").df()
    v = visible.merge(true, on=["basket_id", "commodity"])
    assert len(v) == len(visible) and (v.vis_units <= v.units).all()


def test_member_history_is_point_in_time(con):
    pairs, visible = masked_training_pairs(con, TRAIN_DAYS, seed=1)
    X = build_features(con, pairs, visible)
    trips = con.sql("SELECT basket_id, member_id, trip_order FROM obs_trips").df()
    bc = con.sql("SELECT basket_id, commodity FROM obs_basket_comm").df().merge(trips)
    sample = X.sample(min(300, len(X)), random_state=0)
    for r in sample.itertuples():
        earlier = bc[
            (bc.member_id == r.member_id)
            & (bc.commodity == r.commodity)
            & (bc.trip_order < r.trip_order)
        ]
        assert r.cum_trips == len(earlier)
    assert X[FEATURES].notna().any().all()


def test_greedy_fill_respects_budget():
    b = pd.DataFrame(
        {
            "basket_id": [1, 1, 1, 2, 2],
            "score": [5, 4, 3, 2, 1],
            "cost": [4, 2, 1, 1, 1],
        }
    )
    sel = greedy_fill(b, budget=3)
    assert sel.groupby("basket_id").cost.sum().le(3).all()
    assert set(sel[sel.basket_id == 1].score) == {4, 3}  # skips the item that does not fit


def test_train_score_and_evaluate(con, monkeypatch):
    monkeypatch.setattr(basket_model, "PARAMS", {**basket_model.PARAMS, "n_estimators": 20})
    model, metrics = basket_model.train(con)
    assert 0.5 < metrics["masked_test_auc"] <= 1
    ids = missing_sessions_in_test_period(con)
    assert len(ids) > 0
    scored = basket_model.score_sessions(con, model, ids)
    assert scored.p_missing.between(0, 1).all()
    ev = evaluate_budget(scored, missing_truth(con), true_units(con), SimConfig(), budgets=(3,))
    assert (
        ev[["accidental", "deliberate", "value_found"]].apply(lambda s: s.between(0, 1)).all().all()
    )
    assert (ev.units_counted <= 3 + 1e-9).all()
    assert np.isfinite(ev.value_found).all()
