"""Simulator behaviour on the synthetic warehouse (CI), and calibration on real data (local)."""

import numpy as np
import pandas as pd
import pytest

from smartshop.config import WAREHOUSE_PATH
from smartshop.simulate.config import SimConfig
from smartshop.simulate.rescan import FULL, NONE, PARTIAL, rescan_outcomes
from smartshop.simulate.simulator import build_world, load_baskets, simulate, simulate_behaviour
from smartshop.warehouse import connect


@pytest.fixture(scope="module")
def sample_con(sample_warehouse):
    c = connect(sample_warehouse)
    yield c
    c.close()


@pytest.fixture(scope="module")
def tables(sample_con):
    return simulate(sample_con, SimConfig(seed=1))


@pytest.fixture(scope="module")
def world(sample_con):
    trips, lines = load_baskets(sample_con)
    return build_world(trips, lines, SimConfig(seed=1))


def test_observed_tables_hold_no_truth(tables):
    for name, df in tables.items():
        gt_cols = [c for c in df.columns if c.startswith("gt_")]
        if name.startswith("sim_obs."):
            assert not gt_cols, f"{name} leaks {gt_cols}"
    assert tables["sim_truth.sessions"].filter(like="gt_").shape[1] > 5


def test_scan_log_accounts_for_every_unit(tables):
    ev = tables["sim_obs.scan_events"]
    net = ev.assign(n=np.where(ev.action == "scan", 1, -1)).groupby("basket_id").n.sum()
    t = tables["sim_truth.sessions"].set_index("basket_id")
    expected = t.gt_true_units - t.gt_nonscan_units + t.gt_overscan_units
    assert (net.reindex(t.index, fill_value=0) == expected).all()


def test_every_session_is_nectar_linked_and_after_adoption(tables):
    s = tables["sim_obs.sessions"].merge(tables["sim_truth.members"], on="member_id")
    assert s.member_id.notna().all()
    assert (s.day >= s.adoption_day).all()
    assert set(s.channel_x) <= {"app", "handset"}


def test_simulation_is_reproducible(sample_con, tables):
    again = simulate(sample_con, SimConfig(seed=1))
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], again[name])
    other = simulate(sample_con, SimConfig(seed=2))
    assert not tables["sim_truth.members"].gt_type.equals(other["sim_truth.members"].gt_type)


def test_deterrence_with_common_random_numbers(world):
    cfg = SimConfig(seed=1)
    s0, u0 = simulate_behaviour(world, cfg, rescan_prob=0.0)
    s1, u1 = simulate_behaviour(world, cfg, rescan_prob=0.6)
    t = s0.gt_type
    # the same draws: honest and careless shoppers behave identically under both policies
    calm = u0.gt_type.isin(["honest", "careless"])
    assert (u0.loc[calm, "gt_nonscan"].values == u1.loc[calm, "gt_nonscan"].values).all()
    # opportunists and adaptive shoppers steal less when rescans are likelier
    for typ in ["opportunist", "adaptive"]:
        assert s1.gt_intent[t == typ].sum() <= s0.gt_intent[t == typ].sum()
    assert s1.gt_intent.sum() < s0.gt_intent.sum()
    # committed (deliberate) shoppers are not deterred
    assert (s0.gt_intent[t == "deliberate"].values == s1.gt_intent[t == "deliberate"].values).all()


def test_rescans_find_only_what_is_there(world):
    cfg = SimConfig(seed=1)
    s, u = simulate_behaviour(world, cfg, rescan_prob=0.1)
    ids = s.basket_id.to_numpy()
    full = rescan_outcomes(ids, np.full(len(ids), FULL), u, cfg).set_index("basket_id")
    part = rescan_outcomes(ids, np.full(len(ids), PARTIAL), u, cfg).set_index("basket_id")
    none = rescan_outcomes(ids, np.full(len(ids), NONE), u, cfg)
    truth = u.groupby("basket_id").agg(ns=("gt_nonscan", "sum"), ov=("gt_overscan", "sum"))
    assert none.empty
    assert (full.found_missing_units <= truth.ns.reindex(full.index)).all()
    assert (full.found_over_units <= truth.ov.reindex(full.index)).all()
    assert (part.found_over_units == 0).all()  # a partial rescan cannot see double scans
    assert (part.units_checked <= cfg.partial_items).all()
    assert full.found_missing_units.sum() > part.found_missing_units.sum()
    assert (full.rescan_seconds >= part.rescan_seconds.reindex(full.index)).mean() > 0.9


SIM_BUILT = WAREHOUSE_PATH.exists()


@pytest.mark.skipif(not SIM_BUILT, reason="warehouse not built")
def test_calibration_on_real_data():
    from smartshop.simulate.calibration import calibration_table

    con = connect()
    if not con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'sim_truth'"
    ).fetchall()[0][0]:
        pytest.skip("simulation not run")
    t = con.sql("SELECT * FROM sim_truth.sessions").df()
    r = con.sql("SELECT * FROM sim_obs.rescans").df()
    cal = calibration_table(t, r).simulated
    assert cal["P(error), ~20 units"] == pytest.approx(0.30, abs=0.04)
    assert cal["P(error), ~50 units"] == pytest.approx(0.60, abs=0.07)
    assert cal["P(error), ~100 units"] == pytest.approx(0.90, abs=0.05)
    assert cal["Non-scans as share of item errors"] == pytest.approx(0.77, abs=0.03)
    assert 0.46 <= cal["Deliberate share of non-scanned value"] <= 0.56
    assert 0.0096 <= cal["Net loss, share of sales"] <= 0.0468
    assert cal["Rescan rate"] == pytest.approx(0.12, abs=0.01)
    assert cal["Partial rescans finding an error"] == pytest.approx(0.0288, abs=0.015)
