"""Risk model on the synthetic warehouse (runs in CI)."""

import inspect
import re
import shutil

import numpy as np
import pandas as pd
import pytest

from smartshop.basket.data import prepare
from smartshop.risk import features as risk_features
from smartshop.risk.evaluate import flag_metrics
from smartshop.risk.features import NOT_FOR_SCORING, build
from smartshop.risk.model import FEATURE_SETS, observed_labels, select_and_fit, value_capture
from smartshop.simulate.config import SimConfig
from smartshop.simulate.simulator import simulate, write
from smartshop.warehouse import connect


@pytest.fixture(scope="module")
def con(sample_warehouse, tmp_path_factory):
    path = tmp_path_factory.mktemp("risk") / "sim.duckdb"
    shutil.copy(sample_warehouse, path)
    c = connect(path)
    # many random full rescans so the tiny sample has enough labels to train on
    cfg = SimConfig(seed=5, status_quo_full_rate=0.4, status_quo_partial_rate=0.1)
    tables = simulate(c, cfg)
    c.close()
    write(tables, path)
    c = connect(path)
    prepare(c, min_baskets=3)
    yield c
    c.close()


@pytest.fixture(scope="module")
def feats(con):
    return build(con)


def test_features_only_read_observable_tables():
    from smartshop.basket import data as basket_data

    for module in (risk_features, basket_data):
        src = inspect.getsource(module)
        assert not re.search(r"(FROM|JOIN)\s+sim_truth", src, flags=re.IGNORECASE)
        assert not re.search(r"\bgt_\w+", src)


def test_excluded_features_never_scored():
    for cols in FEATURE_SETS.values():
        assert not set(cols) & set(NOT_FOR_SCORING)
    demo = {"age_group", "income_level", "household_size", "kids"}
    for cols in FEATURE_SETS.values():
        assert not set(cols) & demo


def test_one_row_per_session(con, feats):
    n = con.execute("SELECT COUNT(*) FROM sim_obs.sessions").fetchall()[0][0]
    assert len(feats) == n and feats.basket_id.is_unique


def test_scan_features_match_event_log(con, feats):
    ev = con.sql("SELECT * FROM sim_obs.scan_events").df()
    net = ev.assign(n=np.where(ev.action == "scan", 1, -1)).groupby("basket_id").n.sum()
    deletes = ev[ev.action == "delete"].groupby("basket_id").size()
    f = feats.set_index("basket_id")
    assert (f.scanned_units == net.reindex(f.index, fill_value=0)).all()
    assert (f.n_deletes == deletes.reindex(f.index, fill_value=0)).all()


def test_history_uses_only_earlier_trips(con, feats):
    """Recompute member history in pandas from earlier trips only and compare."""
    trips = con.sql("SELECT basket_id, member_id, trip_order FROM obs_trips").df()
    resc = con.sql("SELECT basket_id, found_missing_units FROM sim_obs.rescans").df()
    tr = trips.merge(resc, on="basket_id", how="left")
    tr["found"] = (tr.found_missing_units > 0).astype(int)
    tr = tr.sort_values(["member_id", "trip_order"])
    tr["exp_prior_found"] = tr.groupby("member_id").found.cumsum() - tr.found
    tr["exp_prior_trips"] = tr.groupby("member_id").cumcount()
    m = feats.merge(tr[["basket_id", "exp_prior_found", "exp_prior_trips"]], on="basket_id")
    assert (m.prior_found == m.exp_prior_found).all()
    assert (m.prior_trips == m.exp_prior_trips).all()


def test_current_rescan_outcome_is_not_a_feature(con, feats):
    """A session's own rescan must not leak into its features: flip it and nothing changes."""
    resc = con.sql("SELECT basket_id FROM sim_obs.rescans").df()
    first = feats[feats.basket_id.isin(resc.basket_id)].sort_values("trip_order").iloc[0]
    # prior counts at a member's rescanned session exclude that session itself
    later = feats[(feats.member_id == first.member_id) & (feats.trip_order > first.trip_order)]
    assert first.prior_found <= (later.prior_found.min() if len(later) else np.inf)


def test_new_users_are_not_penalised_for_missing_history(feats):
    from smartshop.risk.model import candidates

    cols = FEATURE_SETS["session+history"]
    model = candidates()["logistic"]
    X = feats[cols].copy()
    y = (np.arange(len(X)) % 7 == 0).astype(int)
    model.fit(X, y)
    row = X.iloc[[0]].copy()
    hist = [
        c
        for c in cols
        if c.startswith(("m_", "prior_"))
        or c in ("units_vs_usual", "hr_share_gap", "min_per_unit_vs_usual")
    ]
    filled = row.copy()
    filled[hist] = X[hist].median().values
    empty = row.copy()
    empty[hist] = np.nan
    # no history scores exactly like typical history
    assert model.predict_proba(empty)[0, 1] == pytest.approx(model.predict_proba(filled)[0, 1])


def test_select_and_fit_on_observable_labels(con, feats):
    labels = observed_labels(con)
    model, cols, res, n = select_and_fit(feats, labels)
    assert n > 0 and len(res) > 0
    p = model.predict_proba(feats[cols])[:, 1]
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_value_capture_and_flag_metrics():
    v = np.array([0, 0, 10, 0, 30])
    assert value_capture(np.array([0, 0, 1, 0, 2]), v, 0.4) == pytest.approx(1.0)
    assert value_capture(np.array([5, 4, 0, 3, 1]), v, 0.4) == pytest.approx(0.0)
    t = pd.DataFrame(
        {
            "gt_deliberate_value": [0, 0, 5, 0, 5],
            "gt_net_loss": [1, 0, 5, 0, 5],
            "deliberate": [False, False, True, False, True],
        }
    )
    m = flag_metrics(np.array([0, 0, 1, 0, 2]), t, 0.4)
    assert m["deliberate_value_flagged"] == 1.0 and m["flagged_with_nothing_deliberate"] == 0.0
