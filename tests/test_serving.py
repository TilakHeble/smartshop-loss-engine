"""Serving: features match the SQL, decisions follow the policy, API, drift, release gates."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from smartshop.monitoring.drift import compare, overall, profile
from smartshop.serving.bundle import Bundle, smooth_probability
from smartshop.serving.demo_bundle import demo_bundle
from smartshop.serving.features import SESSION_COLUMNS, session_features


def _events(n=10, deletes=1):
    ev = [
        {"t": 30.0 + 60 * i, "action": "scan", "product_id": 1 + 7 * i, "unit_price": 2.5}
        for i in range(n)
    ]
    ev += [
        {"t": 30.0 + 60 * n + i, "action": "delete", "product_id": 8, "unit_price": 2.5}
        for i in range(deletes)
    ]
    return pd.DataFrame(ev)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("bundle") / "artifacts"
    demo_bundle().save(path)
    return Bundle.load(path)


@pytest.fixture(scope="module")
def sim_con(sample_warehouse, tmp_path_factory):
    from smartshop.basket.data import prepare
    from smartshop.simulate.config import SimConfig
    from smartshop.simulate.simulator import simulate, write
    from smartshop.warehouse import connect

    path = tmp_path_factory.mktemp("serving") / "sim.duckdb"
    shutil.copy(sample_warehouse, path)
    c = connect(path)
    tables = simulate(c, SimConfig(seed=5, status_quo_full_rate=0.4, status_quo_partial_rate=0.1))
    c.close()
    write(tables, path)
    c = connect(path)
    prepare(c, min_baskets=3)
    yield c
    c.close()


def test_session_features_match_the_warehouse_sql(sim_con):
    """The API computes features in pandas; training used SQL. They must agree exactly."""
    from smartshop.risk.features import session_sql

    sql = sim_con.sql(session_sql("sim_obs.sessions", "sim_obs.scan_events")).df()
    ev = sim_con.execute(
        "SELECT basket_id, seconds_from_start AS t, action, product_id, unit_price, "
        "is_high_risk FROM sim_obs.scan_events ORDER BY basket_id, event_seq"
    ).df()
    groups = dict(list(ev.groupby("basket_id")))
    assert len(sql) > 50
    for _, s in sql.iterrows():
        f = session_features(groups.get(s.basket_id, ev.iloc[:0]), s.duration_min)
        for c in SESSION_COLUMNS:
            a, b = f[c], s[c]
            assert (pd.isna(a) and pd.isna(b)) or np.isclose(a, b, rtol=1e-6, atol=1e-9), c


def test_bundle_builds_from_a_warehouse_and_decides(sim_con, tmp_path):
    from smartshop.serving.bundle import build_bundle

    policy = {"kind": "smooth", "floor": 0.02, "cap": 0.5, "tau": 0.05, "t": 0.0}
    b = build_bundle(sim_con, policy=policy, min_baskets=3)
    m = b.manifest
    assert m["training_labels"]["random_full_rescans"] > 0
    # recalibrated to today's colleague time
    used = (b.recent.p * b.recent.cost_s).mean()
    assert used == pytest.approx(m["budget"]["colleague_seconds_per_session"], rel=0.01)
    assert b.recent.p.between(0.02 - 1e-9, 0.5 + 1e-9).all()
    loaded = Bundle.load(b.save(tmp_path / "artifacts"))
    member = int(loaded.profiles.index[0])
    out = loaded.decide(member, 12.0, _events(), draw=0.5)
    assert out["known_member"] and 0.02 <= out["rescan_probability"] <= 0.5


def test_empty_session_has_no_gap_features():
    f = session_features(_events(0, 0), 5.0)
    assert f["scanned_units"] == 0 and np.isnan(f["max_gap_min"])


def test_probability_is_within_floor_and_cap():
    s = np.linspace(0.001, 0.999, 200)
    p = smooth_probability(s, t=0.0, floor=0.02, cap=0.5, tau=0.05)
    assert p.min() >= 0.02 and p.max() <= 0.5 and np.all(np.diff(p) >= 0)


def test_trust_never_goes_below_the_floor(bundle):
    assert bundle.probability(0.001, trust_mult=0.25) == pytest.approx(0.02)
    assert bundle.probability(0.999, trust_mult=0.25) == pytest.approx(0.125, rel=0.01)


def test_decision_follows_the_draw(bundle):
    ev = _events()
    low = bundle.decide(3, 15.0, ev, draw=0.0)
    high = bundle.decide(3, 15.0, ev, draw=0.999)
    assert low["decision"] == "full_rescan" and low["audit_sample"]
    assert high["decision"] == "no_rescan" and high["reasons"] == []
    assert low["rescan_probability"] == high["rescan_probability"]


def test_unknown_member_and_products_are_handled(bundle):
    ev = _events()
    ev.loc[0, "product_id"] = 999_999
    out = bundle.decide(424242, 15.0, ev, draw=0.5)
    assert not out["known_member"] and out["trust_multiplier"] == 1.0
    assert out["unknown_products"] == 1


def test_look_here_first_lists_usual_items_not_scanned(bundle):
    ev = _events()  # products 1..64: product 1 and 8 are BEEF
    out = bundle.decide(3, 15.0, ev, draw=0.0)  # member 3 usually buys BEEF|CHEESE
    assert out["look_here_first"] == ["CHEESE"]


def test_save_is_atomic_and_round_trips(tmp_path):
    b = demo_bundle()
    path = b.save(tmp_path / "artifacts")
    assert not (tmp_path / "artifacts.tmp").exists()
    again = Bundle.load(path)
    assert again.manifest["bundle_version"] == "demo"
    assert again.profiles.shape == b.profiles.shape


def test_drift_flags_a_shift_and_not_the_same_data():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(size=5000), "y": rng.normal(size=5000)})
    prof = profile(ref, ["x", "y"])
    same = pd.DataFrame({"x": rng.normal(size=5000), "y": rng.normal(size=5000)})
    shifted = same.assign(x=same.x + 1)
    assert overall(compare(prof, same)) == "ok"
    report = compare(prof, shifted)
    assert overall(report) == "alert" and report.feature.iloc[0] == "x"


def test_drift_counts_missing_values_as_their_own_bin():
    ref = pd.DataFrame({"x": np.r_[np.arange(900.0), [np.nan] * 100]})
    cur = pd.DataFrame({"x": np.r_[np.arange(500.0), [np.nan] * 500]})
    assert overall(compare(profile(ref, ["x"]), cur)) == "alert"


def test_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from smartshop.serving import app as appmod

    demo_bundle().save(tmp_path / "artifacts")
    monkeypatch.setattr(appmod, "BUNDLE_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(appmod, "DECISION_LOG", str(tmp_path / "decisions.jsonl"))
    monkeypatch.setattr(appmod, "state", appmod.State())
    with TestClient(appmod.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json()["bundle_version"] == "demo"
        assert client.get("/model").json()["policy"]["floor"] == 0.02
        assert client.get("/members/2/trust").json()["trust_multiplier"] == 0.75
        assert client.get("/members/777/trust").json()["known_member"] is False
        body = {
            "session_id": "s1",
            "member_id": 2,
            "duration_min": 15,
            "events": _events().to_dict("records"),
            "draw": 0.0,
        }
        r = client.post("/decide", json=body)
        assert r.status_code == 200 and r.json()["decision"] == "full_rescan"
        assert "X-Process-Time-Ms" in r.headers
        bad = client.post("/decide", json=body | {"duration_min": -1})
        assert bad.status_code == 422
        logged = [json.loads(x) for x in (tmp_path / "decisions.jsonl").read_text().splitlines()]
        assert logged[0]["rescan_probability"] > 0 and "draw" in logged[0]


def test_api_hot_reloads_a_new_bundle(tmp_path, monkeypatch):
    import os

    from fastapi.testclient import TestClient

    from smartshop.serving import app as appmod

    path = tmp_path / "artifacts"
    demo_bundle().save(path)
    monkeypatch.setattr(appmod, "BUNDLE_DIR", path)
    monkeypatch.setattr(appmod, "state", appmod.State())
    with TestClient(appmod.app) as client:
        assert client.get("/ready").json()["bundle_version"] == "demo"
        b = demo_bundle(seed=1)
        b.manifest["bundle_version"] = "next-week"
        b.save(path)
        st = (path / "manifest.json").stat()
        os.utime(path / "manifest.json", (st.st_atime, st.st_mtime + 5))
        assert client.get("/ready").json()["bundle_version"] == "next-week"


def test_api_is_not_ready_without_a_bundle(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from smartshop.serving import app as appmod

    monkeypatch.setattr(appmod, "BUNDLE_DIR", tmp_path / "missing")
    monkeypatch.setattr(appmod, "state", appmod.State())
    with TestClient(appmod.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_release_gates_block_a_bad_bundle():
    from smartshop.pipeline.weekly import excess_rate, gates

    b = demo_bundle()
    rng = np.random.default_rng(0)
    n = 1000
    b.recent = pd.DataFrame(
        {
            "p": np.full(n, 0.05),
            "cost_s": np.full(n, 100.0),
            "channel": np.where(np.arange(n) % 2 == 0, "app", "handset"),
            "prior_ss_sessions": rng.integers(0, 5, n),
            "risk_score": rng.random(n),
        }
    )
    b.manifest["budget"]["colleague_seconds_per_session"] = 5.0
    checks, _ = gates(b, None)
    assert all(c["pass"] for c in checks.values())
    # app shoppers rescanned twice as often: blocked
    b.recent.loc[b.recent.channel == "app", "p"] = 0.10
    b.manifest["budget"]["colleague_seconds_per_session"] = 7.5
    checks, _ = gates(b, None)
    assert not checks["channel parity"]["pass"]
    # a small group is reported, not gated
    tiny = pd.Series(np.r_[np.ones(10), np.zeros(990)].astype(bool))
    assert np.isnan(excess_rate(pd.Series(np.r_[np.ones(10), np.full(990, 0.1)]), tiny))


def test_v11_bundle_uses_size_adjusted_scores_and_keeps_the_budget(sim_con, tmp_path):
    from smartshop.serving.bundle import build_bundle

    policy = {
        "kind": "smooth",
        "floor": 0.02,
        "cap": 0.5,
        "tau": 0.05,
        "t": 0.0,
        "size_neutral_strength": 0.5,
    }
    b = build_bundle(sim_con, policy=policy, min_baskets=3)
    assert "size_neutral" in b.manifest["policy"]
    used = (b.recent.p * b.recent.cost_s).mean()
    assert used == pytest.approx(b.manifest["budget"]["colleague_seconds_per_session"], rel=0.01)
    loaded = Bundle.load(b.save(tmp_path / "artifacts"))
    out = loaded.decide(int(loaded.profiles.index[0]), 12.0, _events(), draw=0.5)
    assert 0 <= out["size_adjusted_score"] <= 1
    assert 0.02 <= out["rescan_probability"] <= 0.5
