"""The rescan game on the synthetic warehouse (runs in CI)."""

import shutil

import numpy as np
import pytest

from smartshop.basket.data import prepare
from smartshop.policy.game import (
    STRATEGIC,
    build_game,
    fit_to_budget,
    outcome,
    random_policy,
    responses,
    smooth_policy,
    threshold_policy,
    today_hours,
)
from smartshop.risk.features import build
from smartshop.risk.model import FEATURE_SETS, LABELS, candidates, observed_labels
from smartshop.simulate.config import SimConfig
from smartshop.simulate.simulator import (
    build_world,
    load_baskets,
    simulate,
    simulate_behaviour,
    write,
)
from smartshop.warehouse import connect

CFG = SimConfig(seed=7, status_quo_full_rate=0.3, status_quo_partial_rate=0.1)


@pytest.fixture(scope="module")
def game(sample_warehouse, tmp_path_factory):
    path = tmp_path_factory.mktemp("policy") / "sim.duckdb"
    shutil.copy(sample_warehouse, path)
    c = connect(path)
    tables = simulate(c, CFG)
    c.close()
    write(tables, path)
    c = connect(path)
    prepare(c, min_baskets=3)
    feats = build(c)
    cols = FEATURE_SETS["session+history"]
    lab = feats.merge(observed_labels(c), on="basket_id")
    model = candidates()["logistic"].fit(lab[cols], LABELS["found anything"](lab))
    g = build_game(c, feats, model, cols, (1, 711), cfg=CFG)
    yield c, g
    c.close()


def test_non_strategic_shoppers_are_the_same_in_every_version(game):
    _, g = game
    s = g.sessions
    calm = ~s.gt_type.isin(STRATEGIC)
    for v in ["score", "net_loss", "deliberate_value", "full_find"]:
        m = g.matrix(v)[:, calm.to_numpy()]
        assert np.allclose(m, m[0], equal_nan=True)


def test_haul_grows_with_level(game):
    _, g = game
    haul = g.matrix("deliberate_value")
    assert (np.diff(haul, axis=0) >= -1e-9).all()
    strategic = g.sessions.gt_type.isin(STRATEGIC).to_numpy()
    assert (
        haul[0][strategic] == 0
    ).all()  # the honest version of a strategic shopper takes nothing


def test_opportunists_match_the_simulator_under_todays_rescans(game):
    c, g = game
    q = CFG.status_quo_partial_rate + CFG.status_quo_full_rate
    choice = responses(g, None, q_today=q)
    world = build_world(*load_baskets(c), CFG)
    s, _ = simulate_behaviour(world, CFG, rescan_prob=q)
    sim = s.set_index("basket_id").gt_intent
    opp = g.sessions.gt_type == "opportunist"
    ids = g.sessions.basket_id[opp]
    assert ((choice[opp.to_numpy()] > 0) == sim.loc[ids].to_numpy()).all()


def test_more_rescans_deter_strategic_shoppers(game):
    _, g = game
    low, high = outcome(g, random_policy(0.01)), outcome(g, random_policy(0.6))
    assert high["opportunists_stealing"] <= low["opportunists_stealing"]
    assert high["adaptive_stealing"] <= low["adaptive_stealing"]


def test_accounting(game):
    _, g = game
    for pol in [
        random_policy(0.1),
        threshold_policy(0.5, 0.02),
        smooth_policy(0.0, 0.02, 0.5, 0.25),
    ]:
        o = outcome(g, pol)
        assert o["loss_remaining"] == pytest.approx(o["loss_before_rescans"] - o["recovered"])
        assert 0 <= o["honest_stops"] <= o["rescans"] + 1e-9
        assert o["colleague_hours"] >= 0


def test_policies_respect_floor_and_cap():
    score = np.linspace(0.001, 0.999, 200)
    p = smooth_policy(0.0, 0.02, 0.5, 0.25)(score)
    assert p.min() >= 0.02 - 1e-12 and p.max() <= 0.5 + 1e-12
    assert (np.diff(p) >= 0).all()
    t = threshold_policy(0.5, 0.02)(score)
    assert set(np.unique(t)) == {0.02, 1.0}


def test_fit_to_budget_uses_no_more_than_the_budget(game):
    _, g = game
    hours = today_hours(g)
    pol, _ = fit_to_budget(g, lambda t: smooth_policy(t, 0.02, 0.5, 0.25), hours, -12, 12, iters=25)
    assert outcome(g, pol)["colleague_hours"] <= hours * 1.001


# --- trust ---------------------------------------------------------------------------------


def _events(rows):
    import pandas as pd

    return pd.DataFrame(rows, columns=["member_id", "day", "kind", "found_units", "found_value"])


def test_trust_rule():
    from smartshop.policy.trust import TrustRule, multipliers

    rule = TrustRule(step=0.25, max_discount=0.75)
    ev = _events(
        [
            (1, 10, "full", 0, 0.0),  # member 1: two clean full rescans -> 50% off
            (1, 20, "full", 0, 0.0),
            (2, 10, "full", 0, 0.0),  # member 2: clean, then a real find -> reset
            (2, 30, "full", 3, 25.0),
            (3, 10, "full", 0, 0.0),  # member 3: a small slip does not reset
            (3, 30, "full", 1, 4.0),
            (4, 10, "partial", 0, 0.0),  # member 4: partial checks do not count by default
            (5, 10, "full", 0, 0.0),  # member 5: six clean rescans, capped at 75% off
            (5, 11, "full", 0, 0.0),
            (5, 12, "full", 0, 0.0),
            (5, 13, "full", 0, 0.0),
            (5, 14, "full", 0, 0.0),
            (5, 15, "full", 0, 0.0),
        ]
    )
    m = multipliers(ev, np.array([1, 2, 3, 4, 5, 6]), day=100, rule=rule)
    assert m[1] == pytest.approx(0.5)
    assert m[2] == pytest.approx(1.0)
    assert m[3] == pytest.approx(0.5)  # one clean rescan + one slip (counted as clean)
    assert m[4] == pytest.approx(1.0)
    assert m[5] == pytest.approx(0.25)
    assert m[6] == pytest.approx(1.0)  # no history, no trust
    partial = multipliers(ev, np.array([4]), day=100, rule=TrustRule(count_partial=True))
    assert partial[4] == pytest.approx(0.75)
    old = multipliers(ev, np.array([1]), day=10 + 400, rule=rule)  # outside the 365-day window
    assert old[1] == pytest.approx(1.0)


def test_trust_cycles(game):
    from smartshop.policy.trust import RECOMMENDED_TRUST, TrustRule, run_cycles

    _, g = game
    params = {"floor": 0.02, "cap": 0.5, "tau": 0.25}
    ev = _events([])
    for rule in (TrustRule(enabled=False), RECOMMENDED_TRUST):
        cyc, real = run_cycles(g, ev, params, rule, n_cycles=3, seed=0)
        assert len(cyc) == 3 and real.cycle.nunique() == 3
        assert real.p.between(0.02 - 1e-9, 1).all()
        assert (real.trust_mult >= 1 - rule.max_discount - 1e-9).all()
        assert not (real.found & ~real.rescanned).any()
    # without trust every multiplier is 1
    _, real0 = run_cycles(g, ev, params, TrustRule(enabled=False), n_cycles=3, seed=0)
    assert (real0.trust_mult == 1).all()


def test_size_neutral_scores_remove_the_basket_size_effect():
    from smartshop.policy.neutral import SizeNeutraliser

    rng = np.random.default_rng(0)
    units = rng.integers(1, 80, 20_000).astype(float)
    # bigger baskets score higher for everyone; within a size, some sessions are riskier
    signal = rng.normal(size=len(units))
    score = 1 / (1 + np.exp(-(-3 + 0.04 * units + signal)))
    n = SizeNeutraliser.fit(score, units)
    full = n(score, units, 1.0)
    half = n(score, units, 0.5)
    assert np.corrcoef(score, units)[0, 1] > 0.4
    assert abs(np.corrcoef(full, units)[0, 1]) < 0.05  # size no longer drives the score
    assert abs(np.corrcoef(half, units)[0, 1]) < np.corrcoef(score, units)[0, 1]
    assert np.corrcoef(full, signal)[0, 1] > 0.9  # but the within-size signal is kept
    assert np.allclose(n(score, units, 0.0), score)
    # the overall distribution is kept, so the policy's shape and volume still apply
    assert abs(np.median(full) - np.median(score)) < 0.02
    # and it survives a round trip through the bundle manifest
    again = SizeNeutraliser.from_dict(n.to_dict())
    assert np.allclose(again(score[:100], units[:100]), n(score[:100], units[:100]))
