"""Loss estimators and stress-test definitions (runs in CI)."""

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from smartshop.experiments.measure import estimates
from smartshop.experiments.sensitivity import BASE, scenarios


def _realised(p, found_value, net_loss, u):
    return pd.DataFrame(
        {
            "p": p,
            "u": u,
            "rescanned": u < p,
            "found_value": np.where(u < p, found_value, 0.0),
            "net_loss": net_loss,
        }
    )


def test_estimators_agree_when_everyone_is_rescanned():
    n = 50
    loss = np.linspace(0, 10, n)
    r = _realised(np.ones(n), loss, loss, np.full(n, 0.5))
    e = estimates(r, floor=1.0, detect=1.0)
    assert e["inverse-probability weighted"] == pytest.approx(loss.sum())
    assert e["naive (average of rescans x sessions)"] == pytest.approx(loss.sum())
    assert e["truth"] == pytest.approx(loss.sum())


def test_weighting_removes_the_bias_of_targeted_rescans():
    rng = np.random.default_rng(0)
    n = 20_000
    loss = np.where(rng.random(n) < 0.05, rng.uniform(5, 20, n), 0.0)
    # risky sessions (those with loss) are much more likely to be rescanned
    p = np.where(loss > 0, 0.5, 0.02)
    ht, naive, floor_only = [], [], []
    for seed in range(40):
        u = np.random.default_rng(1000 + seed).random(n)  # independent of the losses
        e = estimates(_realised(p, loss, loss, u), floor=0.02, detect=1.0)
        ht.append(e["inverse-probability weighted"])
        naive.append(e["naive (average of rescans x sessions)"])
        floor_only.append(e["random floor only"])
    truth = loss.sum()
    assert np.mean(ht) == pytest.approx(truth, rel=0.03)
    assert np.mean(floor_only) == pytest.approx(truth, rel=0.08)
    assert np.mean(naive) > 3 * truth


def test_each_scenario_changes_something():
    base = {f.name: getattr(BASE, f.name) for f in fields(BASE)}
    for name, cfg in scenarios().items():
        changed = [k for k, v in base.items() if getattr(cfg, k) != v]
        if name == "Base assumptions":
            assert changed == []
        else:
            assert changed, name
