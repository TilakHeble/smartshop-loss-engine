"""Week 10 reporting: fairness check, trial sizing, the dashboard on its snapshot."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartshop.reporting.fairness import ATTRIBUTES, SIZE_BANDS, fairness_table
from smartshop.reporting.trial import measurement_cv, sizing_table, stores_per_arm

ROOT = Path(__file__).resolve().parents[1]


def _sessions(n_members=400, per_member=10, gap=0.0, seed=0):
    """Honest sessions; group 'B' members get `gap` added to their rescan probability."""
    rng = np.random.default_rng(seed)
    members = np.repeat(np.arange(n_members), per_member)
    group = np.where(members % 2 == 0, "A", "B")
    units = rng.integers(1, 50, len(members))
    p = 0.03 + 0.01 * rng.random(len(members)) + np.where(group == "B", gap, 0.0)
    df = pd.DataFrame({"member_id": members, "p": p, "honest": True, "scanned_units": units})
    for col in ATTRIBUTES.values():
        df[col] = group
    df["size_band"] = pd.cut(df.scanned_units, SIZE_BANDS, right=False)
    return df


def test_equal_rates_pass():
    t = fairness_table(_sessions(), today_rate=0.12, n_boot=60)
    assert (t["four-fifths ratio"] > 0.9).all()
    assert (t.verdict == "pass").all()


def test_a_real_gap_is_flagged_and_not_called_chance():
    t = fairness_table(_sessions(gap=0.03), today_rate=0.12, n_boot=60)
    assert (t["four-fifths ratio"] < 0.8).all()
    assert (t["chance of a gap this large"] < 0.05).all()
    assert (t.verdict == "review").all()


def test_gap_from_basket_size_is_recognised():
    df = _sessions()
    # group B buys bigger baskets, and bigger baskets get rescanned more, for everyone
    df["scanned_units"] = np.where(df.channel == "B", df.scanned_units + 30, df.scanned_units)
    df["size_band"] = pd.cut(df.scanned_units, SIZE_BANDS, right=False)
    df["p"] = 0.02 + 0.002 * df.scanned_units.clip(upper=45)
    t = fairness_table(df, today_rate=0.12, n_boot=60)
    assert t.loc["channel", "four-fifths ratio"] < 0.8
    assert t.loc["channel", "ratio, same basket sizes"] > t.loc["channel", "four-fifths ratio"]


def test_stores_per_arm_matches_the_textbook_formula():
    # two-sample comparison of means: n = 2 (z_0.975 + z_0.8)^2 sigma^2 / delta^2
    expected = math.ceil(2 * (1.959964 + 0.841621) ** 2 * 0.3**2 / 0.12**2)
    assert stores_per_arm(0.12, 0.3, 0.0) == expected
    # a correlated baseline cuts the variance by (1 - rho^2)
    assert stores_per_arm(0.12, 0.3, 0.7) < stores_per_arm(0.12, 0.3, 0.0) * 0.52


def test_measurement_error_scales_with_sessions():
    assert measurement_cv(40_000, 0.15, 40_000) == pytest.approx(0.15)
    assert measurement_cv(10_000, 0.15, 40_000) == pytest.approx(0.30)
    table = sizing_table(0.12, 0.15, 47_000)
    assert (table.diff().dropna() >= 0).all().all()  # more spread, more stores


def test_snapshot_is_aggregate_only_and_strict_json():
    path = ROOT / "dashboards" / "snapshot" / "snapshot.json"
    text = path.read_text()

    def reject(c):
        raise AssertionError(f"non-standard JSON constant {c}")

    snap = json.loads(text, parse_constant=reject)
    for key in ("member_id", "basket_id", "household_key"):
        assert key not in text
    assert {"headline", "fairness", "measurement", "trial"} <= set(snap)


def test_dashboard_runs_on_the_snapshot():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboards" / "app.py"), default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Loss remaining" in labels and "Honest shoppers stopped" in labels
