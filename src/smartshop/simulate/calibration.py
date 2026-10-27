"""Compare a simulation with the published targets (simulate.config.TARGETS)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartshop.simulate.config import TARGETS


def error_curve(truth: pd.DataFrame, sizes=(20, 50, 100), window: float = 0.25) -> pd.Series:
    """P(at least one error) for baskets near each size (+/- 25% of units)."""
    out = {}
    for n in sizes:
        near = truth[truth.gt_true_units.between(n * (1 - window), n * (1 + window))]
        out[n] = near.gt_has_error.mean() if len(near) else np.nan
    return pd.Series(out, name="p_error")


def calibration_table(truth: pd.DataFrame, rescans: pd.DataFrame) -> pd.DataFrame:
    curve = error_curve(truth)
    item_errors = truth.gt_nonscan_units.sum() + truth.gt_overscan_units.sum()
    partial = rescans[rescans.rescan_type == "partial"]
    full = rescans[rescans.rescan_type == "full"]
    rows = [
        ("P(error), ~20 units", curve[20], TARGETS["error_prob_20_items"]),
        ("P(error), ~50 units", curve[50], TARGETS["error_prob_50_items"]),
        ("P(error), ~100 units", curve[100], TARGETS["error_prob_100_items"]),
        (
            "Non-scans as share of item errors",
            truth.gt_nonscan_units.sum() / item_errors,
            TARGETS["nonscan_share_of_item_errors"],
        ),
        (
            "Deliberate share of non-scanned value",
            truth.gt_deliberate_value.sum() / truth.gt_nonscan_value.sum(),
            TARGETS["malicious_share_of_loss"],
        ),
        (
            "Net loss, share of sales",
            truth.gt_net_loss.sum() / truth.gt_true_value.sum(),
            TARGETS["net_loss_share_of_sales"],
        ),
        ("Rescan rate", len(rescans) / len(truth), TARGETS["audit_rate"]),
        (
            "Partial rescans finding an error",
            partial.found_error.mean(),
            TARGETS["partial_rescan_hit_rate"],
        ),
        ("Full rescans finding an error", full.found_error.mean(), None),
    ]
    df = pd.DataFrame(rows, columns=["check", "simulated", "published"])
    return df.set_index("check")
