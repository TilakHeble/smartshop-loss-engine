"""Session features at the exit, from the scan log the app or handset sends.

The same definitions as the warehouse SQL (risk.features.session_sql), computed in pandas for one
session at a time; a test checks the two agree on every session of a simulated warehouse.
Member history comes from the weekly bundle's member profile; a member with no profile (a new
SmartShop user) gets typical values, so no history never raises the score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartshop.risk.features import HISTORY_COLUMNS, derive

SESSION_COLUMNS = [
    "scanned_units",
    "scanned_value",
    "hr_units",
    "hr_value",
    "n_products",
    "n_deletes",
    "max_gap_min",
    "gap_cv",
    "first_scan_frac",
    "late_share",
    "repeat_scans",
]


def session_features(events: pd.DataFrame, duration_min: float) -> dict:
    """One session's scan-log features.

    events: rows in event order with columns t (seconds from start), action ('scan'/'delete'),
    product_id, unit_price, is_high_risk.
    """
    if events.empty:
        return {c: 0.0 for c in SESSION_COLUMNS} | {
            "max_gap_min": np.nan,
            "gap_cv": np.nan,
            "first_scan_frac": np.nan,
        }
    sign = np.where(events.action.to_numpy() == "scan", 1.0, -1.0)
    price = events.unit_price.to_numpy(dtype=float)
    hr = events.is_high_risk.to_numpy(dtype=bool)
    t = events.t.to_numpy(dtype=float)
    scans = events.action.to_numpy() == "scan"
    gaps = np.diff(t)
    late = (scans & (t > 0.8 * duration_min * 60)).sum()
    n_scans = scans.sum()
    per_product = events[scans].groupby("product_id").size()
    return {
        "scanned_units": float(sign.sum()),
        "scanned_value": float((sign * price).sum()),
        "hr_units": float((sign * hr).sum()),
        "hr_value": float((sign * price * hr).sum()),
        "n_products": float(events.product_id.nunique()),
        "n_deletes": float((~scans).sum()),
        "max_gap_min": float(gaps.max() / 60) if len(gaps) else np.nan,
        "gap_cv": float(gaps.std(ddof=1) / gaps.mean())
        if len(gaps) > 1 and gaps.mean() != 0
        else np.nan,
        "first_scan_frac": float(t.min() / (duration_min * 60)),
        "late_share": float(late / n_scans) if n_scans else 0.0,
        "repeat_scans": float((per_product > 1).sum()),
    }


def feature_row(session: dict, profile: dict | None, day: int = 0) -> pd.DataFrame:
    """Full feature row for scoring: session features + member profile + derived ratios."""
    row = {**session, "duration_min": session["duration_min"], "day": day, "basket_id": 0}
    for c in HISTORY_COLUMNS:
        row[c] = (profile or {}).get(c, np.nan)
    for c in ["prior_found", "prior_rescans", "prior_full_rescans", "prior_full_found"]:
        row[c] = 0.0 if row[c] is None or pd.isna(row[c]) else row[c]
    df = pd.DataFrame([row])
    for c in ["hour_of_day", "channel", "member_id", "store_id", "trip_order", "is_peak", "is_app"]:
        if c not in df:
            df[c] = 0
    return derive(df)
