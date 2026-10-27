"""What a rescan finds, and how long it takes. Used for today's random rescans (week 3) and for
every policy evaluated later, so all policies face the same detection model.

    full     every unit is rescanned: finds non-scans (and over-scans, which are refunded)
    partial  the colleague rescans k units picked at random from the bags: finds a non-scan
             only if it is among them, and cannot see over-scans (a double scan needs the full
             list) [ECR18]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NONE, PARTIAL, FULL = "none", "partial", "full"


def rescan_outcomes(basket_ids: np.ndarray, decisions: np.ndarray, units: pd.DataFrame, cfg):
    """Rescan results for the sessions that were rescanned (observable: what was found).

    units must carry basket_id, unit_price, gt_nonscan, gt_deliberate, gt_overscan, u_pick, u_det.
    """
    dec = pd.Series(decisions, index=basket_ids, name="rescan_type")
    dec = dec[dec != NONE]
    u = units[units.basket_id.isin(dec.index)].copy()
    u["rescan_type"] = u.basket_id.map(dec)
    u["pick_rank"] = u.groupby("basket_id").u_pick.rank(method="first")
    full = u.rescan_type == FULL
    checked = full | (u.pick_rank <= cfg.partial_items)
    p_ns = np.where(
        full,
        cfg.full_detect_nonscan,
        np.where(u.gt_deliberate, cfg.partial_detect_deliberate, cfg.partial_detect_nonscan),
    )
    u["found_missing"] = checked & u.gt_nonscan & (u.u_det < p_ns)
    u["found_over"] = full & u.gt_overscan & (u.u_det < cfg.full_detect_overscan)
    u["checked"] = checked

    out = (
        u.assign(
            found_missing_value=np.where(u.found_missing, u.unit_price, 0.0),
            found_over_value=np.where(u.found_over, u.unit_price, 0.0),
        )
        .groupby("basket_id")
        .agg(
            rescan_type=("rescan_type", "first"),
            units_checked=("checked", "sum"),
            found_missing_units=("found_missing", "sum"),
            found_missing_value=("found_missing_value", "sum"),
            found_over_units=("found_over", "sum"),
            found_over_value=("found_over_value", "sum"),
        )
        .reset_index()
    )
    per_unit = np.where(
        out.rescan_type == FULL, cfg.full_seconds_per_unit, cfg.partial_seconds_per_unit
    )
    out["rescan_seconds"] = cfg.rescan_fixed_seconds + per_unit * out.units_checked
    out["found_error"] = (out.found_missing_units + out.found_over_units) > 0
    return out
