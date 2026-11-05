"""What a rescan finds, and how long it takes. Used for today's random rescans (week 3) and for
every policy evaluated later, so all policies face the same detection model.

    full     every unit is rescanned: finds non-scans (and over-scans, which are refunded)
    partial  the colleague rescans a share of the units (about 20-25% of the basket, at least
             2) picked from the bags: finds a non-scan only if it is among them, and cannot see
             over-scans [ECR18]. If it finds anything, the check device suggests a full check of
             the rest
             ("escalated"); an escalated check is not a random full rescan, so it is never used
             as a training label
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NONE, PARTIAL, FULL = "none", "partial", "full"


def partial_count(n, cfg):
    """Units a partial check covers in a basket of n: a share, at least a minimum, at most n."""
    n = np.asarray(n)
    share = cfg.partial_share
    bands = getattr(cfg, "partial_bands", None)
    if bands:
        limits = [b[0] for b in bands[:-1]]
        share = np.asarray([b[1] for b in bands])[np.searchsorted(limits, n, side="left")]
    return np.minimum(n, np.maximum(cfg.partial_min_items, np.ceil(share * n)))


def rescan_outcomes(basket_ids: np.ndarray, decisions: np.ndarray, units: pd.DataFrame, cfg):
    """Rescan results for the sessions that were rescanned (observable: what was found).

    units must carry basket_id, unit_price, loss_price, gt_nonscan, gt_deliberate, gt_switched,
    gt_overscan, u_pick, u_det.
    """
    dec = pd.Series(decisions, index=basket_ids, name="rescan_type")
    dec = dec[dec != NONE]
    u = units[units.basket_id.isin(dec.index)].copy()
    u["rescan_type"] = u.basket_id.map(dec)
    u["pick_rank"] = u.groupby("basket_id").u_pick.rank(method="first")
    n = u.groupby("basket_id").basket_id.transform("size")
    full = (u.rescan_type == FULL).to_numpy()
    picked = full | (u.pick_rank <= partial_count(n, cfg)).to_numpy()
    # concealed = skipped on purpose and hidden; a switched barcode is in the bags
    concealed = (u.gt_deliberate & ~u.gt_switched).to_numpy()
    p_ns = np.where(
        full,
        cfg.full_detect_nonscan,
        np.where(concealed, cfg.partial_detect_deliberate, cfg.partial_detect_nonscan),
    )
    nonscan, det = u.gt_nonscan.to_numpy(), u.u_det.to_numpy()
    found_first = picked & nonscan & (det < p_ns)
    # a partial check that finds anything escalates to a full check of the whole basket
    hit = pd.Series(found_first & ~full).groupby(u.basket_id.to_numpy()).transform("any")
    escalated = (~full) & hit.to_numpy() & cfg.escalate_partial
    found_full_stage = escalated & nonscan & (det < cfg.full_detect_nonscan)
    u["found_missing"] = found_first | found_full_stage
    u["found_over"] = (
        (full | escalated) & u.gt_overscan.to_numpy() & (det < cfg.full_detect_overscan)
    )
    u["checked"] = picked
    u["escalated"] = escalated

    out = (
        u.assign(
            found_missing_value=np.where(u.found_missing, u.loss_price, 0.0),
            found_over_value=np.where(u.found_over, u.unit_price, 0.0),
            full_stage_units=escalated,
        )
        .groupby("basket_id")
        .agg(
            rescan_type=("rescan_type", "first"),
            escalated=("escalated", "any"),
            units_checked=("checked", "sum"),
            full_stage_units=("full_stage_units", "sum"),
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
    out["rescan_seconds"] = (
        cfg.rescan_fixed_seconds
        + per_unit * out.units_checked
        + cfg.full_seconds_per_unit * out.full_stage_units
    )
    out = out.drop(columns="full_stage_units")
    out["found_error"] = (out.found_missing_units + out.found_over_units) > 0
    return out
