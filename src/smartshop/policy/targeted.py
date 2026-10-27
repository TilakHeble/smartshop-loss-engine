"""Are quick targeted checks worth colleague time? And would an in-app prompt help?

Week 4 built check lists from the basket model. Here they compete for colleague time with full
rescans, session by session, at the risk bands where the policy makes its choices:

    full rescan      finds missing items in the bags and (mostly) concealed ones
    targeted check   counts the basket model's top product types (3-unit budget, as in week 4);
                     finds honest mistakes in them, rarely concealed items

Value per colleague-minute decides. An in-app prompt ("did you forget...?") costs no colleague
time, so it is judged instead by how often it is right and how much it could reach.

Approximation, stated: lists and missing items use each session's simulated scan under today's
policy (shoppers' response to the new policy changes few lists: only strategic sessions differ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartshop.basket.evaluate import greedy_fill, missing_truth, true_units
from smartshop.basket.model import score_sessions


def band_yields(con, basket_model, bands: dict[str, pd.DataFrame], cfg, budget_units=3, seed=0):
    """Recovered value per colleague-minute of full vs targeted checks, per risk band.

    Each band frame needs basket_id, full_find (expected net recovery of a full rescan) and
    full_cost_min (its duration in minutes).
    """
    truth, bags = missing_truth(con), true_units(con)
    rows = []
    for name, d in bands.items():
        sc = score_sessions(con, basket_model, d.basket_id)
        x = sc.merge(truth, on=["basket_id", "commodity"], how="left")
        x = x.merge(bags, on=["basket_id", "commodity"], how="left")
        x = x.fillna({"miss_accidental": 0, "miss_deliberate": 0, "bag_units": 0})
        x["cost"] = np.maximum(x.bag_units, 1)
        x["score"] = x.exp_value / (np.maximum(x.vis_units, 1) + x.p_missing)
        sel = greedy_fill(x, budget_units)
        g = sel.groupby("basket_id")
        found = (
            cfg.partial_detect_nonscan * g.miss_accidental.sum()
            + cfg.partial_detect_deliberate * g.miss_deliberate.sum()
        ).reindex(d.basket_id, fill_value=0)
        units = g.bag_units.sum().reindex(d.basket_id, fill_value=0)
        t_min = (cfg.rescan_fixed_seconds + cfg.partial_seconds_per_unit * units) / 60
        rows.append(
            {
                "band": name,
                "sessions": len(d),
                "full rescan: $ per minute": d.full_find.sum() / d.full_cost_min.sum(),
                "targeted check: $ per minute": found.sum() / t_min.sum(),
                "full rescan: minutes": d.full_cost_min.mean(),
                "targeted check: minutes": t_min.mean(),
            }
        )
    return pd.DataFrame(rows).set_index("band")


def prompt_reach(con, basket_model, basket_ids, thresholds=(0.2, 0.3, 0.4, 0.5)):
    """If the app asked 'did you forget...?' about its top suspected product type, when the
    model is at least `threshold` sure: how many shoppers are asked, how often it is right, and
    how much of the value missed by mistake it points at (before anyone acts on it)."""
    sc = score_sessions(con, basket_model, basket_ids)
    tr = missing_truth(con)
    x = sc.merge(tr, on=["basket_id", "commodity"], how="left")
    x = x.fillna({"miss_accidental": 0, "miss_units": 0})
    top = x.sort_values(["basket_id", "p_missing"], ascending=[True, False])
    top = top.groupby("basket_id").head(1)
    acc = con.sql("SELECT basket_id, gt_accidental_value FROM sim_truth.sessions").df()
    total_acc = acc[acc.basket_id.isin(basket_ids)].gt_accidental_value.sum()
    rows = []
    for thr in thresholds:
        asked = top[top.p_missing >= thr]
        rows.append(
            {
                "model at least this sure": thr,
                "shoppers asked": len(asked) / len(basket_ids),
                "prompts that are right": (asked.miss_units > 0).mean(),
                "mistake value pointed at": asked.miss_accidental.sum() / total_acc,
            }
        )
    return pd.DataFrame(rows).set_index("model at least this sure")
