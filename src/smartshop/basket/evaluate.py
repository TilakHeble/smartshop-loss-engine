"""Evaluate check lists on SmartShop sessions with known missing items (hidden truth).

A check list names commodities for a colleague to count against the scan list. Counting a
commodity costs the units of it in the bags (at least 1: looking for it). Every method gets the
same per-session budget of units to count, and the same detection rules as the simulator:
units missed by mistake are found at the partial-check rate; units skipped on purpose are
usually concealed outside the bags, so a partial check, random or targeted, finds them only at
the lower concealed rate. Only a full rescan empties the bags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartshop.basket.data import TEST_DAYS


def missing_truth(con) -> pd.DataFrame:
    """Per SmartShop session x commodity: missing units and value, split by cause."""
    return con.sql(
        """
        SELECT u.basket_id, p.commodity,
               SUM(u.gt_nonscan::INT) AS miss_units,
               SUM(u.unit_price * u.gt_nonscan::INT) AS miss_value,
               SUM(u.unit_price * u.gt_deliberate::INT) AS miss_deliberate,
               SUM(u.unit_price * u.gt_accidental::INT) AS miss_accidental
        FROM sim_truth.units u
        JOIN marts.dim_product p USING (product_id)
        GROUP BY 1, 2
        HAVING SUM(u.gt_nonscan::INT) > 0
        """
    ).df()


def true_units(con) -> pd.DataFrame:
    """Units physically in the bags per session x commodity: the effort of counting it."""
    return con.sql(
        """
        SELECT u.basket_id, p.commodity, COUNT(*) AS bag_units
        FROM sim_truth.units u JOIN marts.dim_product p USING (product_id)
        GROUP BY 1, 2
        """
    ).df()


def missing_sessions_in_test_period(con) -> np.ndarray:
    lo, hi = TEST_DAYS
    return (
        con.sql(
            f"""
            SELECT s.basket_id FROM sim_obs.sessions s
            JOIN sim_truth.sessions t USING (basket_id)
            WHERE s.day BETWEEN {lo} AND {hi} AND t.gt_nonscan_units > 0
            """
        )
        .df()
        .basket_id.to_numpy()
    )


def rankers(s: pd.DataFrame) -> dict[str, np.ndarray]:
    """Priority per (session, commodity) for each way of choosing what to count."""
    # Before checking, the colleague knows what was scanned, not what is in the bags.
    exp_count = np.maximum(s.vis_units, 1) + s.p_missing
    return {
        "Basket model: value per unit counted": (s.exp_value / exp_count).to_numpy(),
        "Basket model: expected value": s.exp_value.to_numpy(),
        "Most valuable scanned items": np.where(s.is_visible > 0, s.vis_units * s.unit_price, -1),
        "Member's usual items": s.m_rate.fillna(0).to_numpy(),
        "Most popular items": s.popularity.to_numpy(),
    }


def greedy_fill(b: pd.DataFrame, budget: float, max_candidates: int = 40) -> pd.DataFrame:
    """Per session, walk down the list by score and count each commodity that still fits the
    budget (skipping ones that do not), as a colleague with a fixed time would."""
    x = b.sort_values(["basket_id", "score"], ascending=[True, False])
    x = x[x.groupby("basket_id").cumcount() < max_candidates]
    ids = x.basket_id.to_numpy()
    cost = x.cost.to_numpy()
    keep = np.zeros(len(x), dtype=bool)
    left, current = budget, None
    for i in range(len(x)):
        if ids[i] != current:
            current, left = ids[i], budget
        if cost[i] <= left:
            keep[i] = True
            left -= cost[i]
    return x[keep]


def evaluate_budget(scored, truth, bags, cfg, budgets=(3, 5, 10)) -> pd.DataFrame:
    """Share of missing value found with at most `budget` units counted per session."""
    sessions = scored.basket_id.unique()
    t = truth[truth.basket_id.isin(sessions)]
    tot = t[["miss_value", "miss_accidental", "miss_deliberate"]].sum()
    b = scored.merge(t, on=["basket_id", "commodity"], how="left")
    b = b.merge(bags, on=["basket_id", "commodity"], how="left")
    b = b.fillna({"miss_accidental": 0, "miss_deliberate": 0, "bag_units": 0})
    b["cost"] = np.maximum(b.bag_units, 1)
    d_acc, d_del = cfg.partial_detect_nonscan, cfg.partial_detect_deliberate
    n_bag = bags[bags.basket_id.isin(sessions)].groupby("basket_id").bag_units.sum()

    def row(method, budget, units, acc, dlb):
        return {
            "method": method,
            "budget": budget,
            "units_counted": units,
            "accidental": d_acc * acc / tot.miss_accidental,
            "deliberate": d_del * dlb / tot.miss_deliberate,
            "value_found": (d_acc * acc + d_del * dlb) / tot.miss_value,
        }

    rows = []
    for budget in budgets:
        # Today: `budget` random units from the bags (expected values, exact).
        frac = np.minimum(1.0, budget / t.basket_id.map(n_bag))
        units = float(np.minimum(budget, n_bag).mean())
        rows.append(
            row(
                "Today: random units",
                budget,
                units,
                (t.miss_accidental * frac).sum(),
                (t.miss_deliberate * frac).sum(),
            )
        )
        for name, score in rankers(b).items():
            sel = greedy_fill(b.assign(score=score), budget)
            used = sel.groupby("basket_id").bag_units.sum().reindex(n_bag.index, fill_value=0)
            rows.append(
                row(
                    name,
                    budget,
                    float(used.mean()),
                    sel.miss_accidental.sum(),
                    sel.miss_deliberate.sum(),
                )
            )
    return pd.DataFrame(rows)
